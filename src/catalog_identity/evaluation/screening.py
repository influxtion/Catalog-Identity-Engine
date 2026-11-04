"""Fixed train-only entity-disjoint matcher screening; never reads validation/test."""

import hashlib
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import average_precision_score

from catalog_identity.evaluation.metrics import classification_metrics
from catalog_identity.evaluation.provenance import implementation_manifest
from catalog_identity.evaluation.runner import file_hash, write_json
from catalog_identity.ingestion.loaders import BenchmarkSplit, load_wdc_pairs
from catalog_identity.matching.baseline import select_match_threshold
from catalog_identity.matching.learned import LearnedMatcher

CANDIDATES = (
    ("v3-small", "3", 15, 200),
    ("v4-small", "4", 15, 200),
    ("v3-large", "3", 31, 400),
    ("v4-large", "4", 31, 400),
)


def entity_folds(split, seed=42):
    if split.name != "train" or set(split.entity_ids) != set(split.listings):
        raise ValueError("screening requires complete training entity metadata, without extra keys")
    if any(not isinstance(e, str) or not e for e in split.entity_ids.values()):
        raise ValueError("entity identifiers must be nonempty strings")
    buckets = {
        k: int(hashlib.sha256(f"{seed}:{e}".encode()).hexdigest(), 16) % 3
        for k, e in split.entity_ids.items()
    }
    folds = []
    for bucket in range(3):
        parts = []
        for held in (False, True):
            listings = {k: v for k, v in split.listings.items() if (buckets[k] == bucket) == held}
            pairs = tuple(p for p in split.pairs if p.left in listings and p.right in listings)
            parts.append(
                BenchmarkSplit(
                    "train" if not held else "validation",
                    listings,
                    pairs,
                    {k: split.entity_ids[k] for k in listings},
                    split.identity_level,
                )
            )
        fit, assess = parts
        if any({p.label for p in part.pairs} != {0, 1} for part in parts):
            raise ValueError("each training and assessment fold needs both classes")
        folds.append((fit, assess, len(split.pairs) - len(fit.pairs) - len(assess.pairs)))
    return folds


def envelope(split, scores, target, weights):
    labels = np.array([p.label for p in split.pairs])
    selected = select_match_threshold(labels, scores, target, 5, weights)
    predicted = (
        scores >= selected.threshold
        if selected.threshold is not None
        else np.zeros(len(scores), bool)
    )
    raw = classification_metrics(labels, predicted)
    weighted_counts = {
        name: float(weights[mask].sum())
        for name, mask in (
            ("true_positives", (labels == 1) & predicted),
            ("false_positives", (labels == 0) & predicted),
            ("false_negatives", (labels == 1) & ~predicted),
            ("true_negatives", (labels == 0) & ~predicted),
        )
    }
    return {
        "selection": asdict(selected),
        "raw": raw,
        "weighted_counts": weighted_counts,
        "weighted_precision": selected.validation_precision,
        "weighted_recall": selected.validation_recall,
        "distinct_matched_positive_entities": len(
            {
                split.entity_ids[p.left]
                for i, p in enumerate(split.pairs)
                if predicted[i] and p.label
            }
        ),
        "threshold_selected_on_reported_population": True,
        "precision_status": "empirical; not an established precision guarantee",
    }


def select_candidate(candidates):
    baseline_ap = candidates[0]["mean_weighted_average_precision"]
    eligible = [
        r
        for r in candidates
        if r["supported_folds_99"] >= 2
        and r["mean_weighted_average_precision"] >= 0.95 * baseline_ap
    ]
    if not eligible:
        return candidates[0]["name"]
    return max(
        eligible,
        key=lambda r: (
            r["mean_weighted_recall_99"],
            r["supported_folds_99"],
            r["mean_weighted_average_precision"],
            -candidates.index(r),
        ),
    )["name"]


def screen(config, output: Path):
    if output.exists():
        raise ValueError("output already exists")
    if config.seed != 42:
        raise ValueError("registered screening matrix requires seed 42")
    train = load_wdc_pairs(config.train, "train")
    folds = entity_folds(train, config.seed)
    output.mkdir(parents=True)
    report = {
        "phase": "train_only_entity_disjoint_screening",
        "development_only": True,
        "validation_opened": False,
        "test_opened": False,
        "synthetic": config.synthetic,
        "dataset_name": config.dataset_name,
        "dataset_version": config.dataset_version,
        "protocol": config.protocol.model_dump() if config.protocol else None,
        "assessment_weights": {"positive": 1, "negative": 3},
        "seed": config.seed,
        "inputs": {"train": file_hash(config.train)},
        "implementation": implementation_manifest(),
        "warning": "Within-fold selected envelopes are optimistic diagnostics, "
        "not independent policies. "
        "Fold raw scores are not pooled. Weight 3 for negatives corrects expected assessment "
        "retention; restricted training folds also change class balance. This is approximate "
        "training-distribution selection, not a target-domain guarantee.",
        "candidates": [],
    }
    for name, version, leaves, trees in CANDIDATES:
        candidate = {"name": name, "folds": []}
        for index, (fit, assess, excluded) in enumerate(folds):
            started = perf_counter()
            model = LearnedMatcher(
                feature_version=version, num_leaves=leaves, n_estimators=trees, seed=config.seed
            ).fit(fit)
            scores = model.predict(assess, calibrated=False)
            labels = np.array([p.label for p in assess.pairs])
            weights = np.where(labels == 1, 1.0, 3.0)
            candidate["settings"] = model.settings
            candidate["effective_parameters"] = model.booster.params
            positives = {assess.entity_ids[p.left] for p in assess.pairs if p.label}
            candidate["folds"].append(
                {
                    "fold": index,
                    "train_pairs": len(fit.pairs),
                    "train_positives": sum(p.label for p in fit.pairs),
                    "assessment_pairs": len(assess.pairs),
                    "assessment_positives": int(labels.sum()),
                    "boundary_pairs_excluded": excluded,
                    "train_listings": len(fit.listings),
                    "assessment_listings": len(assess.listings),
                    "train_entities": len(set(fit.entity_ids.values())),
                    "assessment_entities": len(set(assess.entity_ids.values())),
                    "assessment_positive_entities": len(positives),
                    "train_assessment_entity_overlap": 0,
                    "raw_average_precision": float(average_precision_score(labels, scores)),
                    "weighted_average_precision": float(
                        average_precision_score(labels, scores, sample_weight=weights)
                    ),
                    "envelopes": {
                        str(target): envelope(assess, scores, target, weights)
                        for target in (0.95, 0.98, 0.99)
                    },
                    "elapsed_seconds": perf_counter() - started,
                }
            )
            print(f"{name} fold {index + 1}/3 complete", flush=True)
        candidate["mean_weighted_average_precision"] = float(
            np.mean([f["weighted_average_precision"] for f in candidate["folds"]])
        )
        candidate["mean_weighted_recall_99"] = float(
            np.mean([f["envelopes"]["0.99"]["weighted_recall"] for f in candidate["folds"]])
        )
        candidate["supported_folds_99"] = sum(
            f["envelopes"]["0.99"]["selection"]["threshold"] is not None for f in candidate["folds"]
        )
        report["candidates"].append(candidate)
        write_json(output / "screening.json", report)
    report["selected_candidate"] = select_candidate(report["candidates"])
    write_json(output / "screening.json", report)
    return report
