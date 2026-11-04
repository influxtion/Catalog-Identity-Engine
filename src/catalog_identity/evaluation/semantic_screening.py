"""Separately registered nine-fit semantic experiment; training file only."""

from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import average_precision_score

from catalog_identity.evaluation.provenance import implementation_manifest
from catalog_identity.evaluation.runner import file_hash, write_json
from catalog_identity.evaluation.screening import entity_folds, envelope
from catalog_identity.ingestion.loaders import load_wdc_pairs
from catalog_identity.matching.learned import LearnedMatcher
from catalog_identity.matching.semantic_features import DEFAULT_CACHE, EmbeddingCache

CANDIDATES = (("v3-control", "3"), ("v5-summaries", "5"), ("v6-interactions", "6"))


def select_semantic_candidate(candidates):
    control = candidates[0]
    eligible = []
    for candidate in candidates[1:]:
        improved = int(
            sum(
                new["envelopes"]["0.99"]["weighted_recall"]
                > old["envelopes"]["0.99"]["weighted_recall"]
                for new, old in zip(candidate["folds"], control["folds"], strict=True)
            )
        )
        candidate["folds_improved_at_99"] = improved
        candidate["qualifies"] = bool(
            candidate["supported_folds_99"] == 3
            and improved >= 2
            and candidate["mean_weighted_recall_99"] >= control["mean_weighted_recall_99"] + 0.02
            and candidate["mean_weighted_average_precision"]
            >= 0.99 * control["mean_weighted_average_precision"]
        )
        if candidate["qualifies"]:
            eligible.append(candidate)
    if not eligible:
        return control["name"]
    return max(
        eligible,
        key=lambda candidate: (
            candidate["mean_weighted_recall_99"],
            candidate["mean_weighted_average_precision"],
            -candidates.index(candidate),
        ),
    )["name"]


def screen_semantic(config, output: Path, cache_directory=DEFAULT_CACHE, embedding_cache=None):
    if output.exists():
        raise ValueError("output already exists")
    if config.seed != 42:
        raise ValueError("registered semantic screening requires seed 42")
    train = load_wdc_pairs(config.train, "train")
    folds = entity_folds(train, config.seed)
    cache = embedding_cache if embedding_cache is not None else EmbeddingCache(cache_directory)
    output.mkdir(parents=True)
    started = perf_counter()
    population = cache.prepare(train.listings)
    preprocessing_seconds = perf_counter() - started
    report = {
        "phase": "train_only_frozen_semantic_screening",
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
        "encoder": cache.metadata,
        "encoding_runtime": getattr(cache.encoder, "runtime", {}),
        "training_embedding_population": population,
        "precompute_seconds": preprocessing_seconds,
        "warning": "Previously exposed training folds are reused for development selection. "
        "Envelopes select thresholds on their reported fold; not independent policies "
        "or precision guarantees. Fold scores are never pooled. Assessment weighting "
        "corrects expected negative retention; fitting restrictions alter class balance. "
        "Frozen pretrained embeddings fit no corpus statistics or labels.",
        "candidates": [],
    }
    for name, version in CANDIDATES:
        candidate = {"name": name, "folds": []}
        for index, (fit, assess, excluded) in enumerate(folds):
            started = perf_counter()
            model = LearnedMatcher(
                feature_version=version,
                num_leaves=15,
                n_estimators=200,
                seed=config.seed,
                embedding_cache=cache,
            ).fit(fit)
            fitted = perf_counter()
            scores = model.predict(assess, calibrated=False)
            labels = np.array([p.label for p in assess.pairs])
            weights = np.where(labels == 1, 1.0, 3.0)
            candidate["settings"] = model.settings
            candidate["effective_parameters"] = model.booster.params
            candidate["feature_count"] = len(model.features.names)
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
                    "assessment_positive_entities": len(
                        {assess.entity_ids[p.left] for p in assess.pairs if p.label}
                    ),
                    "train_assessment_entity_overlap": 0,
                    "raw_average_precision": float(average_precision_score(labels, scores)),
                    "weighted_average_precision": float(
                        average_precision_score(labels, scores, sample_weight=weights)
                    ),
                    "envelopes": {
                        str(target): envelope(assess, scores, target, weights)
                        for target in (0.95, 0.98, 0.99)
                    },
                    "fit_with_features_seconds": fitted - started,
                    "assessment_with_features_seconds": perf_counter() - fitted,
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
        report["embedding_cache_stats"] = dict(cache.stats)
        write_json(output / "screening.json", report)
    report["selected_candidate"] = select_semantic_candidate(report["candidates"])
    report["official_validation_authorized_by_selection"] = (
        report["selected_candidate"] != CANDIDATES[0][0]
    )
    write_json(output / "screening.json", report)
    return report
