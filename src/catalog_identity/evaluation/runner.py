import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from catalog_identity import __version__
from catalog_identity.evaluation.config import RunConfig
from catalog_identity.evaluation.metrics import classification_metrics, probability_metrics
from catalog_identity.ingestion.audit import audit_splits, profile
from catalog_identity.ingestion.loaders import BenchmarkSplit, load_wdc_pairs
from catalog_identity.matching.baseline import LogisticMatcher, decision, select_match_threshold
from catalog_identity.matching.features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    conflicts,
    pair_features,
)
from catalog_identity.normalization import NORMALIZATION_VERSION, normalize
from catalog_identity.retrieval import CandidateIndex


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def git_metadata(root: Path) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

    return {"commit": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--porcelain")}


def retrieval_evaluation(
    index: CandidateIndex,
    split: BenchmarkSplit,
    ks: list[int],
    output: Path,
) -> tuple[dict, dict[str, dict[str, int]]]:
    normalized = {key: normalize(x) for key, x in split.listings.items()}
    started = perf_counter()
    index.build(list(normalized.values()))
    build_seconds = perf_counter() - started
    known_positives: dict[str, set[str]] = defaultdict(set)
    for pair in split.pairs:
        if pair.label:
            known_positives[pair.left].add(pair.right)
            known_positives[pair.right].add(pair.left)
    labeled_denominator = sum(map(len, known_positives.values()))
    entity_counts = Counter(split.entity_ids.values())
    complete_entities = len(split.entity_ids) == len(split.listings)
    entity_denominator = sum(n * (n - 1) for n in entity_counts.values())
    labeled_hits = Counter()
    entity_hits = Counter()
    ranks: dict[str, dict[str, int]] = {}
    latency_ms = []
    channel_counts = Counter()
    with output.open("w", encoding="utf-8") as stream:
        for key in sorted(normalized):
            started = perf_counter()
            candidates = index.query(normalized[key], max(ks))
            latency_ms.append((perf_counter() - started) * 1000)
            ranks[key] = {c.listing_id: c.rank for c in candidates}
            for candidate in candidates:
                stream.write(json.dumps({"query_id": key, **asdict(candidate)}) + "\n")
                channel_counts.update(candidate.sources)
            for k in ks:
                selected = {c.listing_id for c in candidates[:k]}
                labeled_hits[k] += len(selected & known_positives[key])
                if complete_entities:
                    entity_hits[k] += sum(
                        split.entity_ids[other] == split.entity_ids[key] for other in selected
                    )
    matrix = index.matrix
    matrix_bytes = (
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        if matrix is not None
        else 0
    )
    return {
        "population": "all unique listings in the evaluated split, including both pair sides",
        "self_matches_excluded": True,
        "vocabulary_fit": "training only",
        "direction": "directed; each gold pair contributes two queries",
        "labeled_positive_directions": labeled_denominator,
        "labeled_pair_recall_at_k": {
            str(k): labeled_hits[k] / labeled_denominator if labeled_denominator else None
            for k in ks
        },
        "entity_positive_directions": entity_denominator if complete_entities else None,
        "entity_recall_at_k": {
            str(k): entity_hits[k] / entity_denominator if entity_denominator else None for k in ks
        }
        if complete_entities
        else None,
        "query_latency_ms": {
            "p50": float(np.percentile(latency_ms, 50)),
            "p95": float(np.percentile(latency_ms, 95)),
        },
        "index_build_seconds": build_seconds,
        "sparse_matrix_bytes": matrix_bytes,
        "channel_hits": dict(channel_counts),
        "queries": len(normalized),
        "memory_note": "Sparse matrix storage only, not peak process or full index memory.",
    }, ranks


def run_evaluation(config: RunConfig, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"output already exists; choose a fresh run directory: {output}")
    splits = [
        load_wdc_pairs(getattr(config, name), name) for name in ("train", "validation", "test")
    ]
    if any(not split.pairs for split in splits):
        raise ValueError("each split must contain labeled pairs")
    audit = audit_splits(splits, test_entity_disjoint=config.test_entity_disjoint)
    train, validation, test = splits
    started = perf_counter()
    matcher = LogisticMatcher(config.seed, config.regularization).fit(train)
    training_seconds = perf_counter() - started
    started = perf_counter()
    validation_probabilities = matcher.predict(validation)
    validation_scoring_seconds = perf_counter() - started
    selection = select_match_threshold(
        np.array([p.label for p in validation.pairs]),
        validation_probabilities,
        config.target_precision,
        config.minimum_validation_matches,
    )
    # Fail on inconsistent reject/match policy before writing or scoring test data.
    decision(0.5, selection.threshold, config.reject_threshold)
    started = perf_counter()
    index = CandidateIndex(config.max_features).fit([normalize(x) for x in train.listings.values()])
    vocabulary_fit_seconds = perf_counter() - started
    output.mkdir(parents=True)
    write_json(output / "config.json", config.model_dump(mode="json"))
    root = Path(__file__).resolve().parents[3]
    package_root = Path(__file__).resolve().parents[1]
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "feature_version": FEATURE_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_metadata(root),
        "dependencies": {
            name: version(name) for name in ("numpy", "polars", "pydantic", "scikit-learn", "scipy")
        },
        "inputs": {
            s.name: {
                "path": str(getattr(config, s.name)),
                "sha256": file_hash(getattr(config, s.name)),
            }
            for s in splits
        },
        "source_sha256": {
            str(p.relative_to(package_root)): file_hash(p)
            for p in sorted(package_root.rglob("*.py"))
        },
        "lock_sha256": file_hash(root / "uv.lock") if (root / "uv.lock").exists() else None,
    }
    write_json(output / "manifest.json", manifest)
    report = {
        "dataset_name": config.dataset_name,
        "dataset_version": config.dataset_version,
        "synthetic": config.synthetic,
        "identity_level": train.identity_level.value,
        "warning": "SYNTHETIC FIXTURE: these results are not benchmark evidence."
        if config.synthetic
        else "Official binary entity labels; no family/variant accuracy claim.",
        "split_audit": audit,
        "threshold_selection": asdict(selection),
        "training_seconds": training_seconds,
        "training_profile": profile(train.listings),
        "vocabulary_fit_seconds": vocabulary_fit_seconds,
        "splits": {},
    }
    for split in (validation, test):
        started = perf_counter()
        probabilities = (
            validation_probabilities if split.name == "validation" else matcher.predict(split)
        )
        scoring_seconds = (
            validation_scoring_seconds if split.name == "validation" else perf_counter() - started
        )
        labels = np.array([p.label for p in split.pairs])
        predictions = (
            probabilities >= selection.threshold
            if selection.threshold is not None
            else np.zeros(len(probabilities), dtype=bool)
        )
        retrieval, ranks = retrieval_evaluation(
            index,
            split,
            sorted(set(config.retrieval_k + [config.pipeline_k])),
            output / f"{split.name}-candidates.jsonl",
        )
        normalized = {key: normalize(x) for key, x in split.listings.items()}
        retrieved = np.array(
            [
                ranks[p.left].get(p.right, config.pipeline_k + 1) <= config.pipeline_k
                or ranks[p.right].get(p.left, config.pipeline_k + 1) <= config.pipeline_k
                for p in split.pairs
            ]
        )
        decisions = Counter()
        errors = Counter()
        with (output / f"{split.name}-pairs.jsonl").open("w", encoding="utf-8") as stream:
            for i, pair in enumerate(split.pairs):
                pair_decision = decision(
                    float(probabilities[i]), selection.threshold, config.reject_threshold
                )
                decisions[pair_decision] += 1
                error = (
                    "false_positive"
                    if predictions[i] and not pair.label
                    else "false_negative"
                    if not predictions[i] and pair.label
                    else None
                )
                evidence = pair_features(normalized[pair.left], normalized[pair.right])
                disagreements = conflicts(normalized[pair.left], normalized[pair.right])
                if error:
                    errors[error] += 1
                    if pair.hard_negative:
                        errors[f"{error}:hard_negative"] += 1
                    for conflict in disagreements:
                        errors[f"{error}:conflict:{conflict}"] += 1
                stream.write(
                    json.dumps(
                        {
                            **asdict(pair),
                            "probability": float(probabilities[i]),
                            "decision": pair_decision,
                            "identity_level": split.identity_level.value,
                            "model_version": __version__,
                            "error": error,
                            "retrieved_at_pipeline_k": bool(retrieved[i]),
                            "left_rank": ranks[pair.left].get(pair.right),
                            "right_rank": ranks[pair.right].get(pair.left),
                            "features": evidence,
                            "conflicts": disagreements,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
        report["splits"][split.name] = {
            "profile": profile(split.listings),
            "retrieval": retrieval,
            "pair_classification": classification_metrics(labels, predictions),
            "probability_quality": probability_metrics(labels, probabilities),
            "decisions": dict(decisions),
            "error_breakdown": dict(errors),
            "scoring_seconds": scoring_seconds,
            "retrieval_plus_matching_on_labeled_pairs": {
                **classification_metrics(labels, predictions & retrieved),
                "k": config.pipeline_k,
                "note": (
                    "Either retrieval direction qualifies. Unlabeled candidates are not "
                    "scored as negatives; this is not full-catalog precision."
                ),
            },
        }
    scaler, logistic = matcher.pipeline.steps[0][1], matcher.pipeline.steps[1][1]
    write_json(
        output / "model.json",
        {
            "kind": "standard_scaler_logistic_regression",
            "features": list(FEATURE_NAMES),
            "feature_version": FEATURE_VERSION,
            "classes": logistic.classes_.tolist(),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coefficients": logistic.coef_.tolist(),
            "intercept": logistic.intercept_.tolist(),
            "iterations": logistic.n_iter_.tolist(),
            "match_threshold": selection.threshold,
            "reject_threshold": config.reject_threshold,
        },
    )
    write_json(output / "report.json", report)
    return report
