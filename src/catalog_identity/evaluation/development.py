"""Validation-only model development. This module never opens the test file."""

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from catalog_identity.evaluation.config import RunConfig
from catalog_identity.evaluation.metrics import classification_metrics, probability_metrics
from catalog_identity.evaluation.runner import file_hash, git_metadata, write_json
from catalog_identity.ingestion.loaders import BenchmarkSplit, load_wdc_pairs
from catalog_identity.matching.baseline import select_match_threshold
from catalog_identity.matching.learned import LearnedMatcher


def validation_partitions(split: BenchmarkSplit, seed: int = 42):
    if split.name != "validation" or len(split.entity_ids) != len(split.listings):
        raise ValueError("complete validation entity metadata is required")
    buckets = {
        key: int(hashlib.sha256(f"{seed}:{entity}".encode()).hexdigest(), 16) % 2
        for key, entity in split.entity_ids.items()
    }
    partitions = []
    for bucket in (0, 1):
        listings = {k: v for k, v in split.listings.items() if buckets[k] == bucket}
        pairs = tuple(p for p in split.pairs if p.left in listings and p.right in listings)
        partitions.append(
            BenchmarkSplit(
                "validation",
                listings,
                pairs,
                {k: split.entity_ids[k] for k in listings},
                split.identity_level,
            )
        )
    return partitions


def develop(config: RunConfig, output: Path, learners=("logistic", "lightgbm")) -> dict:
    if output.exists():
        raise ValueError("output already exists")
    train = load_wdc_pairs(config.train, "train")
    validation = load_wdc_pairs(config.validation, "validation")
    if train.listings.keys() & validation.listings.keys():
        raise ValueError("train/validation listing leakage")
    calibration, policy = validation_partitions(validation, config.seed)
    output.mkdir(parents=True)
    report = {
        "phase": "validation_only",
        "test_opened": False,
        "synthetic": config.synthetic,
        "partition_seed": config.seed,
        "calibration_pairs": len(calibration.pairs),
        "policy_pairs": len(policy.pairs),
        "cross_partition_pairs_excluded": len(validation.pairs)
        - len(calibration.pairs)
        - len(policy.pairs),
        "inputs": {"train": file_hash(config.train), "validation": file_hash(config.validation)},
        "git": git_metadata(Path(__file__).resolve().parents[3]),
        "models": {},
    }
    y = np.array([p.label for p in policy.pairs])
    for learner in learners:
        model = LearnedMatcher(learner=learner, seed=config.seed).fit(train)
        raw = model.predict(policy, calibrated=False)
        model.calibrate(calibration)
        p = model.predict(policy)
        selection = select_match_threshold(
            y, p, config.target_precision, config.minimum_validation_matches
        )
        rejection = select_match_threshold(
            1 - y, 1 - p, config.target_precision, config.minimum_validation_matches
        )
        reject_threshold = 1 - rejection.threshold if rejection.threshold is not None else None
        if (
            selection.threshold is not None
            and reject_threshold is not None
            and reject_threshold >= selection.threshold
        ):
            raise ValueError("overlapping operating thresholds")
        prediction = (
            p >= selection.threshold if selection.threshold is not None else np.zeros(len(p), bool)
        )
        model_path = output / learner
        model.save(model_path)
        policy_record = {
            "match": asdict(selection),
            "reject": asdict(rejection),
            "reject_threshold": reject_threshold,
            "inputs": report["inputs"],
            "synthetic": config.synthetic,
            "identity_level": "benchmark_entity",
        }
        write_json(model_path / "policy.json", policy_record)
        metrics = {
            "match_policy": asdict(selection),
            "reject_threshold": reject_threshold,
            "classification": classification_metrics(y, prediction),
            "raw_probability_quality": probability_metrics(y, raw),
            "calibrated_probability_quality": probability_metrics(y, p),
        }
        report["models"][learner] = metrics
    write_json(output / "validation.json", report)
    return report
