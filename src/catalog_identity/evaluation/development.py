"""Validation-only model development. This module never opens the test file."""

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from catalog_identity.evaluation.config import RunConfig
from catalog_identity.evaluation.metrics import classification_metrics, probability_metrics
from catalog_identity.evaluation.provenance import implementation_manifest
from catalog_identity.evaluation.runner import file_hash, write_json
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


def develop(
    config: RunConfig, output: Path, learners=("logistic", "lightgbm"), negative_policy="all"
) -> dict:
    if output.exists():
        raise ValueError("output already exists")
    train = load_wdc_pairs(config.train, "train")
    if negative_policy not in ("all", "random_only"):
        raise ValueError("unknown training negative policy")
    original_training_pairs = len(train.pairs)
    if negative_policy == "random_only":
        train = BenchmarkSplit(
            train.name,
            train.listings,
            tuple(p for p in train.pairs if p.label or not p.hard_negative),
            train.entity_ids,
            train.identity_level,
        )
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
        "implementation": implementation_manifest(),
        "partition_weighting": (
            "Positive weight 1, negative weight 2 corrects unequal inclusion probabilities. "
            "Classification counts and probability diagnostics describe the retained subset."
        ),
        "models": {},
        "training_selection": {
            "negative_policy": negative_policy,
            "original_pairs": original_training_pairs,
            "used_pairs": len(train.pairs),
            "positives": sum(p.label for p in train.pairs),
        },
    }
    y = np.array([p.label for p in policy.pairs])
    policy_weights = np.where(y == 1, 1.0, 2.0)
    calibration_weights = np.array([1.0 if p.label else 2.0 for p in calibration.pairs])
    for learner in learners:
        model = LearnedMatcher(learner=learner, seed=config.seed).fit(train)
        raw = model.predict(policy, calibrated=False)
        model.calibrate(calibration, calibration_weights)
        p = model.predict(policy)
        selection = select_match_threshold(
            y, p, config.target_precision, config.minimum_validation_matches, policy_weights
        )
        rejection = select_match_threshold(
            1 - y, 1 - p, config.target_precision, config.minimum_validation_matches, policy_weights
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
            "training_selection": report["training_selection"],
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
