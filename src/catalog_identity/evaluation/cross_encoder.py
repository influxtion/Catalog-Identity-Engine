"""Timing gate and fixed train-only screening for the registered joint matcher."""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from catalog_identity.evaluation.development import validation_partitions
from catalog_identity.evaluation.metrics import classification_metrics, probability_metrics
from catalog_identity.evaluation.provenance import implementation_manifest
from catalog_identity.evaluation.runner import file_hash, write_json
from catalog_identity.evaluation.screening import entity_folds, envelope
from catalog_identity.evaluation.semantic_screening import select_semantic_candidate
from catalog_identity.ingestion.audit import development_overlap
from catalog_identity.ingestion.loaders import BenchmarkSplit, load_wdc_pairs
from catalog_identity.matching.baseline import select_match_threshold
from catalog_identity.matching.cross_encoder import ARCHITECTURE, TRAINING, JointMatcher
from catalog_identity.matching.execution import configure_execution, require_same_execution


def subset(split, indices):
    pairs = tuple(split.pairs[i] for i in indices)
    keys = {key for pair in pairs for key in (pair.left, pair.right)}
    return BenchmarkSplit(
        split.name,
        {key: split.listings[key] for key in sorted(keys)},
        pairs,
        {key: split.entity_ids[key] for key in keys},
        split.identity_level,
    )


def projected_cost(train_count, assessment_count, train_rate, inference_rate, overhead):
    # Rate is pairs/second at the actual microbatch/padded length; includes accumulation.
    return 1.5 * (3 * train_count / train_rate + assessment_count / inference_rate + overhead)


def pilot(
    config,
    output: Path,
    validation_report: Path,
    matcher_factory=JointMatcher,
    microbatch=8,
    device="cpu",
):
    """Uses real training text but synthetic compute-only labels, never accuracy."""
    import psutil
    import torch

    if output.exists():
        raise ValueError("output already exists")
    if config.seed != 42 or microbatch not in (4, 8):
        raise ValueError("registered timing settings differ")
    train = load_wdc_pairs(config.train, "train")
    folds = entity_folds(train)
    prior_validation = json.loads(validation_report.read_text(encoding="utf-8"))
    if prior_validation["development"]["inputs"]["train"] != file_hash(config.train):
        raise ValueError("validation aggregate belongs to different training input")
    validation_counts = {
        "calibration": prior_validation["development"]["calibration_pairs"],
        "policy": prior_validation["development"]["policy_pairs"],
        "full_replay": prior_validation["full_validation_replay"]["classification"]["pairs"],
    }
    if any(type(count) is not int or count < 1 for count in validation_counts.values()):
        raise ValueError("prior validation aggregate has invalid pair counts")
    output.mkdir(parents=True)
    started = perf_counter()
    model = matcher_factory(device=device)

    def synchronize():
        if model.device == "cuda:0":
            torch.cuda.synchronize(0)

    gpu_initial_memory = torch.cuda.mem_get_info(0) if model.device == "cuda:0" else None
    load_seconds = perf_counter() - started
    started = perf_counter()
    numeric = model.fit_features(train)
    feature_fit_seconds = perf_counter() - started
    # Stratify by combined text length to span short through long pairs deterministically.
    lengths = np.array(
        [
            sum(
                len(train.listings[key].title or "") + len(train.listings[key].description or "")
                for key in (pair.left, pair.right)
            )
            for pair in train.pairs
        ]
    )
    ordered = np.argsort(lengths, kind="stable")
    sample_size = min(len(ordered), 20 * microbatch)
    indices = ordered[np.linspace(0, len(ordered) - 1, sample_size, dtype=int)]
    sample = subset(train, indices)
    started = perf_counter()
    prepared = model.prepare(sample, numeric[indices])
    prepared["labels"] = (np.arange(len(indices)) % 2).astype(np.float32)
    sample_prepare_seconds = perf_counter() - started
    optimizer = model.optimizer()
    model.network.train()
    accumulation = 32 // microbatch
    rss = [psutil.Process().memory_info().rss]
    timings = []
    optimizer.zero_grad(set_to_none=True)
    for batch_index in range(20):
        batch = np.arange(batch_index * microbatch, (batch_index + 1) * microbatch) % len(indices)
        synchronize()
        started = perf_counter()
        tokens, values = model.batch(prepared, batch, batch_index % 2)
        synthetic_labels = torch.from_numpy(prepared["labels"][batch]).to(model.device)
        logits = model.network(tokens, values)
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(logits, synthetic_labels)
            / accumulation
        )
        if not torch.isfinite(loss):
            raise ValueError("nonfinite timing-pilot loss; execution stopped")
        loss.backward()
        if (batch_index + 1) % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        synchronize()
        elapsed = perf_counter() - started
        if batch_index == 3 and model.device == "cuda:0":
            torch.cuda.reset_peak_memory_stats(0)
        if batch_index >= 4:
            timings.append(elapsed)
        rss.append(psutil.Process().memory_info().rss)
    synchronize()
    started = perf_counter()
    model.logits_prepared(prepared, microbatch)
    synchronize()
    inference_seconds = perf_counter() - started
    rss.append(psutil.Process().memory_info().rss)
    train_rate = 16 * microbatch / sum(timings)
    inference_rate = len(indices) / inference_seconds
    # Full numeric feature fitting was measured. Reserve an additional full preparation
    # pass per fold and a conservative 60 seconds for metadata/artifact serialization.
    per_pair_prepare = sample_prepare_seconds / len(indices)
    per_fit_overhead = load_seconds + feature_fit_seconds + per_pair_prepare * len(train.pairs) + 60
    screening = sum(
        projected_cost(
            len(fit.pairs),
            len(assess.pairs),
            train_rate,
            inference_rate,
            per_fit_overhead
            + per_pair_prepare * len(assess.pairs)
            + feature_fit_seconds * len(assess.pairs) / len(train.pairs),
        )
        for fit, assess, _ in folds
    )
    # Only a previously committed aggregate is read, never validation records.
    validation_pass_pairs = sum(validation_counts.values())
    full = projected_cost(
        len(train.pairs),
        validation_pass_pairs,
        train_rate,
        inference_rate,
        per_fit_overhead
        + per_pair_prepare * validation_pass_pairs
        + feature_fit_seconds * validation_pass_pairs / len(train.pairs),
    )
    report = {
        "phase": "joint_encoder_timing_only",
        "accuracy_inspected": False,
        "synthetic_labels_for_compute_only": True,
        "pilot_model_saved": False,
        "development_only": True,
        "synthetic": config.synthetic,
        "dataset_name": config.dataset_name,
        "dataset_version": config.dataset_version,
        "seed": config.seed,
        "protocol": config.protocol.model_dump() if config.protocol else None,
        "assessment_weights": {"positive": 1, "negative": 3},
        "validation_opened": False,
        "test_opened": False,
        "inputs": {"train": file_hash(config.train)},
        "implementation": implementation_manifest(),
        "architecture": ARCHITECTURE,
        "training": {**TRAINING, "microbatch": microbatch, "accumulation": accumulation},
        "encoder": model.encoder_metadata,
        "execution": model.execution,
        "gpu_memory": None
        if gpu_initial_memory is None
        else {
            "free_after_model_load_bytes": gpu_initial_memory[0],
            "total_bytes": gpu_initial_memory[1],
            "peak_allocated_after_warmup_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_after_warmup_bytes": torch.cuda.max_memory_reserved(0),
        },
        "sample_pairs": len(indices),
        "sample_selection": "even quantiles of combined title/description character length",
        "token_statistics": prepared["statistics"],
        "padded_sequence_length": 256,
        "warmup_microbatches": 4,
        "measured_microbatches": 16,
        "measured_training_seconds": sum(timings),
        "training_pairs_per_second": train_rate,
        "symmetric_inference_seconds": inference_seconds,
        "symmetric_inference_pairs_per_second": inference_rate,
        "model_load_seconds": load_seconds,
        "full_numeric_feature_fit_seconds": feature_fit_seconds,
        "sample_tokenization_seconds": sample_prepare_seconds,
        "per_fit_overhead_reserved_seconds": per_fit_overhead,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "sampled_peak_rss_bytes": max(rss),
        "memory_note": "RSS sampled after microbatches, not an exact peak",
        "projection_safety_factor": 1.5,
        "fold_pair_counts": [
            {"train": len(fit.pairs), "assessment": len(assess.pairs)} for fit, assess, _ in folds
        ],
        "conditional_validation_pair_counts_from_prior_aggregate": validation_counts,
        "prior_validation_aggregate_sha256": file_hash(validation_report),
        "projected_screening_seconds": screening,
        "projected_full_fit_validation_seconds": full,
        "screening_limit_seconds": 5400,
        "full_fit_limit_seconds": 2700,
        "gate_passed": screening <= 5400 and full <= 2700,
        "warning": "Timing projection only; no accuracy metric or claim. Pilot weights discarded. "
        "If gate fails, stop without changing architecture, epochs, length or data.",
    }
    write_json(output / "timing.json", report)
    return report


def verify_control(config, control):
    if (
        config.seed != 42
        or control["seed"] != 42
        or control["assessment_weights"] != {"positive": 1, "negative": 3}
    ):
        raise ValueError("control seed/assessment weighting differs")
    if file_hash(config.train) != control["inputs"]["train"]:
        raise ValueError("control training hash differs")
    current = implementation_manifest()["source_sha256"]
    for path in (
        "matching/distinctions.py",
        "matching/enriched.py",
        "matching/features.py",
        "normalization/__init__.py",
        "evaluation/screening.py",
        "ingestion/loaders.py",
        "matching/baseline.py",
    ):
        if current[path] != control["implementation"]["source_sha256"][path]:
            raise ValueError("control numeric feature implementation changed")
    candidate = control["candidates"][0]
    if candidate["settings"] != {
        "learner": "lightgbm",
        "seed": 42,
        "num_leaves": 15,
        "n_estimators": 200,
        "feature_version": "3",
    }:
        raise ValueError("unexpected control settings")
    return candidate


def screen_joint(config, output: Path, timing_path: Path, control_path: Path, device="cpu"):
    if output.exists():
        raise ValueError("output already exists")
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    if "execution" not in timing:
        raise ValueError("timing gate lacks an execution fingerprint; a new pilot is required")
    execution = configure_execution(device)
    require_same_execution(timing["execution"], execution)
    if not timing["gate_passed"] or timing["inputs"]["train"] != file_hash(config.train):
        raise ValueError("a passing timing gate for the same training input is required")
    if timing["architecture"] != ARCHITECTURE:
        raise ValueError("architecture differs from timing gate")
    if config.seed != 42 or timing["seed"] != 42:
        raise ValueError("registered seed differs from timing gate")
    if timing["training"]["microbatch"] not in (4, 8):
        raise ValueError("unregistered microbatch in timing gate")
    expected_training = {
        **TRAINING,
        "microbatch": timing["training"]["microbatch"],
        "accumulation": 32 // timing["training"]["microbatch"],
    }
    if timing["training"] != expected_training or timing["training"]["microbatch"] not in (4, 8):
        raise ValueError("training constants differ from timing gate")
    if (
        timing["projected_screening_seconds"] > 5400
        or timing["projected_full_fit_validation_seconds"] > 2700
    ):
        raise ValueError("timing projections exceed registered limits")
    current_sources = implementation_manifest()["source_sha256"]
    for path in (
        "matching/cross_encoder.py",
        "matching/execution.py",
        "evaluation/cross_encoder.py",
    ):
        if timing["implementation"]["source_sha256"][path] != current_sources[path]:
            raise ValueError("joint implementation changed after timing gate")
    control_report = json.loads(control_path.read_text(encoding="utf-8"))
    control = verify_control(config, control_report)
    train = load_wdc_pairs(config.train, "train")
    folds = entity_folds(train)
    for (fit, assess, excluded), previous in zip(folds, control["folds"], strict=True):
        if (len(fit.pairs), len(assess.pairs), excluded) != (
            previous["train_pairs"],
            previous["assessment_pairs"],
            previous["boundary_pairs_excluded"],
        ):
            raise ValueError("control fold definition differs")
    output.mkdir(parents=True)
    candidate = {"name": "joint-minilm-v1", "folds": []}
    report = {
        "phase": "joint_encoder_train_only_screening",
        "complete": False,
        "development_only": True,
        "synthetic": config.synthetic,
        "dataset_name": config.dataset_name,
        "dataset_version": config.dataset_version,
        "seed": config.seed,
        "protocol": config.protocol.model_dump() if config.protocol else None,
        "assessment_weights": {"positive": 1, "negative": 3},
        "validation_opened": False,
        "test_opened": False,
        "inputs": {"train": file_hash(config.train)},
        "implementation": implementation_manifest(),
        "architecture": ARCHITECTURE,
        "training": TRAINING,
        "execution": execution,
        "timing_sha256": file_hash(timing_path),
        "control_sha256": file_hash(control_path),
        "control_source_commit": control_report["implementation"]["git"],
        "warning": "Reused development folds; within-fold precision envelopes are optimistic "
        "diagnostics, not independent guarantees.",
        "candidates": [control, candidate],
    }
    last_update = perf_counter()
    started_run = last_update

    def check_deadline():
        if perf_counter() - started_run > 5400:
            raise RuntimeError(
                "registered 90-minute screening resource limit exceeded; incomplete run"
            )

    def progress(epoch, batch, total):
        nonlocal last_update
        now = perf_counter()
        check_deadline()
        if now - last_update >= 30:
            print(f"epoch {epoch + 1}/3, microbatch {batch}/{total}", flush=True)
            last_update = now

    for index, (fit, assess, excluded) in enumerate(folds):
        started = perf_counter()
        model = JointMatcher(device=device)  # Fresh immutable backbone for every fold.
        require_same_execution(execution, model.execution)
        numeric = model.fit_features(fit)
        prepared = model.prepare(fit, numeric)
        check_deadline()
        model.train_prepared(prepared, timing["training"]["microbatch"], progress)
        statistics = prepared["statistics"]
        del prepared
        scores = model.predict(assess, calibrated=False)
        check_deadline()
        labels = np.array([p.label for p in assess.pairs])
        weights = np.where(labels == 1, 1.0, 3.0)
        candidate["folds"].append(
            {
                "fold": index,
                "train_pairs": len(fit.pairs),
                "assessment_pairs": len(assess.pairs),
                "train_positives": sum(p.label for p in fit.pairs),
                "assessment_positives": int(labels.sum()),
                "boundary_pairs_excluded": excluded,
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
                "elapsed_seconds": perf_counter() - started,
                "token_statistics": statistics,
                "training_record": model.training_record,
            }
        )
        del model
        write_json(output / "screening.json", report)
        print(f"joint matcher fold {index + 1}/3 complete", flush=True)
    candidate["mean_weighted_average_precision"] = float(
        np.mean([f["weighted_average_precision"] for f in candidate["folds"]])
    )
    candidate["mean_weighted_recall_99"] = float(
        np.mean([f["envelopes"]["0.99"]["weighted_recall"] for f in candidate["folds"]])
    )
    candidate["supported_folds_99"] = sum(
        f["envelopes"]["0.99"]["selection"]["threshold"] is not None for f in candidate["folds"]
    )
    report["selected_candidate"] = select_semantic_candidate(report["candidates"])
    check_deadline()
    report["complete"] = True
    write_json(output / "screening.json", report)
    return report


def develop_joint(config, output: Path, screening_path: Path, device="cpu"):
    """One conditional full-train fit and one validation replay; never opens test."""
    if output.exists():
        raise ValueError("output already exists")
    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    if not screening.get("complete") or screening.get("selected_candidate") != "joint-minilm-v1":
        raise ValueError("completed qualifying joint-model screening is required")
    if screening["inputs"]["train"] != file_hash(config.train) or config.seed != 42:
        raise ValueError("training input/seed differs from frozen selection")
    if screening["architecture"] != ARCHITECTURE or screening["training"] != TRAINING:
        raise ValueError("joint configuration differs from frozen selection")
    if "execution" not in screening:
        raise ValueError("screening lacks execution fingerprint")
    execution = configure_execution(device)
    require_same_execution(screening["execution"], execution)
    started = perf_counter()
    last_update = started

    def deadline():
        if perf_counter() - started > 2700:
            raise RuntimeError("registered full-fit/validation resource limit exceeded")

    def progress(epoch, batch, total):
        nonlocal last_update
        deadline()
        if perf_counter() - last_update >= 30:
            print(f"full train epoch {epoch + 1}/3, microbatch {batch}/{total}", flush=True)
            last_update = perf_counter()

    train = load_wdc_pairs(config.train, "train")
    validation = load_wdc_pairs(config.validation, "validation")
    overlap = development_overlap(train, validation)
    calibration, policy = validation_partitions(validation, 42)
    model = JointMatcher(device=device)
    require_same_execution(execution, model.execution)
    microbatch = screening["candidates"][1]["folds"][0]["training_record"]["microbatch"]
    numeric = model.fit_features(train)
    prepared = model.prepare(train, numeric)
    deadline()
    model.train_prepared(prepared, microbatch, progress)
    train_statistics = prepared["statistics"]
    del prepared
    calibration_logits = model.logits_prepared(model.prepare(calibration))
    y_cal = np.asarray([pair.label for pair in calibration.pairs])
    calibrator = LogisticRegression(C=1.0).fit(
        calibration_logits[:, None], y_cal, sample_weight=np.where(y_cal == 1, 1.0, 2.0)
    )
    if calibrator.coef_[0, 0] <= 0:
        raise ValueError("calibration reverses score ordering")
    model.calibration = {
        "slope": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
    }
    probabilities = model.predict(policy)
    labels = np.asarray([pair.label for pair in policy.pairs])
    weights = np.where(labels == 1, 1.0, 2.0)
    match = select_match_threshold(
        labels, probabilities, config.target_precision, config.minimum_validation_matches, weights
    )
    rejection = select_match_threshold(
        1 - labels,
        1 - probabilities,
        config.target_precision,
        config.minimum_validation_matches,
        weights,
    )
    reject_threshold = 1 - rejection.threshold if rejection.threshold is not None else None
    if (
        match.threshold is not None
        and reject_threshold is not None
        and reject_threshold >= match.threshold
    ):
        raise ValueError("overlapping operating thresholds")
    deadline()
    # Exactly one diagnostic replay of all official validation pairs.
    replay = model.predict(validation)
    y = np.asarray([pair.label for pair in validation.pairs])
    predicted = replay >= match.threshold if match.threshold is not None else np.zeros(len(y), bool)
    deadline()
    report = {
        "phase": "joint_encoder_selected_validation",
        "development_only": True,
        "test_opened": False,
        "synthetic": config.synthetic,
        "seed": 42,
        "dataset_name": config.dataset_name,
        "dataset_version": config.dataset_version,
        "protocol": config.protocol.model_dump() if config.protocol else None,
        "selection_frozen_sha256": file_hash(screening_path),
        "inputs": {"train": file_hash(config.train), "validation": file_hash(config.validation)},
        "implementation": implementation_manifest(),
        "architecture": ARCHITECTURE,
        "training": model.training_record,
        "encoder": model.encoder_metadata,
        "execution": execution,
        "development_overlap": overlap,
        "training_token_statistics": train_statistics,
        "calibration_pairs": len(calibration.pairs),
        "policy_pairs": len(policy.pairs),
        "cross_partition_pairs_excluded": len(validation.pairs)
        - len(calibration.pairs)
        - len(policy.pairs),
        "match_policy": asdict(match),
        "rejection_policy": asdict(rejection),
        "reject_threshold": reject_threshold,
        "calibration": model.calibration,
        "policy_classification": classification_metrics(
            labels,
            probabilities >= match.threshold
            if match.threshold is not None
            else np.zeros(len(labels), bool),
        ),
        "full_validation_classification": classification_metrics(y, predicted),
        "full_validation_probability_quality": probability_metrics(y, replay, "sigmoid"),
        "matched_positive_entities": len(
            {
                validation.entity_ids[p.left]
                for i, p in enumerate(validation.pairs)
                if predicted[i] and p.label
            }
        ),
        "threshold_selected_on_reported_population": True,
        "precision_status": "empirical; not an established precision guarantee",
        "warning": "Seen-entity validation reuses calibration and threshold-selection data; "
        "not independent performance evidence.",
    }
    model.provenance = {
        key: report[key] for key in ("inputs", "implementation", "selection_frozen_sha256")
    }
    output.mkdir(parents=True)
    model.save(output / "model")
    deadline()
    report["elapsed_seconds"] = perf_counter() - started
    write_json(output / "validation.json", report)
    return report
