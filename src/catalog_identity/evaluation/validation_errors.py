"""Replay saved policies on validation only and export inspectable error evidence."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from catalog_identity.evaluation.development import validation_partitions
from catalog_identity.evaluation.metrics import classification_metrics, probability_metrics
from catalog_identity.evaluation.provenance import implementation_manifest
from catalog_identity.evaluation.runner import file_hash, write_json
from catalog_identity.ingestion.loaders import load_wdc_pairs
from catalog_identity.matching.distinctions import distinction_hints
from catalog_identity.matching.learned import LearnedMatcher
from catalog_identity.normalization import normalize


def validation_errors(config, model_directory: Path, output: Path):
    if output.exists():
        raise ValueError("output already exists")
    policy = json.loads((model_directory / "policy.json").read_text(encoding="utf-8"))
    for name in ("train", "validation"):
        if file_hash(getattr(config, name)) != policy["inputs"][name]:
            raise ValueError("inputs differ from saved model development")
    if policy["synthetic"] != config.synthetic:
        raise ValueError("model and dataset synthetic status differ")
    split = load_wdc_pairs(config.validation, "validation")
    calibration, operating = validation_partitions(split, config.seed)
    partition = {p.pair_id: "calibration" for p in calibration.pairs}
    partition.update({p.pair_id: "policy" for p in operating.pairs})
    model = LearnedMatcher.load(model_directory)
    x = model.features.split(split)
    p = model.score_features(x)
    y = np.array([pair.label for pair in split.pairs])
    threshold = policy["match"]["threshold"]
    predicted = p >= threshold if threshold is not None else np.zeros(len(p), bool)
    rows, errors = [], Counter()
    for i, pair in enumerate(split.pairs):
        left, right = split.listings[pair.left], split.listings[pair.right]
        a, b = distinction_hints(normalize(left).title), distinction_hints(normalize(right).title)
        error = (
            "false_positive"
            if predicted[i] and not pair.label
            else ("false_negative" if pair.label and not predicted[i] else None)
        )
        observations = [
            f"{key}:conflict" for key in a if a[key] and b[key] and a[key].isdisjoint(b[key])
        ]
        observations += [f"{key}:one_sided" for key in a if bool(a[key]) != bool(b[key])]
        if error:
            errors[error] += 1
            if pair.hard_negative:
                errors[f"{error}:hard_negative"] += 1
            for observed in observations:
                errors[f"{error}:{observed}"] += 1
        rows.append(
            {
                "pair_id": pair.pair_id,
                "partition": partition.get(pair.pair_id, "cross_partition"),
                "label": pair.label,
                "probability": float(p[i]),
                "error": error,
                "hard_negative": pair.hard_negative,
                "left_title": left.title,
                "right_title": right.title,
                "observations": observations,
                "features": dict(zip(model.features.names, x[i].tolist(), strict=True)),
            }
        )
    report = {
        "phase": "validation_diagnostics",
        "development_only": True,
        "development_overlap": policy.get("development_overlap"),
        "threshold_selected_on_reported_population": True,
        "precision_status": "empirical; not an established precision guarantee",
        "test_opened": False,
        "synthetic": config.synthetic,
        "warning": (
            "Full validation includes calibration and policy-selection data; "
            "not an independent performance estimate."
        ),
        "inputs": policy["inputs"],
        "model_settings": model.settings,
        "model_files": {
            p.name: file_hash(p) for p in sorted(model_directory.iterdir()) if p.is_file()
        },
        "implementation": implementation_manifest(),
        "match_threshold": threshold,
        "classification": classification_metrics(y, predicted),
        "probability_quality": probability_metrics(
            y, p, "sigmoid" if model.calibration else "none"
        ),
        "error_breakdown": dict(errors),
        "matched_positive_entities": len(
            {
                split.entity_ids[pair.left]
                for i, pair in enumerate(split.pairs)
                if predicted[i] and pair.label
            }
        ),
    }
    output.mkdir(parents=True)
    with (output / "validation-pairs.jsonl").open("w", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda row: -row["probability"]):
            stream.write(json.dumps(row) + "\n")
    write_json(output / "report.json", report)
    return report
