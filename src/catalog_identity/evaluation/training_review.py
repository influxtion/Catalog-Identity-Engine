"""Reproduce the fixed control and sample local, blinded input-review tasks."""

import json
from collections import Counter

import numpy as np
from sklearn.metrics import average_precision_score

from catalog_identity.evaluation.cross_encoder import verify_control
from catalog_identity.evaluation.metrics import classification_metrics
from catalog_identity.evaluation.runner import file_hash
from catalog_identity.evaluation.screening import entity_folds
from catalog_identity.evaluation.training_audit import REVIEW_STRATA, fingerprint
from catalog_identity.matching.learned import LearnedMatcher


def verify_scores(labels, scores, saved):
    scores = np.asarray(scores)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("invalid reproduced control scores")
    threshold = saved["envelopes"]["0.99"]["selection"]["threshold"]
    predicted = scores >= threshold if threshold is not None else np.zeros(len(labels), bool)
    counts = classification_metrics(labels, predicted)
    ap = float(average_precision_score(labels, scores, sample_weight=np.where(labels == 1, 1, 3)))
    if counts != saved["envelopes"]["0.99"]["raw"] or not np.isclose(
        ap, saved["weighted_average_precision"], rtol=0, atol=1e-12
    ):
        raise ValueError("control reproduction differs; do not publish error attribution")
    return predicted, counts, ap


def reproduce_control(config, split, details, path):
    previous = json.loads(path.read_text(encoding="utf-8"))
    candidate = verify_control(config, previous)
    if len(candidate["folds"]) != 3:
        raise ValueError("expected three saved control folds")
    reports = []
    for index, (fit, assess, excluded) in enumerate(entity_folds(split, config.seed)):
        saved = candidate["folds"][index]
        actual = {
            "fold": index,
            "train_pairs": len(fit.pairs),
            "assessment_pairs": len(assess.pairs),
            "boundary_pairs_excluded": excluded,
            "train_entities": len(set(fit.entity_ids.values())),
            "assessment_entities": len(set(assess.entity_ids.values())),
        }
        if any(saved[k] != v for k, v in actual.items()):
            raise ValueError("saved control fold differs")
        model = LearnedMatcher(**candidate["settings"]).fit(fit)
        if model.booster.params != candidate["effective_parameters"]:
            raise ValueError("control effective parameters differ")
        scores = model.predict(assess, calibrated=False)
        labels = np.asarray([p.label for p in assess.pairs])
        predicted, counts, ap = verify_scores(labels, scores, saved)
        for pair, score, decision in zip(assess.pairs, scores, predicted, strict=True):
            outcome = ("tp" if decision else "fn") if pair.label else ("fp" if decision else "tn")
            row = details[pair.pair_id]
            if "control" in row:
                raise ValueError("a pair was assessed in multiple folds")
            row["control"] = {"fold": index, "score": float(score), "outcome": outcome}
            if outcome in ("fp", "fn"):
                row["tags"].append(
                    "control_false_positive" if outcome == "fp" else "control_false_negative"
                )
        reports.append(
            {
                **actual,
                "classification": counts,
                "weighted_AP": ap,
                "saved_threshold": saved["envelopes"]["0.99"]["selection"]["threshold"],
            }
        )
        print(f"Control fold {index + 1}/3 reproduced", flush=True)
    assessed = [p for p in split.pairs if "control" in details[p.pair_id]]
    return {
        "source_report_sha256": file_hash(path),
        "fits": 3,
        "folds": reports,
        "settings": candidate["settings"],
        "thresholds_reselected": False,
        "assessed_pairs": len(assessed),
        "assessed_positives": sum(p.label for p in assessed),
        "assessed_negatives": sum(1 - p.label for p in assessed),
        "unassessed_boundary_negatives": len(split.pairs) - len(assessed),
        "warning": "Reused development folds and saved within-fold selected thresholds. "
        "Not independent accuracy evidence. No pooled scores or new operating policy.",
    }


def build_review(split, details, seed=42, quota=20, entity_cap=2):
    if split.name != "train" or quota < 1 or entity_cap < 1:
        raise ValueError("training split and positive review limits required")
    selected, usage, sampling = {}, Counter(), {}
    for tag in REVIEW_STRATA:
        eligible = [
            p for p in split.pairs if tag == "reference" or tag in details[p.pair_id]["tags"]
        ]
        ordered = sorted(eligible, key=lambda p: fingerprint([seed, tag, p.pair_id]))
        taken = 0
        for pair in ordered:
            entities = {split.entity_ids[k] for k in (pair.left, pair.right)}
            if pair.pair_id in selected or any(usage[e] >= entity_cap for e in entities):
                continue
            selected[pair.pair_id] = (pair, tag)
            usage.update(entities)
            taken += 1
            if taken == quota:
                break
        sampling[tag] = {
            "eligible_pairs": len(eligible),
            "selected_as_primary": taken,
            "quota_shortfall": quota - taken,
        }

    blind, key = [], []
    fields = (
        "title",
        "brand",
        "description",
        "category",
        "model",
        "gtin",
        "attributes",
        "price",
        "currency",
        "product_url",
    )
    for pair, primary in selected.values():
        review_id = fingerprint(["training-review-v1", seed, pair.pair_id])
        blind.append(
            {
                "review_id": review_id,
                "left": {f: getattr(split.listings[pair.left], f) for f in fields},
                "right": {f: getattr(split.listings[pair.right], f) for f in fields},
                "judgment": {
                    "input_sufficiency": None,
                    "benchmark_identity": None,
                    "family_relation": None,
                    "variant_relation": None,
                    "evidence_fields": [],
                    "reason": None,
                },
            }
        )
        key.append(
            {
                "review_id": review_id,
                "pair_id": pair.pair_id,
                "left": pair.left,
                "right": pair.right,
                "benchmark_label": pair.label,
                "entity_ids": sorted({split.entity_ids[pair.left], split.entity_ids[pair.right]}),
                "primary_stratum": primary,
                **details[pair.pair_id],
            }
        )
    blind.sort(key=lambda r: r["review_id"])
    key.sort(key=lambda r: r["review_id"])
    return (
        blind,
        key,
        {
            "seed": seed,
            "quota_per_stratum": quota,
            "entity_cap": entity_cap,
            "priority_order": list(REVIEW_STRATA),
            "strata": sampling,
            "selected_pairs": len(blind),
            "distinct_entities": len(usage),
            "maximum_selected_pairs_per_entity": max(usage.values(), default=0),
            "warning": "Targeted overlapping strata; greedy priority and global entity cap. "
            "Reference drawn from remaining eligible pairs. "
            "Not a prevalence or performance sample. "
            "Keep the key away from independent reviewers; labels remain unchanged.",
        },
    )
