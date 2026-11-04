"""Operator-only analysis. Review keys never enter the annotation service."""

import hashlib
import json
from collections import Counter, defaultdict

from catalog_identity.annotations.schema import disagreement
from catalog_identity.evaluation.training_audit import REVIEW_STRATA


def summarize(rows):
    sufficiency, comparisons, uncertain, roles = Counter(), Counter(), Counter(), Counter()
    disagreements = sources = 0
    for case, key in rows:
        resolved = case["resolved_decisions"]
        sufficiency[resolved["input_sufficiency"]] += 1
        independent = [r for r in case["submissions"] if r["role"] == "reviewer"]
        disagreements += int(disagreement(*[r["judgment"] for r in independent]))
        for field in ("benchmark_identity", "family_relation", "variant_relation"):
            uncertain[field] += int(resolved[field] == "uncertain")
        identity = resolved["benchmark_identity"]
        comparison = (
            "uncertain"
            if identity == "uncertain"
            else "agrees"
            if (identity == "same") == bool(key["benchmark_label"])
            else "differs"
        )
        comparisons[comparison] += 1
        sources += int(any(r["judgment"]["sources"] for r in case["submissions"]))
        for review in case["submissions"]:
            roles.update(e["role"] for e in review["judgment"]["evidence"])
    count = len(rows)
    insufficient = count - sufficiency["sufficient"]
    return {
        "cases": count,
        "independent_disagreements": disagreements,
        "disagreement_rate": disagreements / count if count else None,
        "resolved_sufficiency_counts": dict(sufficiency),
        "cases_needing_more_evidence": insufficient,
        "needs_more_evidence_rate": insufficient / count if count else None,
        "uncertain_identity_counts": dict(uncertain),
        "benchmark_label_comparison": dict(comparisons),
        "cases_with_recorded_source_checks": sources,
        "evidence_items_by_role_all_submissions": dict(roles),
    }


def analyze(export, key_path):
    cases = export["cases"]
    if not cases or any(c["status"] not in ("agreed", "adjudicated") for c in cases):
        raise ValueError("resolve every case before joining the private review key")
    raw = key_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = export["batch"]["provenance"].get("review_key_sha256")
    if expected and expected != digest:
        raise ValueError("review key checksum differs from the imported audit")
    keys = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ids = [key.get("review_id") for key in keys]
    if len(set(ids)) != len(ids) or set(ids) != {c["case"]["review_id"] for c in cases}:
        raise ValueError("review key must contain exactly the batch's unique case IDs")
    for key in keys:
        if (
            type(key.get("benchmark_label")) is not int
            or key["benchmark_label"] not in (0, 1)
            or key.get("primary_stratum") not in REVIEW_STRATA
            or not isinstance(key.get("tags"), list)
            or any(tag not in REVIEW_STRATA for tag in key["tags"])
        ):
            raise ValueError("invalid review-key label or strata")
    by_id = {key["review_id"]: key for key in keys}
    joined = [(c, by_id[c["case"]["review_id"]]) for c in cases]
    primary, overlapping = defaultdict(list), defaultdict(list)
    for case, key in joined:
        primary[key["primary_stratum"]].append((case, key))
        for tag in set(key["tags"]):
            overlapping[tag].append((case, key))
    return {
        "schema_version": 1,
        "batch_id": export["batch_id"],
        "packet_sha256": export["batch"]["packet_sha256"],
        "review_key_sha256": digest,
        "key_binding": "checksum_verified" if expected else "case_ids_only_legacy_report",
        "guideline_version": export["batch"]["guideline_version"],
        "overall": summarize(joined),
        "primary_strata": {k: summarize(v) for k, v in sorted(primary.items())},
        "overlapping_tags": {k: summarize(v) for k, v in sorted(overlapping.items())},
        "official_labels_modified": False,
        "warning": "Targeted diagnostic sample; rates are not population prevalence or matcher "
        "accuracy. A difference from a benchmark label is not proof the benchmark is wrong. "
        "Primary strata partition cases; overlapping tags do not. Evidence-item counts include "
        "both reviewers and any adjudicator. Source checks are reviewer reports, not automated "
        "verification. No annotations are automatically eligible for model training.",
    }
