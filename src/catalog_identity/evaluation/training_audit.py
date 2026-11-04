"""Training-only evidence diagnostics. No label corrections or model selection."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from catalog_identity.evaluation.provenance import implementation_manifest
from catalog_identity.evaluation.runner import file_hash, write_json
from catalog_identity.ingestion.loaders import load_wdc_pairs
from catalog_identity.matching.distinctions import distinction_hints
from catalog_identity.matching.features import tokens
from catalog_identity.normalization import normalize

TEXT_FIELDS = ("title", "brand", "description")
REVIEW_STRATA = (
    "conflicting_text_inputs",
    "conflicting_token_inputs",
    "control_false_positive",
    "control_false_negative",
    "high_overlap_negative",
    "low_overlap_positive",
    "code_disagreement_positive",
    "missing_description_positive",
    "truncated_positive",
    "reference",
)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


def pair_signature(left, right):
    return fingerprint(sorted((left, right)))


def collision_summary(split, signatures):
    """Opposite labels on identical observable pairs, not a finding of label noise."""
    groups = defaultdict(list)
    for pair in split.pairs:
        groups[signatures[pair.pair_id]].append(pair)
    mixed = [group for group in groups.values() if len({p.label for p in group}) == 2]
    affected = {p.pair_id for group in mixed for p in group}
    return {
        "unique_pair_inputs": len(groups),
        "repeated_input_groups": sum(len(g) > 1 for g in groups.values()),
        "pairs_beyond_first_input_occurrence": len(split.pairs) - len(groups),
        "mixed_label_groups": len(mixed),
        "affected_pairs": len(affected),
        "affected_positives": sum(p.label for g in mixed for p in g),
        "affected_negatives": sum(1 - p.label for g in mixed for p in g),
        "minimum_observed_errors": sum(min(Counter(p.label for p in g).values()) for g in mixed),
    }, affected


def listing_collisions(split, signatures):
    groups = defaultdict(list)
    for key, signature in signatures.items():
        groups[signature].append(key)
    mixed = [g for g in groups.values() if len({split.entity_ids[k] for k in g}) > 1]
    return {
        "unique_listing_inputs": len(groups),
        "cross_entity_groups": len(mixed),
        "listings_in_cross_entity_groups": sum(map(len, mixed)),
        "entities_in_cross_entity_groups": len({split.entity_ids[k] for g in mixed for k in g}),
    }


def present(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(present(v) for v in value.values())
    if isinstance(value, list):
        return any(present(v) for v in value)
    return value is not None


def summarize_inputs(split, pair_tokenizer=None):
    if split.name != "train" or not split.pairs:
        raise ValueError("input audit requires nonempty training data")
    if set(split.entity_ids) != set(split.listings) or any(
        not isinstance(e, str) or not e for e in split.entity_ids.values()
    ):
        raise ValueError("complete training entity metadata required")
    records = {key: normalize(value) for key, value in split.listings.items()}
    listing_signatures = {
        "raw_text": {
            k: fingerprint([getattr(v, f) for f in TEXT_FIELDS]) for k, v in split.listings.items()
        },
        "normalized_title": {k: fingerprint(v.title) for k, v in records.items()},
        "normalized_text": {
            k: fingerprint([getattr(v, f) for f in TEXT_FIELDS]) for k, v in records.items()
        },
    }
    pair_signatures = {
        name: {p.pair_id: pair_signature(values[p.left], values[p.right]) for p in split.pairs}
        for name, values in listing_signatures.items()
    }
    details = {p.pair_id: {"tags": [], "truncated_fields": []} for p in split.pairs}
    truncations = {str(label): Counter() for label in (0, 1)}
    if pair_tokenizer is not None:
        parts = {k: pair_tokenizer.parts(v) for k, v in split.listings.items()}
        unbounded = {k: fingerprint(v) for k, v in parts.items()}
        pair_signatures["untruncated_tokens"] = {
            p.pair_id: pair_signature(unbounded[p.left], unbounded[p.right]) for p in split.pairs
        }
        pair_signatures["pair_tokens_256"] = {}
        for pair in split.pairs:
            encoded, truncated = [], set()
            for a, b in ((pair.left, pair.right), (pair.right, pair.left)):
                row, stats = pair_tokenizer.encode_parts(parts[a], parts[b])
                encoded.append(fingerprint({k: v.tolist() for k, v in row.items()}))
                for side in ("left_truncated", "right_truncated"):
                    truncated.update(k for k, v in stats[side].items() if v)
            pair_signatures["pair_tokens_256"][pair.pair_id] = pair_signature(*encoded)
            details[pair.pair_id]["truncated_fields"] = sorted(truncated)
            if truncated:
                truncations[str(pair.label)].update(["any", *sorted(truncated)])
    collisions, affected = {}, {}
    for name, signatures in pair_signatures.items():
        collisions[name], affected[name] = collision_summary(split, signatures)

    word_sets = {k: tokens(v.title) for k, v in records.items()}
    codes = {k: distinction_hints(v.title)["code"] for k, v in records.items()}
    for pair in split.pairs:
        row = details[pair.pair_id]
        a, b = word_sets[pair.left], word_sets[pair.right]
        overlap = len(a & b) / len(a | b) if a | b else 0.0
        row["title_token_jaccard"] = overlap
        row["signature"] = pair_signatures["normalized_text"][pair.pair_id]
        tags = row["tags"]
        if pair.pair_id in affected["normalized_text"]:
            tags.append("conflicting_text_inputs")
        if pair.pair_id in affected.get("pair_tokens_256", set()):
            tags.append("conflicting_token_inputs")
        if not pair.label and overlap >= 0.8:
            tags.append("high_overlap_negative")
        if pair.label and overlap <= 0.2:
            tags.append("low_overlap_positive")
        ac, bc = codes[pair.left], codes[pair.right]
        if pair.label and ac and bc and ac.isdisjoint(bc):
            tags.append("code_disagreement_positive")
        if pair.label and any(not records[k].description for k in (pair.left, pair.right)):
            tags.append("missing_description_positive")
        if pair.label and row["truncated_fields"]:
            tags.append("truncated_positive")

    fields = (*TEXT_FIELDS, "category", "model", "gtin", "attributes", "price", "currency")
    raw_fields = sorted({f for v in split.listings.values() for f in v.raw})
    entities = Counter(split.entity_ids.values())
    positive_entities = Counter(split.entity_ids[p.left] for p in split.pairs if p.label)
    report = {
        "pairs": len(split.pairs),
        "positives": sum(p.label for p in split.pairs),
        "negatives": sum(1 - p.label for p in split.pairs),
        "listings": len(split.listings),
        "entities": len(entities),
        "nonempty_listing_fields": {
            f: sum(present(getattr(v, f)) for v in split.listings.values()) for f in fields
        },
        "nonempty_source_fields": {
            f: sum(present(v.raw.get(f)) for v in split.listings.values()) for f in raw_fields
        },
        "entity_support": {
            "listing_count_quantiles": np.quantile(
                list(entities.values()), [0, 0.5, 0.9, 1]
            ).tolist(),
            "positive_pair_count_quantiles": np.quantile(
                [positive_entities[e] for e in entities], [0, 0.5, 0.9, 1]
            ).tolist(),
            "entities_without_positive_pairs": sum(positive_entities[e] == 0 for e in entities),
        },
        "listing_collisions": {
            name: listing_collisions(split, values) for name, values in listing_signatures.items()
        },
        "pair_collisions": collisions,
        "text_collision_pairs_resolved_by_brand_description": len(
            affected["normalized_title"] - affected["normalized_text"]
        ),
        "token_audit": None
        if pair_tokenizer is None
        else {
            "pairs_with_truncation_by_label": {k: dict(v) for k, v in truncations.items()},
            "new_mixed_pairs_after_tokenization": len(
                affected["untruncated_tokens"] - affected["normalized_text"]
            ),
            "new_mixed_pairs_after_budget": len(
                affected["pair_tokens_256"] - affected["untruncated_tokens"]
            ),
        },
        "declared_hard_negative_pairs": sum(p.hard_negative for p in split.pairs if not p.label),
        "strata": stratum_counts(split, details),
        "interpretation": "Hints and collisions are review signals, not corrected labels. "
        "Collision bounds apply only to these observed inputs and their stated representation. "
        "Raw-field presence does not establish trustworthy identifiers or annotation quality.",
    }
    return report, details


def stratum_counts(split, details):
    result = {}
    for tag in REVIEW_STRATA:
        pairs = [p for p in split.pairs if tag == "reference" or tag in details[p.pair_id]["tags"]]
        assessed = [p for p in pairs if "control" in details[p.pair_id]]
        counts = Counter(details[p.pair_id]["control"]["outcome"] for p in assessed)
        result[tag] = {
            "pairs": len(pairs),
            "positives": sum(p.label for p in pairs),
            "entities": len({split.entity_ids[k] for p in pairs for k in (p.left, p.right)}),
            "control_assessed_pairs": len(assessed),
            "control_counts": {k: counts[k] for k in ("tp", "fp", "fn", "tn")},
        }
    return result


def local_tokenizer(directory):
    from transformers import AutoTokenizer

    from catalog_identity.matching.cross_encoder import PairTokenizer

    directory = directory.resolve(strict=True)
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
        "config.json",
        "added_tokens.json",
    )
    files = {name: file_hash(directory / name) for name in names if (directory / name).is_file()}
    tokenizer = AutoTokenizer.from_pretrained(
        str(directory), local_files_only=True, token=False, trust_remote_code=False
    )
    return PairTokenizer(tokenizer), {"snapshot": directory.name, "files_sha256": files}


def run_audit(config, output: Path, tokenizer_directory=None, control_report=None):
    if output.exists():
        raise ValueError("audit output already exists")
    split = load_wdc_pairs(config.train, "train")
    tokenizer, tokenizer_metadata = (None, None)
    if tokenizer_directory is not None:
        tokenizer, tokenizer_metadata = local_tokenizer(tokenizer_directory)
    output.mkdir(parents=True)
    (output / ".gitignore").write_text("review-blind.jsonl\nreview-key.jsonl\n", encoding="utf-8")
    report, details = summarize_inputs(split, tokenizer)
    report.update(
        {
            "schema_version": "training-input-audit-v1",
            "complete": True,
            "dataset_name": config.dataset_name,
            "dataset_version": config.dataset_version,
            "synthetic": config.synthetic,
            "identity_level": split.identity_level,
            "inputs": {"train": file_hash(config.train)},
            "seed": config.seed,
            "validation_opened": False,
            "test_opened": False,
            "implementation": implementation_manifest(),
            "tokenizer": tokenizer_metadata,
        }
    )
    if control_report is not None:
        from catalog_identity.evaluation.training_review import reproduce_control

        report["control_reproduction"] = reproduce_control(config, split, details, control_report)
        report["strata"] = stratum_counts(split, details)
    from catalog_identity.evaluation.training_review import build_review

    blind, key, sampling = build_review(split, details, seed=config.seed)
    report["review_sampling"] = sampling
    for name, rows in (("review-blind.jsonl", blind), ("review-key.jsonl", key)):
        (output / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    report["review_artifacts_sha256"] = {
        name: file_hash(output / name) for name in ("review-blind.jsonl", "review-key.jsonl")
    }
    write_json(output / "audit.json", report)
    return report
