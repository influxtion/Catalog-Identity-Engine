"""Evaluate a saved validation-selected model without refitting it."""

import json
from pathlib import Path

import numpy as np

from catalog_identity.evaluation.metrics import classification_metrics, probability_metrics
from catalog_identity.evaluation.provenance import implementation_manifest
from catalog_identity.evaluation.runner import file_hash, retrieval_evaluation, write_json
from catalog_identity.ingestion.audit import audit_splits
from catalog_identity.ingestion.loaders import load_wdc_pairs
from catalog_identity.matching.learned import LearnedMatcher
from catalog_identity.normalization import normalize
from catalog_identity.retrieval import CandidateIndex
from catalog_identity.retrieval.semantic import HybridIndex, SentenceEncoder


def evaluate_frozen(config, model_directory: Path, output: Path, hybrid=False):
    if output.exists():
        raise ValueError("output already exists")
    policy = json.loads((model_directory / "policy.json").read_text(encoding="utf-8"))
    if policy["synthetic"] != config.synthetic:
        raise ValueError("model and dataset synthetic status differ")
    for name in ("train", "validation"):
        if file_hash(getattr(config, name)) != policy["inputs"][name]:
            raise ValueError(f"{name} differs from frozen model development inputs")
    splits = [
        load_wdc_pairs(getattr(config, name), name) for name in ("train", "validation", "test")
    ]
    audit = audit_splits(splits, config.test_entity_disjoint)
    model = LearnedMatcher.load(model_directory)
    test = splits[2]
    probabilities = model.predict(test)
    labels = np.array([p.label for p in test.pairs])
    threshold = policy["match"]["threshold"]
    predictions = (
        probabilities >= threshold if threshold is not None else np.zeros(len(labels), bool)
    )
    index = HybridIndex(SentenceEncoder()) if hybrid else CandidateIndex(config.max_features)
    index.fit([normalize(x) for x in splits[0].listings.values()])
    output.mkdir(parents=True)
    retrieval, ranks = retrieval_evaluation(
        index, test, config.retrieval_k, output / "candidates.jsonl"
    )
    if hybrid:
        retrieval["embedding_note"] = (
            "Fixed MiniLM revision; indexed embeddings cached for queries."
        )
        retrieval["faiss_index_bytes"] = index.semantic.index_bytes
    rows = []
    for pair, probability, predicted in zip(test.pairs, probabilities, predictions, strict=True):
        reject = policy["reject_threshold"]
        decision = (
            "match"
            if predicted
            else "reject"
            if reject is not None and probability <= reject
            else "review"
        )
        rows.append(
            {
                "pair_id": pair.pair_id,
                "left": pair.left,
                "right": pair.right,
                "label": pair.label,
                "probability": float(probability),
                "decision": decision,
                "left_rank": ranks[pair.left].get(pair.right),
                "right_rank": ranks[pair.right].get(pair.left),
            }
        )
    (output / "pairs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    report = {
        "phase": "frozen_test_evaluation",
        "synthetic": config.synthetic,
        "model_settings": model.settings,
        "policy": policy,
        "split_audit": audit,
        "model_files": {
            p.name: file_hash(p) for p in sorted(model_directory.iterdir()) if p.is_file()
        },
        "test_sha256": file_hash(config.test),
        "implementation": implementation_manifest(),
        "classification": classification_metrics(labels, predictions),
        "probability_quality": probability_metrics(labels, probabilities),
        "retrieval": retrieval,
    }
    write_json(output / "report.json", report)
    return report
