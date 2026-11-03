from pathlib import Path

from catalog_identity.evaluation.provenance import implementation_manifest
from catalog_identity.evaluation.runner import file_hash, retrieval_evaluation, write_json
from catalog_identity.ingestion.loaders import load_wdc_pairs
from catalog_identity.normalization import normalize
from catalog_identity.retrieval import CandidateIndex
from catalog_identity.retrieval.semantic import (
    MINILM_REVISION,
    HybridIndex,
    SemanticIndex,
    SentenceEncoder,
)


def retrieval_ablation(config, output: Path, allow_download=False):
    if output.exists():
        raise ValueError("output already exists")
    train = load_wdc_pairs(config.train, "train")
    validation = load_wdc_pairs(config.validation, "validation")
    if train.listings.keys() & validation.listings.keys():
        raise ValueError("listing leakage")
    encoder = SentenceEncoder(allow_download=allow_download)
    output.mkdir(parents=True)
    report = {
        "implementation": implementation_manifest(),
        "phase": "validation_only",
        "synthetic": config.synthetic,
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": MINILM_REVISION,
        "license": "Apache-2.0 (model card); benchmark data licensing remains separate",
        "query_timing_note": "Indexed listing embeddings are cached; build includes encoding.",
        "inputs": {"train": file_hash(config.train), "validation": file_hash(config.validation)},
        "channels": {},
    }
    for name, index in (
        ("lexical", CandidateIndex(config.max_features)),
        ("semantic", SemanticIndex(encoder)),
        ("hybrid", HybridIndex(encoder, config.max_features)),
    ):
        index.fit([normalize(x) for x in train.listings.values()])
        metrics, _ = retrieval_evaluation(
            index, validation, config.retrieval_k, output / f"{name}.jsonl"
        )
        if name != "lexical":
            metrics["vocabulary_fit"] = (
                "fixed pretrained encoder; hybrid lexical vocabulary uses training"
            )
            metrics["faiss_index_bytes"] = (
                index.index_bytes if name == "semantic" else index.semantic.index_bytes
            )
        report["channels"][name] = metrics
    write_json(output / "retrieval.json", report)
    return report
