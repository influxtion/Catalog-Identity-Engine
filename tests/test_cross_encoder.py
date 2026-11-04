import json
from pathlib import Path

import numpy as np
import pytest

from catalog_identity.evaluation.cross_encoder import (
    pilot,
    projected_cost,
    require_same_encoder,
    screen_joint,
)
from catalog_identity.evaluation.runner import file_hash
from catalog_identity.evaluation.synthetic import write_synthetic
from catalog_identity.ingestion.loaders import BenchmarkSplit, LabeledPair, load_wdc_pairs
from catalog_identity.matching.cross_encoder import ARCHITECTURE, JointMatcher, PairTokenizer
from catalog_identity.schema import Listing


class FakeTokenizer:
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.vocabulary = {}

    def encode(self, text, add_special_tokens=False):
        return [
            self.vocabulary.setdefault(word, len(self.vocabulary) + 10) for word in text.split()
        ]

    def num_special_tokens_to_add(self, pair=True):
        return 3

    def prepare_for_model(self, a, pair_ids, **kwargs):
        assert kwargs["truncation"] == "longest_first"
        ids = [1, *a, 2, *pair_ids, 2]
        assert len(ids) <= 256
        mask = [1] * len(ids) + [0] * (256 - len(ids))
        types = [0] * (len(a) + 2) + [1] * (len(pair_ids) + 1) + [0] * (256 - len(ids))
        return {
            "input_ids": ids + [0] * (256 - len(ids)),
            "attention_mask": mask,
            "token_type_ids": types,
        }

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True)
        (Path(path) / "tokenizer.json").write_text(json.dumps(self.vocabulary))


def test_same_declared_revision_cannot_hide_changed_encoder_assets():
    expected = {
        "revision": "unchanged",
        "asset_sha256": {"weights": "original", "tokenizer": "original"},
    }
    require_same_encoder(expected, json.loads(json.dumps(expected)))
    for key in ("weights", "tokenizer"):
        changed = json.loads(json.dumps(expected))
        changed["asset_sha256"][key] = "modified"
        with pytest.raises(ValueError, match="encoder assets"):
            require_same_encoder(expected, changed)


def test_frozen_sources_bind_model_protocol_but_allow_unrelated_service_edits(monkeypatch):
    import catalog_identity.evaluation.cross_encoder as module

    previous = {"source_sha256": {path: "stable" for path in module.JOINT_SOURCE_FILES}}
    current = json.loads(json.dumps(previous))
    current["source_sha256"]["api/app.py"] = "unrelated-service-change"
    monkeypatch.setattr(module, "implementation_manifest", lambda: current)
    module.require_same_sources(previous)
    current["source_sha256"]["matching/cross_encoder.py"] = "changed"
    with pytest.raises(ValueError, match="implementation changed"):
        module.require_same_sources(previous)


def test_joint_artifact_policy_preserves_thresholds_and_provenance(tmp_path):
    from catalog_identity.evaluation.cross_encoder import save_joint_policy

    report = {
        "match_policy": {"threshold": 0.9},
        "rejection_policy": {"threshold": 0.8},
        "reject_threshold": 0.2,
        "inputs": {"train": "hash", "validation": "hash"},
        "synthetic": True,
        "development_overlap": {"shared_entities": 0},
        "execution": {"fingerprint": "fixture"},
        "selection_frozen_sha256": "screen-hash",
        "implementation": {"git": {"commit": "fixture"}},
        "training": {"epochs": 3},
    }
    save_joint_policy(tmp_path, report)
    policy = json.loads((tmp_path / "policy.json").read_text())
    assert policy["match"]["threshold"] == 0.9 and policy["reject_threshold"] == 0.2
    assert policy["identity_level"] == "benchmark_entity" and policy["development_only"]
    assert policy["selection_frozen_sha256"] == "screen-hash"


def tiny_matcher(tokenizer=None, device="cpu"):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    config = transformers.BertConfig(
        vocab_size=4096,
        hidden_size=12,
        num_hidden_layers=6,
        num_attention_heads=3,
        intermediate_size=16,
        max_position_embeddings=256,
    )
    return JointMatcher(
        transformers.BertModel(config),
        tokenizer or FakeTokenizer(),
        {"fictional": True},
        device=device,
    )


def test_balanced_pair_tokens_preserve_both_titles_and_account_for_labels():
    tokenizer = PairTokenizer(FakeTokenizer())
    a = Listing(
        source="fiction",
        source_id="a",
        title="lefttitle " * 200,
        brand="brand " * 20,
        description="long " * 1000,
    )
    b = Listing(source="fiction", source_id="b", title="righttitle", description="other " * 1000)
    row, stats = tokenizer.pair(a, b)
    assert len(row["input_ids"]) == 256 and row["attention_mask"].sum() == 256
    assert tokenizer.tokenizer.vocabulary["lefttitle"] in row["input_ids"]
    assert tokenizer.tokenizer.vocabulary["righttitle"] in row["input_ids"]
    assert stats["left_truncated"]["title"] and stats["left_truncated"]["description"]
    assert not stats["right_truncated"]["title"]
    empty = b.model_copy(update={"title": "", "description": None})
    padded, _ = tokenizer.pair(empty, empty)
    assert padded["attention_mask"].sum() < 256
    altered = a.model_copy(
        update={
            "source_id": "secret",
            "raw": {"label": 1, "cluster_id": "gold", "is_hard_negative": True},
        }
    )
    np.testing.assert_array_equal(row["input_ids"], tokenizer.pair(altered, b)[0]["input_ids"])


def test_train_only_scaler_frozen_modules_symmetry_and_safe_roundtrip(tmp_path):
    pytest.importorskip("safetensors")
    torch = pytest.importorskip("torch")
    paths = write_synthetic(tmp_path / "fixture")
    train = load_wdc_pairs(paths["train"], "train")
    model = tiny_matcher()
    numeric = model.fit_features(train)
    mean = model.scaler.mean_.copy()
    vocabulary = dict(model.features.word_vectorizer.vocabulary_)
    # Eight fictional pairs exercise gradient wiring, not benchmark accuracy.
    pairs = train.pairs[:8]
    small = BenchmarkSplit("train", train.listings, pairs, train.entity_ids)
    prepared = model.prepare(small, numeric[:8])
    frozen_before = model.network.backbone.embeddings.word_embeddings.weight.detach().clone()
    upper_before = next(model.network.backbone.encoder.layer[4].parameters()).detach().clone()
    head_before = next(model.network.head.parameters()).detach().clone()
    model.train_prepared(prepared)
    assert model.training_record["epochs"] == 3
    assert not model.network.backbone.embeddings.training
    assert all(not layer.training for layer in model.network.backbone.encoder.layer[:4])
    assert model.network.backbone.encoder.layer[4].training
    assert torch.equal(frozen_before, model.network.backbone.embeddings.word_embeddings.weight)
    assert not torch.equal(upper_before, next(model.network.backbone.encoder.layer[4].parameters()))
    assert not torch.equal(head_before, next(model.network.head.parameters()))
    reverse = BenchmarkSplit(
        "validation",
        train.listings,
        tuple(LabeledPair(p.right, p.left, p.label, p.pair_id) for p in pairs),
        train.entity_ids,
    )
    scores = model.predict(small)
    np.testing.assert_allclose(scores, model.predict(reverse), atol=1e-8)
    np.testing.assert_array_equal(mean, model.scaler.mean_)
    assert vocabulary == model.features.word_vectorizer.vocabulary_
    with pytest.raises(ValueError, match="training"):
        model.fit_features(reverse)
    model.save(tmp_path / "model")
    fresh = tiny_matcher(model.tokenizer.tokenizer)
    restored = JointMatcher.load(
        tmp_path / "model",
        fresh.network.backbone,
        fresh.tokenizer.tokenizer,
        fresh.encoder_metadata,
    )
    np.testing.assert_allclose(scores, restored.predict(small), atol=1e-8)
    metadata_path = tmp_path / "model/matcher.json"
    metadata = json.loads(metadata_path.read_text())
    legacy = dict(metadata)
    legacy.pop("execution")
    metadata_path.write_text(json.dumps(legacy))
    legacy_model = JointMatcher.load(
        tmp_path / "model",
        fresh.network.backbone,
        fresh.tokenizer.tokenizer,
        fresh.encoder_metadata,
    )
    assert legacy_model.device == "cpu" and legacy_model.training_execution is None
    np.testing.assert_allclose(scores, legacy_model.predict(small), atol=1e-8)
    malformed = json.loads(json.dumps(metadata))
    malformed["execution"]["deterministic_algorithms"] = False
    metadata_path.write_text(json.dumps(malformed))
    with pytest.raises(ValueError, match="execution"):
        JointMatcher.load(
            tmp_path / "model",
            fresh.network.backbone,
            fresh.tokenizer.tokenizer,
            fresh.encoder_metadata,
        )
    metadata["architecture"]["head_hidden"] = 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="architecture"):
        JointMatcher.load(
            tmp_path / "model",
            fresh.network.backbone,
            fresh.tokenizer.tokenizer,
            fresh.encoder_metadata,
        )
    metadata["architecture"]["head_hidden"] = 128
    metadata["calibration"] = {"slope": -1.0, "intercept": 0.0}
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="calibration"):
        JointMatcher.load(
            tmp_path / "model",
            fresh.network.backbone,
            fresh.tokenizer.tokenizer,
            fresh.encoder_metadata,
        )
    metadata["calibration"] = None
    metadata_path.write_text(json.dumps(metadata))
    from safetensors.torch import load_file, save_file

    weights_path = tmp_path / "model/weights.safetensors"
    original = weights_path.read_bytes()
    state = load_file(str(weights_path))
    key = next(iter(state))
    for corruption in ("dtype", "nan", "missing"):
        damaged = {k: v.clone() for k, v in state.items()}
        if corruption == "dtype":
            damaged[key] = damaged[key].double()
        elif corruption == "nan":
            damaged[key].flatten()[0] = float("nan")
        else:
            damaged.pop(key)
        save_file(damaged, str(weights_path))
        with pytest.raises(ValueError, match="tensor"):
            JointMatcher.load(
                tmp_path / "model",
                fresh.network.backbone,
                fresh.tokenizer.tokenizer,
                fresh.encoder_metadata,
            )
    weights_path.write_bytes(original)
    tokenizer_path = tmp_path / "model/tokenizer/tokenizer.json"
    tokenizer_path.write_text("{}")
    with pytest.raises(ValueError, match="tokenizer"):
        JointMatcher.load(
            tmp_path / "model",
            fresh.network.backbone,
            fresh.tokenizer.tokenizer,
            fresh.encoder_metadata,
        )


def test_timing_gate_uses_synthetic_labels_and_never_needs_other_data(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("psutil")
    import catalog_identity.evaluation.cross_encoder as module
    from catalog_identity.evaluation.config import RunConfig

    paths = write_synthetic(tmp_path / "fixture")
    config = RunConfig(
        dataset_name="fiction",
        dataset_version="1",
        synthetic=True,
        train=paths["train"],
        validation=tmp_path / "absent-validation",
        test=tmp_path / "absent-test",
    )
    aggregate = tmp_path / "prior-aggregate.json"
    aggregate.write_text(
        json.dumps(
            {
                "development": {
                    "inputs": {"train": file_hash(config.train)},
                    "calibration_pairs": 5,
                    "policy_pairs": 5,
                },
                "full_validation_replay": {"classification": {"pairs": 20}},
            }
        )
    )
    original_loader = module.load_wdc_pairs
    original_hash = module.file_hash

    def guarded_load(path, name):
        assert path == config.train and name == "train"
        return original_loader(path, name)

    def guarded_hash(path):
        assert path in (config.train, aggregate)
        return original_hash(path)

    monkeypatch.setattr(module, "load_wdc_pairs", guarded_load)
    monkeypatch.setattr(module, "file_hash", guarded_hash)
    report = pilot(config, tmp_path / "pilot", aggregate, matcher_factory=tiny_matcher)
    assert not report["accuracy_inspected"] and not report["pilot_model_saved"]
    assert not report["validation_opened"] and not report["test_opened"]
    assert report["measured_microbatches"] == 16
    assert report["padded_sequence_length"] == 256
    assert report["architecture"] == ARCHITECTURE
    assert not list((tmp_path / "pilot").glob("*.safetensors"))
    assert projected_cost(10, 20, 2, 4, 5) == 37.5
    # A failed resource gate stops before attempting a model fit or opening control/data.
    record = json.loads((tmp_path / "pilot/timing.json").read_text())
    from catalog_identity.matching.execution import fingerprint

    changed = json.loads(json.dumps(record))
    changed["execution"]["dependencies"]["fictional-runtime-change"] = "1"
    changed["execution"]["fingerprint"] = fingerprint(changed["execution"])
    (tmp_path / "pilot/timing.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="execution environment"):
        screen_joint(
            config,
            tmp_path / "mismatch",
            tmp_path / "pilot/timing.json",
            tmp_path / "absent-control",
        )
    record["gate_passed"] = False
    (tmp_path / "pilot/timing.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="passing timing gate"):
        screen_joint(
            config,
            tmp_path / "blocked",
            tmp_path / "pilot/timing.json",
            tmp_path / "absent-control",
        )


def test_execution_manifest_strictness_and_no_automatic_cuda_fallback(monkeypatch):
    torch = pytest.importorskip("torch")
    from catalog_identity.matching.execution import (
        configure_execution,
        fingerprint,
        require_same_execution,
        validate_execution,
    )

    manifest = configure_execution("cpu")
    assert manifest["precision"] == {"global": "ieee", "cuda_matmul": "ieee", "cudnn": "ieee"}
    assert manifest["attention"] == "math" and manifest["autocast"] is False
    changed = json.loads(json.dumps(manifest))
    changed["python"] = "different-runtime"
    changed["fingerprint"] = fingerprint(changed)
    with pytest.raises(ValueError, match="environment"):
        require_same_execution(manifest, changed)
    changed["optimizer"]["fused"] = True
    changed["fingerprint"] = fingerprint(changed)
    with pytest.raises(ValueError, match="flags"):
        validate_execution(changed)
    import os

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "wrong-prior-setting")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="initialized"):
        configure_execution("cpu")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == "wrong-prior-setting"
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        configure_execution("cuda")


def test_cuda_routing_roundtrip_and_default_cpu_loading(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime/device unavailable")
    paths = write_synthetic(tmp_path / "fixture")
    train = load_wdc_pairs(paths["train"], "train")
    small = BenchmarkSplit("train", train.listings, train.pairs[:8], train.entity_ids)
    model = tiny_matcher(device="cuda")
    numeric = model.fit_features(small)
    prepared = model.prepare(small, numeric)
    tokens, numeric_tensor = model.batch(prepared, np.arange(8), 0)
    assert tokens["input_ids"].device.type == numeric_tensor.device.type == "cuda"
    model.train_prepared(prepared)
    expected = model.predict(small)
    model.save(tmp_path / "model")
    fresh = tiny_matcher(model.tokenizer.tokenizer)
    gpu = JointMatcher.load(
        tmp_path / "model",
        fresh.network.backbone,
        fresh.tokenizer.tokenizer,
        fresh.encoder_metadata,
        device="cuda",
    )
    np.testing.assert_allclose(expected, gpu.predict(small), atol=1e-5, rtol=1e-5)
    cpu_fresh = tiny_matcher(model.tokenizer.tokenizer)
    cpu = JointMatcher.load(
        tmp_path / "model",
        cpu_fresh.network.backbone,
        cpu_fresh.tokenizer.tokenizer,
        cpu_fresh.encoder_metadata,
    )
    assert cpu.device == "cpu" and cpu.training_execution["device"] == "cuda:0"
    actual = cpu.predict(small)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(expected, actual, atol=1e-5, rtol=1e-5)
