import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_cross_encoder import FakeTokenizer

from catalog_identity.evaluation.full_encoder import FULL_PROTOCOL
from catalog_identity.ingestion.loaders import BenchmarkSplit, LabeledPair
from catalog_identity.matching.full_encoder import FullMatcher, validate_training_record
from catalog_identity.schema import Listing


def test_full_module_import_does_not_require_optional_torch_runtime():
    script = """
import sys
class BlockOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'sentence_transformers'}:
            raise AssertionError('optional runtime imported: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import catalog_identity.matching.full_encoder
import catalog_identity.matching.learned
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def tiny_full(device="cpu", tokenizer=None):
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
    return FullMatcher(
        transformers.BertModel(config),
        tokenizer or FakeTokenizer(),
        {"fictional": True},
        device=device,
    )


def fictional_split(count=35):
    listings, pairs, entities = {}, [], {}
    for i in range(count):
        left, right, label = f"{i}-a", f"{i}-b", i % 2
        for key, value in ((left, 0), (right, 0 if label else 1)):
            listings[key] = Listing(
                source="fiction", source_id=key, title=f"invented widget {value}"
            )
        pairs.append(LabeledPair(left, right, label, f"pair-{i}"))
        entities[left] = f"entity-{i}"
        entities[right] = entities[left] if label else f"other-{i}"
    return BenchmarkSplit("train", listings, tuple(pairs), entities)


def test_full_protocol_separate_from_original_and_no_numeric_evidence():
    from catalog_identity.evaluation.cross_encoder import ORIGINAL_PROTOCOL

    assert ORIGINAL_PROTOCOL.screening_limit == 5400
    assert ORIGINAL_PROTOCOL.architecture["version"] == "joint-minilm-v1"
    assert FULL_PROTOCOL.screening_limit == 1800 and FULL_PROTOCOL.full_fit_limit == 1200
    model = tiny_full()
    split = fictional_split()
    assert model.fit_features(split).shape == (35, 0)
    assert not hasattr(model, "features") and not hasattr(model, "scaler")
    with pytest.raises(ValueError, match="numeric"):
        model.prepare(split, np.ones((35, 1)))
    optimizer = model.optimizer()
    groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    for name, parameter in model.network.named_parameters():
        group = groups[id(parameter)]
        assert parameter.requires_grad and group["lr"] == 2e-5
        assert group["weight_decay"] == (
            0 if name.endswith(".bias") or "LayerNorm." in name else 0.01
        )
    assert model.network.backbone.pooler is None
    assert not any("pooler" in n for n, _ in model.network.named_parameters())


def test_registered_training_learns_simple_fictional_distinction():
    """Known linear signal, six differentiable toy layers, unchanged registered optimizer."""
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class ToyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=12, type_vocab_size=2)
            self.embeddings = nn.Module()
            self.embeddings.word_embeddings = nn.Embedding(32, 12)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList()
            for _ in range(6):
                layer = nn.Module()
                layer.attention = nn.Module()
                layer.attention.self = nn.Module()
                layer.attention.self.query = nn.Linear(12, 12)
                nn.init.eye_(layer.attention.self.query.weight)
                nn.init.zeros_(layer.attention.self.query.bias)
                self.encoder.layer.append(layer)

        def forward(self, input_ids, **kwargs):
            hidden = self.embeddings.word_embeddings(input_ids[:, 0])
            for layer in self.encoder.layer:
                hidden = layer.attention.self.query(hidden)
            return SimpleNamespace(last_hidden_state=hidden[:, None, :])

    model = FullMatcher(ToyBackbone(), FakeTokenizer(), {"fictional": True})
    # Symmetric positive/negative latent token signals start orthogonal to the normal
    # initialized classifier. No training constant or classifier initialization changes.
    with torch.no_grad():
        direction = torch.zeros(12)
        direction[0] = model.network.classifier.weight[0, 1]
        direction[1] = -model.network.classifier.weight[0, 0]
        direction = 5 * direction / direction.norm()
        model.network.backbone.embeddings.word_embeddings.weight[4] = direction
        model.network.backbone.embeddings.word_embeddings.weight[5] = -direction
    labels = (np.arange(64) % 2).astype(np.float32)
    tokens = {
        "input_ids": np.where(labels[:, None] == 1, 4, 5).astype(np.int64),
        "attention_mask": np.ones((64, 1), np.int64),
        "token_type_ids": np.zeros((64, 1), np.int64),
    }
    prepared = {
        "orientations": [tokens, tokens],
        "labels": labels,
        "numeric": np.empty((64, 0), np.float32),
        "initial_orientation": np.arange(64) % 2,
    }
    before = model.logits_prepared(prepared)
    model.train_prepared(prepared)
    after = model.logits_prepared(prepared)

    def loss(x):
        return np.mean(np.logaddexp(0, x) - labels * x)

    assert loss(after) < loss(before)
    assert (after[labels == 1] > 0).all() and (after[labels == 0] < 0).all()
    validate_training_record(model.training_record)
    assert all(
        v["changed_elements"] > 0 for v in model.training_record["parameter_changes"].values()
    )
    assert model.training_record["epochs"] == 3 and model.training_record["optimizer_steps"] == 6


def test_full_audits_symmetry_artifact_roundtrip_and_corruption(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    model = tiny_full()
    split = fictional_split()
    prepared = model.prepare(split)
    model.train_prepared(prepared)
    record = model.training_record
    validate_training_record(record)
    assert record["optimizer_steps"] == 6
    assert record["epoch_audits"][0]["final_accumulation_examples"] == 3
    assert record["epoch_audits"][0]["effective_batch_size_counts"] == {32: 1, 3: 1}
    assert all(layer.training for layer in model.network.backbone.encoder.layer)
    assert model.network.backbone.embeddings.training
    reverse = BenchmarkSplit(
        "validation",
        split.listings,
        tuple(LabeledPair(p.right, p.left, p.label, p.pair_id) for p in split.pairs),
        split.entity_ids,
    )
    scores = model.predict(split)
    np.testing.assert_allclose(scores, model.predict(reverse), atol=1e-8)
    directory = tmp_path / "model"
    model.save(directory)
    assert not (directory / "features.json").exists()
    other = tiny_full(tokenizer=model.tokenizer.tokenizer)
    loaded = FullMatcher.load(
        directory, other.network.backbone, model.tokenizer.tokenizer, {"fictional": True}
    )
    assert loaded.device == "cpu"
    np.testing.assert_allclose(scores, loaded.predict(split), atol=1e-8)
    for mutate in (
        lambda r: r["epoch_audits"][0].update(preclip_gradient_norm_mean=float("nan")),
        lambda r: r["epoch_audits"][0].update(learning_rate_min=-1),
        lambda r: r["epoch_audits"][0].update(final_accumulation_examples=32),
        lambda r: r["epoch_audits"][0].update(effective_batch_size_counts={"32": 2}),
        lambda r: r["epoch_audits"][0].update(extra="invalid"),
        lambda r: r["class_counts"].update(positive=100),
        lambda r: r.update(epochs=4),
    ):
        changed = copy.deepcopy(record)
        mutate(changed)
        with pytest.raises(ValueError):
            validate_training_record(changed)
    metadata_path = directory / "matcher.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["training"]["trainable_parameter_count"] += 1
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="parameter counts"):
        FullMatcher.load(
            directory, other.network.backbone, model.tokenizer.tokenizer, {"fictional": True}
        )
    metadata["training"]["trainable_parameter_count"] -= 1
    metadata_path.write_text(json.dumps(metadata))
    from safetensors.torch import load_file, save_file

    weights = load_file(str(directory / "weights.safetensors"))
    weights["classifier.bias"] = torch.full_like(weights["classifier.bias"], float("nan"))
    save_file(weights, str(directory / "weights.safetensors"))
    with pytest.raises(ValueError, match="finite"):
        FullMatcher.load(
            directory, other.network.backbone, model.tokenizer.tokenizer, {"fictional": True}
        )


def test_full_cuda_artifact_defaults_to_cpu(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime/device unavailable")
    model = tiny_full("cuda")
    split = fictional_split(8)
    model.fit(split)
    scores = model.predict(split)
    model.save(tmp_path / "gpu")
    other = tiny_full(tokenizer=model.tokenizer.tokenizer)
    loaded = FullMatcher.load(
        tmp_path / "gpu", other.network.backbone, model.tokenizer.tokenizer, {"fictional": True}
    )
    assert loaded.device == "cpu" and loaded.training_execution["device"] == "cuda:0"
    np.testing.assert_allclose(scores, loaded.predict(split), atol=1e-5, rtol=0)


def test_full_sources_and_gate_reject_prior_partial_protocol(tmp_path, monkeypatch):
    from catalog_identity.evaluation import cross_encoder as runner
    from catalog_identity.evaluation.config import RunConfig
    from catalog_identity.evaluation.full_encoder import screen

    original = {"source_sha256": {p: "same" for p in FULL_PROTOCOL.source_files}}
    current = copy.deepcopy(original)
    monkeypatch.setattr(runner, "implementation_manifest", lambda: current)
    runner.require_same_sources(original, FULL_PROTOCOL.source_files)
    current["source_sha256"]["matching/full_encoder.py"] = "changed"
    with pytest.raises(ValueError, match="implementation changed"):
        runner.require_same_sources(original, FULL_PROTOCOL.source_files)
    timing = {
        "execution": {"fixture": True},
        "gate_passed": True,
        "inputs": {"train": "fixture"},
        "architecture": runner.ARCHITECTURE,
    }
    path = tmp_path / "timing.json"
    path.write_text(json.dumps(timing))
    monkeypatch.setattr(runner, "configure_execution", lambda _: {"fixture": True})
    monkeypatch.setattr(runner, "require_same_execution", lambda a, b: None)

    def train_hash_only(path):
        assert str(path).endswith("train.jsonl")
        return "fixture"

    monkeypatch.setattr(runner, "file_hash", train_hash_only)
    config = RunConfig(
        dataset_name="fictional",
        dataset_version="1",
        train=Path("train.jsonl"),
        validation=Path("NONEXISTENT-validation"),
        test=Path("NONEXISTENT-test"),
        synthetic=True,
    )
    with pytest.raises(ValueError, match="architecture differs"):
        screen(config, tmp_path / "out", path, Path("unread-control"))


def test_full_timing_only_reads_training_and_measures_full_preparation(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("psutil")
    from catalog_identity.evaluation import cross_encoder as runner
    from catalog_identity.evaluation.config import RunConfig
    from catalog_identity.evaluation.full_encoder import pilot, screen
    from catalog_identity.evaluation.synthetic import write_synthetic

    paths = write_synthetic(tmp_path / "fixture")
    config = RunConfig(
        dataset_name="fiction",
        dataset_version="1",
        synthetic=True,
        train=paths["train"],
        validation=tmp_path / "absent-validation",
        test=tmp_path / "absent-test",
    )
    aggregate = tmp_path / "prior.json"
    aggregate.write_text(
        json.dumps(
            {
                "development": {
                    "inputs": {"train": runner.file_hash(config.train), "validation": "a" * 64},
                    "calibration_pairs": 5,
                    "policy_pairs": 5,
                },
                "full_validation_replay": {"classification": {"pairs": 20}},
            }
        )
    )
    original_load, original_hash = runner.load_wdc_pairs, runner.file_hash

    def guarded_load(path, name):
        assert path == config.train and name == "train"
        return original_load(path, name)

    def guarded_hash(path):
        assert path in (config.train, aggregate)
        return original_hash(path)

    monkeypatch.setattr(runner, "load_wdc_pairs", guarded_load)
    monkeypatch.setattr(runner, "file_hash", guarded_hash)
    report = pilot(config, tmp_path / "pilot", aggregate, matcher_factory=tiny_full)
    assert report["architecture"] == FULL_PROTOCOL.architecture
    assert report["full_training_tokenization_seconds"] > 0
    assert report["preparation_projection"] == "measured_full_training_pass"
    assert report["screening_limit_seconds"] == 1800 and report["full_fit_limit_seconds"] == 1200
    assert not report["accuracy_inspected"] and not report["test_opened"]
    assert not report["validation_opened"] and not report["pilot_model_saved"]
    report["gate_passed"] = False
    path = tmp_path / "pilot/timing.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="passing timing gate"):
        screen(config, tmp_path / "out", path, tmp_path / "absent-control")


def test_conditional_validation_policy_and_data_are_frozen(monkeypatch):
    from catalog_identity.evaluation import cross_encoder as runner
    from catalog_identity.evaluation.config import RunConfig

    config = RunConfig(
        dataset_name="fiction",
        dataset_version="1",
        synthetic=True,
        train=Path("train"),
        validation=Path("absent-validation"),
        test=Path("absent-test"),
    )
    policy = {"target_precision": 0.99, "minimum_matches": 5, "validation_sha256": "a" * 64}

    def cannot_hash(path):
        raise AssertionError("screening cannot hash validation")

    monkeypatch.setattr(runner, "file_hash", cannot_hash)
    runner.require_registered_validation_policy(config, policy)
    with pytest.raises(ValueError, match="policy differs"):
        runner.require_registered_validation_policy(
            config.model_copy(update={"target_precision": 0.98}), policy
        )
    with pytest.raises(ValueError, match="policy differs"):
        runner.require_registered_validation_policy(
            config.model_copy(update={"minimum_validation_matches": 1}), policy
        )
    monkeypatch.setattr(runner, "file_hash", lambda path: "b" * 64)
    with pytest.raises(ValueError, match="validation input differs"):
        runner.require_registered_validation_policy(config, policy, verify_input=True)
