import json
from pathlib import Path

import numpy as np
import pytest

from catalog_identity.evaluation.cross_encoder import pilot, projected_cost, screen_joint
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


def tiny_matcher(tokenizer=None):
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
        transformers.BertModel(config), tokenizer or FakeTokenizer(), {"fictional": True}
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
    record["gate_passed"] = False
    (tmp_path / "pilot/timing.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="passing timing gate"):
        screen_joint(
            config,
            tmp_path / "blocked",
            tmp_path / "pilot/timing.json",
            tmp_path / "absent-control",
        )
