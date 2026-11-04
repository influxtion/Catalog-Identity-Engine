import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from catalog_identity.evaluation.config import RunConfig
from catalog_identity.evaluation.semantic_screening import (
    screen_semantic,
    select_semantic_candidate,
)
from catalog_identity.evaluation.synthetic import write_synthetic
from catalog_identity.ingestion.loaders import load_wdc_pairs
from catalog_identity.matching.learned import LearnedMatcher
from catalog_identity.matching.semantic_features import (
    DIMENSION,
    MODEL,
    REVISION,
    EmbeddingCache,
    SemanticInteractionFeatures,
)
from catalog_identity.schema import Listing


class FakeEncoder:
    """Deterministic test vectors; no optional runtime or pretrained weights."""

    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.texts = []
        self.metadata = {
            "model": MODEL,
            "revision": REVISION,
            "dimension": DIMENSION,
            "max_tokens": 256,
            "device": "cpu",
            "normalized": True,
            "offline": True,
            "trust_remote_code": False,
            "asset_sha256": {"fictional-test-weights": "0" * 64},
        }

    def encode(self, texts):
        assert all(texts)
        self.texts.extend(texts)
        result = []
        for text in texts:
            if text in self.mapping:
                result.append(self.mapping[text])
            else:
                seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
                vector = np.random.default_rng(seed).normal(size=DIMENSION).astype(np.float32)
                result.append(vector / np.linalg.norm(vector))
        return np.asarray(result)

    def token_lengths(self, texts):
        return [len(text.split()) + 2 for text in texts]


def basis(index):
    vector = np.zeros(DIMENSION, dtype=np.float32)
    vector[index] = 1
    return vector


def test_hand_vectors_symmetry_availability_and_interactions(tmp_path):
    paths = write_synthetic(tmp_path / "fixture")
    train = load_wdc_pairs(paths["train"], "train")
    cache = EmbeddingCache(tmp_path / "cache", FakeEncoder({"alpha": basis(0), "beta": basis(1)}))
    features = SemanticInteractionFeatures(cache).fit(train)
    a = Listing(source="fiction", source_id="a", title="alpha", description="beta")
    b = Listing(source="fiction", source_id="b", title="beta")
    records = {a.key: a, b.key: b}
    x = features.transform(records, [(a.key, b.key)])
    np.testing.assert_allclose(x, features.transform(records, [(b.key, a.key)]))
    evidence = dict(zip(features.names, x[0], strict=True))
    assert evidence["semantic_title_cosine"] == 0
    assert evidence["semantic_description_cosine"] == 0
    assert evidence["semantic_description_both_available"] == 0
    assert evidence["semantic_description_one_sided"] == 1
    assert evidence["semantic_cross_cosine_min"] == 0
    assert evidence["semantic_cross_cosine_max"] == 1
    assert evidence["semantic_cross_both_available"] == 0
    assert evidence["semantic_cross_any_available"] == 1
    assert evidence["semantic_title_absdiff_0"] == evidence["semantic_title_absdiff_1"] == 1
    assert evidence["semantic_title_product_0"] == 0
    assert features.transform({}, []).shape == (0, len(features.names))
    empty = b.model_copy(update={"title": ""})
    row = features.transform({a.key: a, empty.key: empty}, [(a.key, empty.key)])[0]
    assert np.isfinite(row).all()
    assert dict(zip(features.names, row, strict=True))["semantic_title_one_sided"] == 1
    assert "" not in cache.encoder.texts


def test_cache_hits_keys_and_population_counts(tmp_path, monkeypatch):
    encoder = FakeEncoder()
    cache = EmbeddingCache(tmp_path, encoder)
    values, counts = cache.vectors("title", ["alpha", "", "alpha"])
    assert encoder.texts == ["alpha"]
    assert counts.tolist() == [3, 0, 3]
    assert not values[1].any()
    cache.vectors("title", ["alpha"])
    assert encoder.texts == ["alpha"]
    other = EmbeddingCache(tmp_path, FakeEncoder())
    np.testing.assert_allclose(values, other.vectors("title", ["alpha", "", "alpha"])[0])
    assert not other.encoder.texts and other.stats["disk_hits"] >= 1
    assert cache.key("title", "alpha") != cache.key("description", "alpha")
    assert cache.key("title", "alpha") != cache.key("title", "beta")
    before = cache.key("title", "alpha")
    import catalog_identity.matching.semantic_features as module

    monkeypatch.setattr(module, "PREPROCESSING", "changed")
    assert cache.key("title", "alpha") != before
    before = cache.key("title", "alpha")
    encoder.metadata["asset_sha256"]["fictional-test-weights"] = "1" * 64
    assert cache.key("title", "alpha") != before
    listing = Listing(source="fiction", source_id="x", title=" ", description="word " * 300)
    report = cache.prepare({listing.key: listing})
    assert report["fields"]["title"]["missing"] == 1
    assert report["fields"]["description"]["truncated"] == 1


def test_cache_memory_is_bounded_and_eviction_reloads_without_encoding(tmp_path):
    encoder = FakeEncoder()
    cache = EmbeddingCache(tmp_path, encoder, max_memory_entries=1)
    values, _ = cache.vectors("title", ["alpha", "beta", "alpha"])
    assert len(cache.memory) == 1
    assert encoder.texts == ["alpha", "beta"]
    np.testing.assert_allclose(values[0], cache.vectors("title", ["alpha"])[0][0])
    assert len(cache.memory) == 1 and cache.stats["disk_hits"] == 1
    assert encoder.texts == ["alpha", "beta"]


@pytest.mark.parametrize("kind", ["nan", "zero", "dimension", "dtype"])
def test_invalid_encoder_vectors_fail(tmp_path, kind):
    vector = basis(0)
    if kind == "nan":
        vector[0] = np.nan
    elif kind == "zero":
        vector[:] = 0
    elif kind == "dimension":
        vector = vector[:-1]
    else:
        vector = vector.astype(np.float64)
    with pytest.raises(ValueError, match="embedding|encoder output"):
        EmbeddingCache(tmp_path, FakeEncoder({"bad": vector})).vectors("title", ["bad"])


def test_revision_corrupt_cache_and_metadata_fail_clearly(tmp_path):
    encoder = FakeEncoder()
    encoder.metadata["revision"] = "f" * 40
    with pytest.raises(ValueError, match="identity"):
        EmbeddingCache(tmp_path, encoder).vectors("title", ["alpha"])
    cache = EmbeddingCache(tmp_path, FakeEncoder())
    cache.vectors("title", ["alpha"])
    path = tmp_path / f"{cache.key('title', 'alpha')}.npz"
    with path.open("wb") as stream:
        np.savez(stream, descriptor=np.asarray("wrong"), vector=basis(0), tokens=np.asarray(3))
    with pytest.raises(ValueError, match="invalid embedding cache.*identity"):
        EmbeddingCache(tmp_path, FakeEncoder()).vectors("title", ["alpha"])
    truncated = path.read_bytes()[:50]
    path.write_bytes(truncated)
    with pytest.raises(ValueError, match="invalid embedding cache"):
        EmbeddingCache(tmp_path, FakeEncoder()).vectors("title", ["alpha"])
    path.write_bytes(b"not a numpy archive")
    with pytest.raises(ValueError, match="invalid embedding cache"):
        EmbeddingCache(tmp_path, FakeEncoder()).vectors("title", ["alpha"])


@pytest.mark.parametrize("version", ["5", "6"])
def test_model_roundtrip_vocabulary_and_encoder_requirements(tmp_path, version):
    paths = write_synthetic(tmp_path / "fixture")
    train = load_wdc_pairs(paths["train"], "train")
    validation = load_wdc_pairs(paths["validation"], "validation")
    cache = EmbeddingCache(tmp_path / "cache", FakeEncoder())
    model = LearnedMatcher(learner="logistic", feature_version=version, embedding_cache=cache).fit(
        train
    )
    vocabulary = dict(model.features.word_vectorizer.vocabulary_)
    expected = model.predict(validation)
    assert model.features.word_vectorizer.vocabulary_ == vocabulary
    model.save(tmp_path / "model")
    restored = LearnedMatcher.load(tmp_path / "model", embedding_cache=cache)
    assert restored.features.names == model.features.names
    np.testing.assert_allclose(expected, restored.predict(validation))
    changed = FakeEncoder()
    changed.metadata["asset_sha256"]["fictional-test-weights"] = "1" * 64
    with pytest.raises(ValueError, match="requirements"):
        LearnedMatcher.load(
            tmp_path / "model", embedding_cache=EmbeddingCache(tmp_path / "other", changed)
        )
    assert not hasattr(cache.encoder, "fit")


def test_legacy_imports_do_not_require_optional_runtime():
    code = """
import importlib.abc
import sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'sentence_transformers'}:
            raise AssertionError('optional runtime imported: ' + fullname)
sys.meta_path.insert(0, Reject())
from catalog_identity.matching.learned import LearnedMatcher
for version in ('2', '3', '4'):
    LearnedMatcher(feature_version=version)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_semantic_selection_fixed_gates_and_ties():
    def row(name, recalls, ap=0.9, support=3):
        return {
            "name": name,
            "folds": [{"envelopes": {"0.99": {"weighted_recall": r}}} for r in recalls],
            "mean_weighted_recall_99": float(np.mean(recalls)),
            "mean_weighted_average_precision": ap,
            "supported_folds_99": support,
        }

    control = row("base", [0.2, 0.2, 0.2])
    assert select_semantic_candidate([control, row("small-gain", [0.21, 0.21, 0.21])]) == "base"
    assert select_semantic_candidate([control, row("one-fold", [0.9, 0.1, 0.1])]) == "base"
    assert (
        select_semantic_candidate([control, row("ap-collapse", [0.3, 0.3, 0.3], ap=0.88)]) == "base"
    )
    assert (
        select_semantic_candidate([control, row("unsupported", [0.3, 0.3, 0.3], support=2)])
        == "base"
    )
    assert (
        select_semantic_candidate([control, row("v5", [0.3, 0.3, 0.3]), row("v6", [0.3, 0.3, 0.3])])
        == "v5"
    )
    assert (
        select_semantic_candidate([control, row("v5", [0.3, 0.3, 0.3]), row("v6", [0.4, 0.4, 0.4])])
        == "v6"
    )


def test_semantic_screening_never_loads_hashes_or_encodes_other_splits(tmp_path, monkeypatch):
    pytest.importorskip("lightgbm")
    paths = write_synthetic(tmp_path / "fixture")
    config = RunConfig(
        dataset_name="fiction",
        dataset_version="1",
        synthetic=True,
        train=paths["train"],
        validation=tmp_path / "missing-val",
        test=tmp_path / "missing-test",
    )
    import catalog_identity.evaluation.semantic_screening as module

    real_hash = module.file_hash
    real_load = module.load_wdc_pairs

    def guarded_hash(path):
        assert path == config.train
        return real_hash(path)

    def guarded_load(path, split):
        assert path == config.train and split == "train"
        return real_load(path, split)

    monkeypatch.setattr(module, "file_hash", guarded_hash)
    monkeypatch.setattr(module, "load_wdc_pairs", guarded_load)
    cache = EmbeddingCache(tmp_path / "cache", FakeEncoder())
    report = screen_semantic(config, tmp_path / "run", embedding_cache=cache)
    assert not report["validation_opened"] and not report["test_opened"]
    assert len(report["candidates"]) == 3
    assert all(len(candidate["folds"]) == 3 for candidate in report["candidates"])
    train = load_wdc_pairs(config.train, "train")
    from catalog_identity.normalization import normalize

    permitted = {
        getattr(normalize(listing), field)
        for listing in train.listings.values()
        for field in ("title", "description")
    }
    assert set(cache.encoder.texts) <= permitted
    # Frozen vectors are encoded once per unique content/field, never refit by fold.
    assert cache.stats["encoded_texts"] == sum(
        len({getattr(normalize(listing), field) for listing in train.listings.values()} - {""})
        for field in ("title", "description")
    )
    assert json.loads((tmp_path / "run/screening.json").read_text())["inputs"].keys() == {"train"}
