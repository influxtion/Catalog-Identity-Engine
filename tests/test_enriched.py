import numpy as np
import pytest

from catalog_identity.evaluation.development import validation_partitions
from catalog_identity.evaluation.synthetic import write_synthetic
from catalog_identity.ingestion.loaders import load_wdc_pairs
from catalog_identity.matching.enriched import TextFeatures, title_hints
from catalog_identity.matching.learned import LearnedMatcher


@pytest.fixture
def splits(tmp_path):
    paths = write_synthetic(tmp_path / "fixture")
    return {name: load_wdc_pairs(path, name) for name, path in paths.items()}


def test_title_hints_keep_capacity_model_color_and_pack_separate():
    hints = title_hints("aster arc-5 2tb grey 3-pack")
    assert hints == {
        "model": {"arc5", "3pack"},
        "capacity": {"2000gb"},
        "color": {"gray"},
        "pack": {"3"},
    }
    assert title_hints("arc-4 128gb")["model"] != title_hints("arc-5 128gb")["model"]


def test_features_are_symmetric_and_vocabulary_is_train_only(splits):
    features = TextFeatures().fit(splits["train"])
    vocab = dict(features.vectorizers["title"].vocabulary_)
    split = splits["test"]
    pairs = [(p.left, p.right) for p in split.pairs]
    a = features.transform(split.listings, pairs)
    b = features.transform(split.listings, [(right, left) for left, right in pairs])
    np.testing.assert_allclose(a, b)
    assert features.vectorizers["title"].vocabulary_ == vocab
    with pytest.raises(ValueError, match="training"):
        TextFeatures().fit(split)


def test_calibration_and_policy_have_disjoint_entities(splits):
    a, b = validation_partitions(splits["validation"])
    assert not a.listings.keys() & b.listings.keys()
    assert not set(a.entity_ids.values()) & set(b.entity_ids.values())
    assert all(p.left in a.listings and p.right in a.listings for p in a.pairs)
    assert len(a.pairs) + len(b.pairs) <= len(splits["validation"].pairs)


@pytest.mark.parametrize("learner", ["logistic", "lightgbm"])
def test_model_roundtrip_and_calibration(tmp_path, splits, learner):
    if learner == "lightgbm":
        pytest.importorskip("lightgbm")
    model = LearnedMatcher(learner=learner).fit(splits["train"])
    calibration, _ = validation_partitions(splits["validation"])
    model.calibrate(calibration)
    expected = model.predict(splits["test"])
    model.save(tmp_path / "model")
    loaded = LearnedMatcher.load(tmp_path / "model")
    np.testing.assert_allclose(loaded.predict(splits["test"]), expected)
    with pytest.raises(ValueError, match="validation"):
        model.calibrate(splits["test"])
