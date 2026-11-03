import json

import numpy as np
import pytest

from catalog_identity.evaluation.synthetic import write_synthetic
from catalog_identity.ingestion.loaders import load_wdc_pairs
from catalog_identity.matching.distinctions import DistinctionFeatures, distinction_hints
from catalog_identity.matching.learned import LearnedMatcher
from catalog_identity.schema import Listing


def test_distinctions_preserve_counts_form_factors_and_editions():
    a = distinction_hints("8gb ddr3l so-dimm 10 shot pack ax316c10f/4 plus")
    b = distinction_hints("8gb ddr3 udimm 100 sheet ax316c10f/8")
    assert a["quantity"] == {"10"} and b["quantity"] == {"100"}
    assert a["form_factor"] == {"sodimm"} and b["form_factor"] == {"udimm"}
    assert a["code"] == {"ax316c10f4"} and b["code"] == {"ax316c10f8"}
    assert a["edition"] == {"plus"}
    assert not distinction_hints("1x128gb 1600mhz pc3-12800 ddr3l")["code"]


@pytest.mark.parametrize("learner", ["logistic", "lightgbm"])
def test_v3_symmetry_roundtrip_and_train_only_vocabulary(tmp_path, learner):
    if learner == "lightgbm":
        pytest.importorskip("lightgbm")
    paths = write_synthetic(tmp_path / "fixture")
    train = load_wdc_pairs(paths["train"], "train")
    validation = load_wdc_pairs(paths["validation"], "validation")
    model = LearnedMatcher(learner=learner, feature_version="3").fit(train)
    vocab = dict(model.features.word_vectorizer.vocabulary_)
    pairs = [(p.left, p.right) for p in validation.pairs]
    x = model.features.transform(validation.listings, pairs)
    reverse = model.features.transform(validation.listings, [(b, a) for a, b in pairs])
    np.testing.assert_allclose(x, reverse)
    assert model.features.word_vectorizer.vocabulary_ == vocab
    model.calibrate(validation)
    model.save(tmp_path / "model")
    restored = LearnedMatcher.load(tmp_path / "model")
    np.testing.assert_allclose(restored.predict(validation), model.predict(validation))
    assert restored.features.version == "3"
    assert restored.features.transform({}, []).shape == (0, len(restored.features.names))
    with pytest.raises(ValueError, match="training"):
        DistinctionFeatures().fit(validation)


def test_missing_hint_is_not_a_conflict_and_one_sided_edition_is_visible(tmp_path):
    paths = write_synthetic(tmp_path / "fixture")
    features = DistinctionFeatures().fit(load_wdc_pairs(paths["train"], "train"))
    a = Listing(source="fixture", source_id="1", title="Aster A10 Special Edition")
    b = Listing(source="fixture", source_id="2", title="Aster A10")
    values = features.transform({a.key: a, b.key: b}, [(a.key, b.key)])[0]
    evidence = dict(zip(features.names, values, strict=True))
    assert evidence["edition_v3_one_sided"] == 1
    assert evidence["edition_v3_conflict"] == 0
    assert evidence["quantity_v3_missing"] == 1


def test_legacy_v2_model_without_version_setting_still_loads(tmp_path):
    paths = write_synthetic(tmp_path / "fixture")
    train = load_wdc_pairs(paths["train"], "train")
    model = LearnedMatcher(learner="logistic").fit(train)
    model.save(tmp_path / "model")
    path = tmp_path / "model/matcher.json"
    data = json.loads(path.read_text())
    data["settings"].pop("feature_version")
    path.write_text(json.dumps(data))
    np.testing.assert_allclose(
        LearnedMatcher.load(path.parent).predict(train), model.predict(train)
    )
