import numpy as np
import pytest

from catalog_identity.evaluation.synthetic import write_synthetic
from catalog_identity.ingestion.loaders import load_wdc_pairs
from catalog_identity.matching.alignment import AlignmentFeatures, code_alignment, parse_text
from catalog_identity.matching.learned import LearnedMatcher
from catalog_identity.schema import Listing


def test_code_separators_preserve_distinguishing_components():
    assert parse_text("AX-120")[0] == parse_text("AX 120")[0] == parse_text("AX120")[0]
    for other in ("AX1200", "AX120/4", "AX120/8", "AX120.1", "AX120+", "AX120gen2"):
        assert parse_text("AX120")[0] != parse_text(other)[0]
    a, b = parse_text("AX120")[0], parse_text("AX1200")[0]
    values = code_alignment(a, b)
    assert values[1] == 1 and values[-1] == 0  # Numeric prefix is not exact support.
    assert code_alignment(a, parse_text("AX120/4")[0])[-1] == 1


def test_typed_units_modules_packs_and_unknowns():
    assert parse_text("1.6GHz")[1]["frequency"] == parse_text("1600 MHz")[1]["frequency"]
    assert parse_text("1600 MHz")[1]["frequency"] != parse_text("1866 MHz")[1]["frequency"]
    assert parse_text("2x8GB")[1]["capacity"] == {"16"}
    assert parse_text("2x8GB")[1]["module_capacity"] == {"8"}
    assert parse_text("2x8GB")[1]["module_count"] == {"2"}
    assert not parse_text("16GB")[1]["module_count"]
    assert parse_text("single pack")[1]["pack"] == {"1"}
    assert parse_text("twin pack")[1]["pack"] == {"2"}
    assert parse_text("pack of 2")[1]["pack"] == {"2"}
    assert parse_text("10 x 20 mm")[1]["dimension"] == parse_text("1 x 2 cm")[1]["dimension"]
    assert parse_text("10 x 20 mm")[1]["dimension"] != parse_text("10 x 21 mm")[1]["dimension"]


@pytest.mark.parametrize("learner", ["logistic", "lightgbm"])
def test_v4_symmetric_finite_train_only_and_roundtrip(tmp_path, learner):
    if learner == "lightgbm":
        pytest.importorskip("lightgbm")
    paths = write_synthetic(tmp_path / "fixture")
    train = load_wdc_pairs(paths["train"], "train")
    validation = load_wdc_pairs(paths["validation"], "validation")
    model = LearnedMatcher(learner=learner, feature_version="4").fit(train)
    vocabulary = dict(model.features.word_vectorizer.vocabulary_)
    pairs = [(p.left, p.right) for p in validation.pairs]
    x = model.features.transform(validation.listings, pairs)
    np.testing.assert_allclose(
        x, model.features.transform(validation.listings, [(b, a) for a, b in pairs])
    )
    assert np.isfinite(x).all()
    assert model.features.word_vectorizer.vocabulary_ == vocabulary
    repeated = LearnedMatcher(learner=learner, feature_version="4").fit(train)
    np.testing.assert_allclose(model.predict(validation), repeated.predict(validation))
    model.save(tmp_path / "model")
    np.testing.assert_allclose(
        model.predict(validation), LearnedMatcher.load(tmp_path / "model").predict(validation)
    )
    assert model.features.transform({}, []).shape == (0, len(model.features.names))
    with pytest.raises(ValueError, match="training"):
        AlignmentFeatures().fit(validation)


def test_description_support_missingness_and_metadata_invariance(tmp_path):
    paths = write_synthetic(tmp_path / "fixture")
    features = AlignmentFeatures().fit(load_wdc_pairs(paths["train"], "train"))
    a = Listing(
        source="fiction",
        source_id="1",
        title="Orin AX120 2x8GB",
        description="Compatible with Orin AX120/4 and AX120/8",
    )
    b = Listing(source="fiction", source_id="2", title="Orin AX120 16GB")

    def values(left, right):
        return features.transform({left.key: left, right.key: right}, [(left.key, right.key)])[0]

    row = values(a, b)
    evidence = dict(zip(features.names, row, strict=True))
    assert evidence["typed_module_count_one_sided"] == 1
    assert evidence["typed_module_count_conflict"] == 0
    assert evidence["typed_capacity_jaccard"] == 1
    assert evidence["cross_description_one_sided"] == 1
    assert evidence["cross_token_max"] > 0
    # IDs, raw benchmark metadata and hard-negative flags cannot become features.
    altered = a.model_copy(
        update={
            "source_id": "unseen-secret",
            "raw": {"cluster_id": "gold", "label": 0, "is_hard_negative": True},
        }
    )
    np.testing.assert_allclose(row, values(altered, b))
    empty = b.model_copy(update={"title": "", "description": None})
    assert np.isfinite(values(a, empty)).all()


def test_unseen_tokens_retain_weight_without_fitting_on_assessment(tmp_path):
    paths = write_synthetic(tmp_path / "fixture")
    features = AlignmentFeatures().fit(load_wdc_pairs(paths["train"], "train"))
    vocabulary = dict(features.word_vectorizer.vocabulary_)
    known = next(w for w in vocabulary if " " not in w and w.isalpha())
    a = Listing(source="fiction", source_id="a", title=f"{known} novelzzax120")
    b = Listing(source="fiction", source_id="b", title=f"{known} novelzzax990")

    def containment(right):
        row = features.transform({a.key: a, right.key: right}, [(a.key, right.key)])[0]
        return dict(zip(features.names, row, strict=True))["idf_containment_min"]

    assert 0 < containment(b) < 1
    assert containment(b.model_copy(update={"title": a.title})) == 1
    assert features.word_vectorizer.vocabulary_ == vocabulary
