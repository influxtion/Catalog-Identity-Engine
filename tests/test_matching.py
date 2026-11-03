import numpy as np
import pytest

from catalog_identity.matching.baseline import decision, select_match_threshold
from catalog_identity.matching.features import conflicts, pair_features
from catalog_identity.normalization import normalize
from catalog_identity.schema import Listing


def record(id, **kwargs):
    return normalize(Listing(source="test", source_id=id, **kwargs))


def test_conflicting_models_are_visible_despite_matching_gtin():
    a = record("a", model="X1", gtin="036000291452", attributes={"color": "red"})
    b = record("b", model="X2", gtin="036000291452", attributes={"color": "blue"})
    features = pair_features(a, b)
    assert features["model_conflict"] == features["gtin_agree"] == 1
    assert features["attribute_conflicts"] == 1
    assert conflicts(a, b) == ("model", "attribute:color")
    assert pair_features(a, b) == pair_features(b, a)


def test_unknown_is_not_agreement_or_conflict():
    a, b = record("a"), record("b", brand="Aurora")
    features = pair_features(a, b)
    assert features["brand_agree"] == features["brand_conflict"] == 0
    assert features["brand_missing"] == 1
    assert conflicts(a, b) == ()


def test_prices_require_same_known_currency_and_positive_values():
    a = record("a", price="10", currency="USD")
    for b in [
        record("b", price="10", currency="CAD"),
        record("c", price="10"),
        record("d", price="0", currency="USD"),
    ]:
        assert pair_features(a, b)["price_comparable"] == 0
    b = record("b", price="20", currency="USD")
    assert pair_features(a, b)["price_log_distance"] == pytest.approx(np.log(2))


def test_threshold_maximizes_recall_under_precision_constraint():
    result = select_match_threshold(np.array([1, 1, 0, 1]), np.array([0.9, 0.8, 0.7, 0.6]), 1.0, 2)
    assert result.threshold == 0.8
    assert result.validation_precision == 1.0
    assert result.validation_recall == pytest.approx(2 / 3)


def test_threshold_abstains_when_ties_or_support_prevent_target():
    result = select_match_threshold(np.array([1, 0]), np.array([0.8, 0.8]), 0.99, 1)
    assert result.threshold is None
    result = select_match_threshold(np.array([1, 0]), np.array([0.8, 0.2]), 0.99, 2)
    assert result.threshold is None
    assert decision(0.9, None, 0.1) == "review"
    assert decision(0.05, None, 0.1) == "reject"
    assert decision(0.9, 0.8, 0.1) == "match"
    with pytest.raises(ValueError, match="below"):
        decision(0.5, 0.8, 0.8)
