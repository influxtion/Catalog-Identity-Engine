import pytest
from pydantic import ValidationError

from catalog_identity.normalization import (
    normalize,
    normalize_gtin,
    normalize_model,
    normalize_text,
)
from catalog_identity.schema import Listing


def listing(**kwargs):
    return Listing(source="store", source_id="1", **kwargs)


def test_unicode_and_model_information():
    assert normalize_text("  ＳＯＮＹ\u00a0 XM5—Black  ") == "sony xm5-black"
    assert normalize_model("WH-1000 XM5") == "wh1000xm5"
    assert normalize_model("A.B+") != normalize_model("AB")
    assert normalize_model("XM4") != normalize_model("XM5")


def test_raw_provenance_and_configured_alias():
    original = listing(title="  TEST  ", brand="Test Co", attributes={"Color": "Grey"})
    result = normalize(original, {"test co": "test"})
    assert original.title == "  TEST  "
    assert result.title == "test" and result.brand == "test"
    assert result.attributes == {"color": "gray"}
    assert any(c.rule == "configured_brand_alias" for c in result.changes)


def test_units_do_not_erase_variants():
    result = normalize(listing(attributes={"capacity": "128GB", "weight": "1.5 kg"}))
    assert result.attributes == {"capacity": "128 gb", "weight": "1500 g"}
    assert normalize(listing(attributes={"capacity": "128 GiB"})).attributes != result.attributes


@pytest.mark.parametrize("value", [None, "", "123", "0000000000000", "4006381333932", "ABC123"])
def test_invalid_identifiers_do_not_become_keys(value):
    assert normalize_gtin(value) is None


def test_gtin_padding_and_checksum():
    assert normalize_gtin("4006381333931") == "04006381333931"
    assert normalize_gtin("036000291452") == normalize_gtin("0036000291452")


def test_invalid_price_and_gtin_are_reported():
    result = normalize(listing(price="USD 12.00", gtin="bad"))
    assert result.price is None and result.gtin is None
    assert set(result.issues) == {"unparsed_price", "invalid_gtin"}


def test_missing_and_zero_price_are_distinct():
    assert normalize(listing()).price is None
    assert normalize(listing(price="0")).price == 0


def test_colliding_attribute_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        normalize(listing(attributes={"Color": "red", "color": "blue"}))


def test_listing_id_is_stable_and_namespaced():
    assert listing(title="old").key == listing(title="new").key
    assert listing().key != Listing(source="another", source_id="1").key
    with pytest.raises(ValidationError):
        Listing(source="", source_id="1")
    with pytest.raises(ValidationError):
        Listing(source="store", source_id=1)
