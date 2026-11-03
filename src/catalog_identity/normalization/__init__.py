"""Conservative normalization with raw observations and change provenance."""

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from catalog_identity.schema import Listing

NORMALIZATION_VERSION = "1"


def normalize_text(value: str | None) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = value.translate(str.maketrans({"–": "-", "—": "-", "−": "-", "’": "'"}))
    return " ".join(value.split())


def normalize_model(value: str | None) -> str:
    # Preserve slash, decimal point and plus: A/B, A.B and AB can differ.
    return re.sub(r"[\s-]+", "", normalize_text(value))


def normalize_gtin(value: str | None) -> str | None:
    digits = re.sub(r"[\s-]+", "", unicodedata.normalize("NFKC", value or ""))
    if not digits.isascii() or not digits.isdigit() or len(digits) not in (8, 12, 13, 14):
        return None
    if len(set(digits)) == 1:
        return None
    checksum = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(digits[-2::-1]))
    return digits.zfill(14) if (checksum + int(digits[-1])) % 10 == 0 else None


def normalize_attribute(name: str, value: str | None) -> str:
    value = normalize_text(value)
    if name == "color":
        return {"grey": "gray"}.get(value, value)
    if name == "capacity":
        # Canonicalize spacing only; GB and GiB are deliberately distinct.
        return re.sub(r"^(\d+(?:\.\d+)?)\s*(gb|tb|mb|gib|tib|mib)$", r"\1 \2", value)
    if name == "weight":
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(kg|g)", value)
        if match:
            grams = Decimal(match[1]) * (1000 if match[2] == "kg" else 1)
            return f"{grams.normalize():f} g"
    return value


@dataclass(frozen=True)
class Change:
    field: str
    before: str | None
    after: str | None
    rule: str


@dataclass(frozen=True)
class NormalizedListing:
    original: Listing
    title: str
    description: str
    brand: str
    category: str
    model: str
    gtin: str | None
    attributes: dict[str, str]
    price: Decimal | None
    currency: str
    changes: tuple[Change, ...]
    issues: tuple[str, ...]
    version: str = NORMALIZATION_VERSION

    @property
    def key(self) -> str:
        return self.original.key


def normalize(listing: Listing, brand_aliases: dict[str, str] | None = None) -> NormalizedListing:
    changes: list[Change] = []
    issues: list[str] = []

    def record(field: str, before: str | None, after: str | None, rule: str) -> str | None:
        if before != after:
            changes.append(Change(field, before, after, rule))
        return after

    fields = {}
    for name in ("title", "description", "brand", "category", "model", "currency"):
        before = getattr(listing, name)
        after = normalize_model(before) if name == "model" else normalize_text(before)
        fields[name] = record(name, before, after, "unicode_case_whitespace")
    aliases = {normalize_text(k): normalize_text(v) for k, v in (brand_aliases or {}).items()}
    brand = fields["brand"]
    if brand in aliases:
        fields["brand"] = record("brand", brand, aliases[brand], "configured_brand_alias")
    gtin = normalize_gtin(listing.gtin)
    record("gtin", listing.gtin, gtin, "gtin_checksum_and_padding")
    if listing.gtin and not gtin:
        issues.append("invalid_gtin")

    attributes: dict[str, str] = {}
    for raw_name, raw_value in listing.attributes.items():
        name = normalize_text(raw_name)
        if not name or name in attributes:
            raise ValueError(f"empty or duplicate normalized attribute: {raw_name!r}")
        value = normalize_attribute(name, raw_value)
        attributes[name] = value
        record(f"attributes.{name}", raw_value, value, "attribute_value")
        record(f"attribute_key.{raw_name}", raw_name, name, "unicode_case_whitespace")

    price = None
    if listing.price and listing.price.strip():
        try:
            price = Decimal(listing.price)
            if not price.is_finite() or price < 0:
                raise InvalidOperation
        except InvalidOperation:
            price = None
            issues.append("unparsed_price")
    return NormalizedListing(
        original=listing,
        **fields,
        gtin=gtin,
        attributes=attributes,
        price=price,
        changes=tuple(changes),
        issues=tuple(issues),
    )
