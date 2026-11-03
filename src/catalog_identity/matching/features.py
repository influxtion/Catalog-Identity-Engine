import math
import re

from catalog_identity.normalization import NormalizedListing

FEATURE_VERSION = "1"
FEATURE_NAMES = (
    "title_jaccard",
    "title_number_jaccard",
    "title_length_ratio",
    "description_jaccard",
    "brand_agree",
    "brand_conflict",
    "brand_missing",
    "model_agree",
    "model_conflict",
    "model_missing",
    "gtin_agree",
    "gtin_conflict",
    "gtin_missing",
    "category_agree",
    "category_conflict",
    "category_missing",
    "attribute_agreements",
    "attribute_conflicts",
    "shared_attributes",
    "price_log_distance",
    "price_comparable",
)


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+(?:[.+/-]\w+)*", text))


def agreement(a: str | None, b: str | None) -> tuple[float, float, float]:
    if not a or not b:
        return 0.0, 0.0, 1.0
    return float(a == b), float(a != b), 0.0


def pair_features(a: NormalizedListing, b: NormalizedListing) -> dict[str, float]:
    shared = {
        key
        for key in a.attributes.keys() & b.attributes.keys()
        if a.attributes[key] and b.attributes[key]
    }
    agreements = sum(a.attributes[key] == b.attributes[key] for key in shared)
    comparable = bool(
        a.currency
        and a.currency == b.currency
        and a.price is not None
        and b.price is not None
        and a.price > 0
        and b.price > 0
    )
    # Decimal.ln avoids overflowing when a source provides a very large decimal.
    price_distance = min(float(abs(a.price.ln() - b.price.ln())), 20.0) if comparable else 0.0
    values = [
        jaccard(tokens(a.title), tokens(b.title)),
        jaccard(
            set(re.findall(r"\d+(?:\.\d+)?", a.title)), set(re.findall(r"\d+(?:\.\d+)?", b.title))
        ),
        min(len(a.title), len(b.title)) / max(len(a.title), len(b.title), 1),
        jaccard(tokens(a.description), tokens(b.description)),
        *agreement(a.brand, b.brand),
        *agreement(a.model, b.model),
        *agreement(a.gtin, b.gtin),
        *agreement(a.category, b.category),
        float(agreements),
        float(len(shared) - agreements),
        float(len(shared)),
        price_distance,
        float(comparable),
    ]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("non-finite pair feature")
    return dict(zip(FEATURE_NAMES, values, strict=True))


def conflicts(a: NormalizedListing, b: NormalizedListing) -> tuple[str, ...]:
    """Observed disagreements for inspection, not universal family/variant vetoes."""
    result = [
        name
        for name in ("brand", "model", "gtin")
        if getattr(a, name) and getattr(b, name) and getattr(a, name) != getattr(b, name)
    ]
    result.extend(
        f"attribute:{key}"
        for key in sorted(a.attributes.keys() & b.attributes.keys())
        if a.attributes[key] and b.attributes[key] and a.attributes[key] != b.attributes[key]
    )
    return tuple(result)
