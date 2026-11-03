from dataclasses import dataclass

from catalog_identity.normalization import NormalizedListing

POLICY_VERSION = "1"
VARIANT_FIELDS = {
    "headphones": ("color",),
    "phone": ("color", "capacity"),
    "apparel": ("color", "size"),
    "storage": ("capacity", "pack_count"),
}


@dataclass(frozen=True)
class HierarchyEvidence:
    family: str
    variant: str
    relation: str
    conflicts: tuple[str, ...]
    missing: tuple[str, ...]
    policy_version: str = POLICY_VERSION


def compare_identity(a: NormalizedListing, b: NormalizedListing) -> HierarchyEvidence:
    family_conflicts = tuple(
        field
        for field in ("brand", "model")
        if getattr(a, field) and getattr(b, field) and getattr(a, field) != getattr(b, field)
    )
    if family_conflicts:
        return HierarchyEvidence(
            "different", "different", "different_product", family_conflicts, ()
        )
    family_missing = tuple(
        field for field in ("brand", "model") if not getattr(a, field) or not getattr(b, field)
    )
    if family_missing:
        return HierarchyEvidence("unknown", "unknown", "uncertain", (), family_missing)
    if not a.category or a.category != b.category or a.category not in VARIANT_FIELDS:
        return HierarchyEvidence("same", "unknown", "uncertain", (), ("category_policy",))
    fields = VARIANT_FIELDS[a.category]
    conflicts = tuple(
        f"attribute:{field}"
        for field in fields
        if a.attributes.get(field)
        and b.attributes.get(field)
        and a.attributes[field] != b.attributes[field]
    )
    missing = tuple(
        f"attribute:{field}"
        for field in fields
        if not a.attributes.get(field) or not b.attributes.get(field)
    )
    if a.gtin and b.gtin and a.gtin != b.gtin:
        conflicts += ("gtin",)
    if conflicts:
        return HierarchyEvidence(
            "same", "different", "same_family_different_variant", conflicts, missing
        )
    if missing:
        return HierarchyEvidence("same", "unknown", "uncertain", (), missing)
    return HierarchyEvidence("same", "same", "same_exact_variant", (), ())
