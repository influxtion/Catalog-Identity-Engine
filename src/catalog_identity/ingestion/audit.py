from collections import Counter
from itertools import combinations

import polars as pl

from catalog_identity.ingestion.loaders import BenchmarkSplit
from catalog_identity.normalization import normalize
from catalog_identity.schema import Listing


def profile(listings: dict[str, Listing]) -> dict:
    normalized = [normalize(x) for x in listings.values()]
    columns = ("title", "brand", "category", "model", "gtin", "currency")
    frame = pl.DataFrame(
        {name: [getattr(x, name) or None for x in normalized] for name in columns},
        schema={name: pl.String for name in columns},
    )
    return {
        "listings": len(listings),
        "sources": dict(sorted(Counter(x.source for x in listings.values()).items())),
        "missing": frame.null_count().row(0, named=True),
        "normalization_issues": dict(Counter(issue for x in normalized for issue in x.issues)),
    }


def audit_splits(splits: list[BenchmarkSplit], test_entity_disjoint: bool = False) -> dict:
    if sorted(s.name for s in splits) != ["test", "train", "validation"]:
        raise ValueError("provide exactly one train, validation and test split")
    if len({s.identity_level for s in splits}) != 1:
        raise ValueError("split label semantics differ")
    overlaps = {}
    fingerprints = {
        s.name: {
            (n.title, n.brand, n.model) for n in map(normalize, s.listings.values()) if n.title
        }
        for s in splits
    }
    for a, b in combinations(splits, 2):
        if a.listings.keys() & b.listings.keys():
            raise ValueError(f"listing leakage between {a.name} and {b.name}")
        entity_overlap = set(a.entity_ids.values()) & set(b.entity_ids.values())
        if test_entity_disjoint and "test" in (a.name, b.name):
            if len(a.entity_ids) != len(a.listings) or len(b.entity_ids) != len(b.listings):
                raise ValueError("entity-disjoint audit requires complete entity metadata")
            if entity_overlap:
                raise ValueError(f"test entity leakage between {a.name} and {b.name}")
        overlaps[f"{a.name}:{b.name}"] = {
            "entities": len(entity_overlap),
            "identical_normalized_title_brand_model": len(
                fingerprints[a.name] & fingerprints[b.name]
            ),
        }
    return {
        "test_entity_disjoint_required": test_entity_disjoint,
        "overlaps": overlaps,
        "note": (
            "Text overlaps are warnings, not proof of shared identity. "
            "Near-duplicate audit pending."
        ),
    }
