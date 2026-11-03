import csv
import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from catalog_identity.schema import IdentityLevel, Listing

SplitName = Literal["train", "validation", "test"]


def json_rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected an object")
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            yield row


def load_listings(path: Path) -> dict[str, Listing]:
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if "attributes" in row:
                row["attributes"] = json.loads(row["attributes"] or "{}")
    else:
        rows = json_rows(path)
    listings: dict[str, Listing] = {}
    for row in rows:
        listing = Listing.model_validate({**row, "raw": row.get("raw", row)})
        if listing.key in listings:
            raise ValueError(f"duplicate listing: {listing.source}/{listing.source_id}")
        listings[listing.key] = listing
    return listings


@dataclass(frozen=True)
class LabeledPair:
    left: str
    right: str
    label: int
    pair_id: str
    hard_negative: bool = False

    def __post_init__(self):
        if type(self.label) is not int or self.label not in (0, 1):
            raise ValueError("pair label must be integer 0 or 1")
        if self.left == self.right:
            raise ValueError("self pairs are not allowed")


@dataclass(frozen=True)
class BenchmarkSplit:
    name: SplitName
    listings: dict[str, Listing]
    pairs: tuple[LabeledPair, ...]
    entity_ids: dict[str, str]
    identity_level: IdentityLevel = IdentityLevel.BENCHMARK_ENTITY

    def __post_init__(self):
        if self.name not in ("train", "validation", "test"):
            raise ValueError(f"invalid split: {self.name}")
        seen: set[tuple[str, str]] = set()
        pair_ids: set[str] = set()
        for pair in self.pairs:
            if pair.left not in self.listings or pair.right not in self.listings:
                raise ValueError("pair references an unknown listing")
            key = tuple(sorted((pair.left, pair.right)))
            if key in seen or pair.pair_id in pair_ids:
                raise ValueError("duplicate pair or pair_id")
            seen.add(key)
            pair_ids.add(pair.pair_id)
            if pair.left in self.entity_ids and pair.right in self.entity_ids:
                same = self.entity_ids[pair.left] == self.entity_ids[pair.right]
                if same != bool(pair.label):
                    raise ValueError("pair label contradicts entity metadata")


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"expected scalar text, got {type(value).__name__}")
    return str(value)


def load_wdc_pairs(path: Path, split: SplitName) -> BenchmarkSplit:
    listings: dict[str, Listing] = {}
    entities: dict[str, str] = {}
    pairs: list[LabeledPair] = []
    for row in json_rows(path):
        sides = []
        for side in ("left", "right"):
            offer = {
                k.removesuffix(f"_{side}"): v for k, v in row.items() if k.endswith(f"_{side}")
            }
            if offer.get("id") is None:
                raise ValueError(f"missing id_{side}")
            # Gold cluster IDs never enter the Listing, including its raw payload.
            entity = offer.pop("cluster_id", None)
            listing = Listing(
                source="wdc-products",
                source_id=_text(offer["id"]),
                title=_text(offer.get("title")) or "",
                description=_text(offer.get("description")),
                brand=_text(offer.get("brand")),
                price=_text(offer.get("price")),
                currency=_text(offer.get("priceCurrency")),
                raw=offer,
            )
            if listing.key in listings and listings[listing.key] != listing:
                raise ValueError(f"inconsistent repeated offer: {listing.source_id}")
            listings[listing.key] = listing
            if entity is not None:
                entity_id = _text(entity)
                if listing.key in entities and entities[listing.key] != entity_id:
                    raise ValueError(f"inconsistent entity metadata: {listing.source_id}")
                entities[listing.key] = entity_id
            sides.append(listing.key)
        if type(row.get("is_hard_negative", False)) is not bool:
            raise ValueError("is_hard_negative must be boolean")
        pairs.append(
            LabeledPair(
                *sides,
                label=row["label"],
                pair_id=str(row["pair_id"]),
                hard_negative=row.get("is_hard_negative", False),
            )
        )
    return BenchmarkSplit(split, listings, tuple(pairs), entities)
