import gzip
import json

import pytest

from catalog_identity.ingestion.audit import audit_splits, profile
from catalog_identity.ingestion.loaders import BenchmarkSplit, load_listings, load_wdc_pairs
from catalog_identity.schema import Listing


def wdc_row(**overrides):
    return {
        "id_left": 1,
        "id_right": 2,
        "title_left": "Fictional lamp",
        "title_right": "Fictional lamp",
        "brand_left": None,
        "cluster_id_left": 10,
        "cluster_id_right": 10,
        "pair_id": "1#2",
        "label": 1,
        **overrides,
    }


def write_rows(path, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_wdc_adapter_keeps_gold_out_of_features(tmp_path):
    path = tmp_path / "train.json.gz"
    write_rows(path, [wdc_row()])
    result = load_wdc_pairs(path, "train")
    assert len(result.listings) == 2 and result.pairs[0].label == 1
    assert set(result.entity_ids.values()) == {"10"}
    assert all("cluster_id" not in x.raw for x in result.listings.values())
    assert all(x.gtin is None for x in result.listings.values())


@pytest.mark.parametrize(
    "rows,pattern",
    [
        ([wdc_row(), wdc_row()], "duplicate"),
        ([wdc_row(), wdc_row(title_left="changed")], "inconsistent"),
        ([wdc_row(label=0)], "contradicts"),
        ([wdc_row(label=True)], "integer"),
        ([wdc_row(id_left=None)], "missing"),
    ],
)
def test_bad_pairs_fail_loudly(tmp_path, rows, pattern):
    path = tmp_path / "pairs.jsonl"
    write_rows(path, rows)
    with pytest.raises(ValueError, match=pattern):
        load_wdc_pairs(path, "train")


def test_jsonl_and_csv_preserve_source_ids(tmp_path):
    path = tmp_path / "listings.jsonl"
    row = {"source": "shop", "source_id": "001", "title": "Lamp"}
    write_rows(path, [row])
    loaded = next(iter(load_listings(path).values()))
    assert loaded.raw == row and loaded.source_id == "001"
    path = tmp_path / "listings.csv"
    path.write_text("source,source_id,title\nshop,001,Lamp\n", encoding="utf-8")
    assert next(iter(load_listings(path).values())).source_id == "001"


def test_duplicate_listing_and_invalid_json(tmp_path):
    path = tmp_path / "listings.jsonl"
    row = {"source": "shop", "source_id": "1"}
    write_rows(path, [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        load_listings(path)
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match=":1:"):
        load_listings(path)


def split(name, source_id, entity):
    listing = Listing(source="test", source_id=source_id, title="lamp")
    return BenchmarkSplit(name, {listing.key: listing}, (), {listing.key: entity})


def test_official_record_disjoint_does_not_mean_entity_disjoint():
    splits = [split("train", "1", "a"), split("validation", "2", "a"), split("test", "3", "a")]
    assert audit_splits(splits)["overlaps"]["train:test"]["entities"] == 1
    with pytest.raises(ValueError, match="test entity leakage"):
        audit_splits(splits, test_entity_disjoint=True)


def test_record_leakage_is_always_rejected():
    with pytest.raises(ValueError, match="listing leakage"):
        audit_splits(
            [split("train", "1", "a"), split("validation", "2", "b"), split("test", "1", "a")]
        )


def test_profile_counts_missing_values():
    listing = Listing(source="test", source_id="1", gtin="invalid")
    report = profile({listing.key: listing})
    assert report["missing"]["brand"] == 1
    assert report["normalization_issues"]["invalid_gtin"] == 1
    assert profile({})["listings"] == 0
