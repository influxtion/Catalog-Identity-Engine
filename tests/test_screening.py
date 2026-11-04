import hashlib

import numpy as np
import pytest

from catalog_identity.evaluation.config import RunConfig
from catalog_identity.evaluation.screening import entity_folds, envelope, screen, select_candidate
from catalog_identity.evaluation.synthetic import write_synthetic
from catalog_identity.ingestion.audit import development_overlap
from catalog_identity.ingestion.loaders import BenchmarkSplit, LabeledPair, load_wdc_pairs
from catalog_identity.schema import Listing


def fixture():
    entities = {}
    listings = {}
    pairs = []
    # Two entities per deterministic bucket, each with two listings.
    chosen = {i: [] for i in range(3)}
    for i in range(100):
        entity = f"entity-{i}"
        bucket = int(hashlib.sha256(f"42:{entity}".encode()).hexdigest(), 16) % 3
        if len(chosen[bucket]) < 2:
            chosen[bucket].append(entity)
    for entity in sum(chosen.values(), []):
        for suffix in ("a", "b"):
            listing = Listing(
                source="fiction", source_id=f"{entity}-{suffix}", title=f"Fiction {entity}"
            )
            listings[listing.key] = listing
            entities[listing.key] = entity
    keys = list(listings)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            pairs.append(
                LabeledPair(left, right, int(entities[left] == entities[right]), str(len(pairs)))
            )
    return BenchmarkSplit("train", listings, tuple(pairs), entities)


def test_fold_boundaries_counts_weights_and_determinism():
    train = fixture()
    folds = entity_folds(train)
    assert folds == entity_folds(train)
    for fit, assess, excluded in folds:
        assert not set(fit.entity_ids.values()) & set(assess.entity_ids.values())
        assert len(assess.pairs) == 6 and len(fit.pairs) == 28 and excluded == 32
        assert len(fit.pairs) + len(assess.pairs) + excluded == len(train.pairs)
        labels = np.array([p.label for p in assess.pairs])
        record = envelope(assess, np.ones(6), 0.1, np.where(labels == 1, 1.0, 3.0))
        assert record["raw"]["true_positives"] == 2
        assert record["weighted_counts"]["false_positives"] == 12
        assert record["weighted_precision"] == 2 / 14
        assert record["distinct_matched_positive_entities"] == 2
    with pytest.raises(ValueError, match="complete"):
        entity_folds(
            BenchmarkSplit("train", train.listings, train.pairs, {**train.entity_ids, "extra": "x"})
        )


def test_registered_selection_support_ap_guard_and_tie_order():
    def row(name, recall, ap=0.8, support=3):
        return dict(
            name=name,
            mean_weighted_recall_99=recall,
            mean_weighted_average_precision=ap,
            supported_folds_99=support,
        )

    assert (
        select_candidate(
            [
                row("base", 0.1),
                row("low-ap", 0.9, 0.7),
                row("one-fold", 0.8, support=1),
                row("good", 0.2),
            ]
        )
        == "good"
    )
    assert select_candidate([row("base", 0.1), row("tie", 0.1)]) == "base"
    assert select_candidate([row("base", 0, support=0), row("none", 0, support=0)]) == "base"


def test_screen_never_reads_validation_or_test(tmp_path, monkeypatch):
    paths = write_synthetic(tmp_path / "fixture")
    config = RunConfig(
        dataset_name="fixture",
        dataset_version="1",
        synthetic=True,
        train=paths["train"],
        validation=tmp_path / "absent-val",
        test=tmp_path / "absent-test",
    )
    # Keep the integration test short; the production matrix is fixed and separately asserted.
    import catalog_identity.evaluation.screening as module

    assert len(module.CANDIDATES) == 4
    monkeypatch.setattr(module, "CANDIDATES", (("tiny", "4", 3, 3),))
    report = screen(config, tmp_path / "screen")
    assert not report["validation_opened"] and not report["test_opened"]
    assert set(report["inputs"]) == {"train"}
    assert len(report["candidates"][0]["folds"]) == 3


def test_official_shared_entities_are_reported_without_rejection(tmp_path):
    paths = write_synthetic(tmp_path / "fixture")
    train = load_wdc_pairs(paths["train"], "train")
    # New offers of the same entities: permitted seen-entity development protocol.
    listings = {
        f"new-{k}": v.model_copy(update={"source_id": f"new-{v.source_id}"})
        for k, v in train.listings.items()
    }
    validation = BenchmarkSplit(
        "validation", listings, (), {f"new-{k}": e for k, e in train.entity_ids.items()}
    )
    report = development_overlap(train, validation)
    assert report["listing_overlap"] == 0
    assert report["shared_entities"] == report["validation_entities"]
    assert report["protocol"] == "seen_entity_development"
