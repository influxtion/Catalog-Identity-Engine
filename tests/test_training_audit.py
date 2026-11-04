import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score
from test_cross_encoder import FakeTokenizer

from catalog_identity.evaluation.config import RunConfig
from catalog_identity.evaluation.metrics import classification_metrics
from catalog_identity.evaluation.training_audit import run_audit, summarize_inputs
from catalog_identity.evaluation.training_review import build_review, verify_scores
from catalog_identity.ingestion.loaders import BenchmarkSplit, LabeledPair
from catalog_identity.matching.cross_encoder import PairTokenizer
from catalog_identity.schema import Listing


def fixture(titles=None):
    titles = titles or ["widget"] * 6
    listings = {
        str(i): Listing(
            source="fiction",
            source_id=str(i),
            title=title,
            description="same" if i < 4 else f"different {i}",
        )
        for i, title in enumerate(titles)
    }
    pairs = (
        LabeledPair("0", "1", 1, "a"),
        LabeledPair("2", "3", 0, "b"),
        LabeledPair("4", "5", 1, "c"),
    )
    entities = dict(zip(listings, ("a", "a", "b", "c", "d", "d"), strict=True))
    return BenchmarkSplit("train", listings, pairs, entities)


def test_collisions_are_representation_specific_and_label_preserving():
    split = fixture()
    report, details = summarize_inputs(split)
    titles, full = (report["pair_collisions"][n] for n in ("normalized_title", "normalized_text"))
    assert titles["mixed_label_groups"] == 1 and titles["affected_pairs"] == 3
    assert titles["minimum_observed_errors"] == 1
    assert full["affected_pairs"] == 2 and full["minimum_observed_errors"] == 1
    assert report["text_collision_pairs_resolved_by_brand_description"] == 1
    assert report["listing_collisions"]["normalized_text"]["cross_entity_groups"] == 1
    assert report["nonempty_listing_fields"]["gtin"] == 0
    assert "conflicting_text_inputs" not in details["c"]["tags"]
    assert [p.label for p in split.pairs] == [1, 0, 1]
    assert report["token_audit"] is None
    reverse = BenchmarkSplit(
        "train",
        split.listings,
        tuple(LabeledPair(p.right, p.left, p.label, p.pair_id) for p in split.pairs),
        split.entity_ids,
    )
    assert summarize_inputs(reverse)[0] == report


def test_normalization_collision_is_not_a_raw_collision():
    report, _ = summarize_inputs(fixture(["WIDGET", "WIDGET", "widget", "widget", "x", "y"]))
    assert report["pair_collisions"]["raw_text"]["mixed_label_groups"] == 0
    assert report["pair_collisions"]["normalized_text"]["mixed_label_groups"] == 1


def test_budget_removes_distinguishing_title_tokens():
    prefix = "prefix " * 100
    split = fixture([prefix + v for v in ("same", "same", "left", "right", "other", "other")])
    report, details = summarize_inputs(split, PairTokenizer(FakeTokenizer()))
    assert report["pair_collisions"]["untruncated_tokens"]["mixed_label_groups"] == 0
    assert report["pair_collisions"]["pair_tokens_256"]["mixed_label_groups"] == 1
    assert report["token_audit"]["new_mixed_pairs_after_budget"] == 2
    assert report["token_audit"]["pairs_with_truncation_by_label"]["0"]["title"] == 1
    assert "truncated_positive" in details["a"]["tags"]


def test_review_is_deterministic_blinded_and_entity_capped():
    split = fixture()
    _, details = summarize_inputs(split)
    blind, key, sampling = build_review(split, details, quota=1, entity_cap=1)
    assert (blind, key, sampling) == build_review(split, details, quota=1, entity_cap=1)
    reversed_split = BenchmarkSplit(
        "train", split.listings, tuple(reversed(split.pairs)), split.entity_ids
    )
    assert (blind, key, sampling) == build_review(reversed_split, details, quota=1, entity_cap=1)
    counts = Counter(e for row in key for e in row["entity_ids"])
    assert max(counts.values()) == sampling["maximum_selected_pairs_per_entity"] == 1
    assert {r["review_id"] for r in blind} == {r["review_id"] for r in key}
    assert all(set(row) == {"review_id", "left", "right", "judgment"} for row in blind)
    for row in blind:
        assert row["judgment"]["benchmark_identity"] is None
        for side in ("left", "right"):
            assert not ({"source_id", "raw", "entity_ids", "label", "score"} & row[side].keys())
    assert "benchmark_label" in key[0] and "primary_stratum" in key[0]


def test_saved_threshold_used_and_control_mismatch_rejected():
    labels = np.asarray([1, 1, 0, 0])
    scores = np.asarray([0.9, 0.6, 0.7, 0.1])
    saved = {
        "envelopes": {
            "0.99": {
                "selection": {"threshold": 0.8},
                "raw": classification_metrics(labels, scores >= 0.8),
            }
        },
        "weighted_average_precision": average_precision_score(
            labels, scores, sample_weight=[1, 1, 3, 3]
        ),
    }
    predicted, _, _ = verify_scores(labels, scores, saved)
    assert predicted.tolist() == [True, False, False, False]
    with pytest.raises(ValueError, match="reproduction differs"):
        verify_scores(labels, np.asarray([0.9, 0.6, 0.95, 0.1]), saved)
    with pytest.raises(ValueError, match="invalid"):
        verify_scores(labels, np.asarray([0.9, 0.6, float("nan"), 0.1]), saved)


def test_only_training_is_loaded_or_hashed_and_local_rows_are_ignored(tmp_path, monkeypatch):
    from catalog_identity.evaluation import training_audit as module

    config = RunConfig(
        dataset_name="fiction",
        dataset_version="1",
        synthetic=True,
        train=Path("train"),
        validation=Path("MISSING-valid"),
        test=Path("MISSING-test"),
    )
    calls = []

    def loader(path, name):
        assert path == config.train and name == "train"
        calls.append("load")
        return fixture()

    def hasher(path):
        assert path == config.train
        calls.append("hash")
        return "a" * 64

    monkeypatch.setattr(module, "load_wdc_pairs", loader)
    monkeypatch.setattr(module, "file_hash", hasher)
    monkeypatch.setattr(module, "implementation_manifest", lambda: {"fictional": True})
    report = run_audit(config, tmp_path / "out")
    assert calls == ["load", "hash"]
    assert not report["validation_opened"] and not report["test_opened"]
    assert "review-key.jsonl" in (tmp_path / "out/.gitignore").read_text()
    assert json.loads((tmp_path / "out/audit.json").read_text())["complete"]
    serialized = json.dumps(report)
    assert "widget" not in serialized and '"pair_id"' not in serialized
    with pytest.raises(ValueError, match="already exists"):
        run_audit(config, tmp_path / "out")
    for name in ("validation", "test"):
        split = fixture()
        with pytest.raises(ValueError, match="training"):
            summarize_inputs(BenchmarkSplit(name, split.listings, split.pairs, split.entity_ids))


def test_control_reproduction_keeps_entities_disjoint_and_does_not_pool_scores(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from test_screening import fixture as entity_fixture

    from catalog_identity.evaluation import training_review as module
    from catalog_identity.evaluation.screening import entity_folds

    split = entity_fixture()
    config = RunConfig(
        dataset_name="fiction",
        dataset_version="1",
        synthetic=True,
        train=Path("not-opened"),
        validation=Path("missing-valid"),
        test=Path("missing-test"),
    )
    saved = []
    for i, (fit, assess, excluded) in enumerate(entity_folds(split)):
        y = np.asarray([p.label for p in assess.pairs])
        saved.append(
            {
                "fold": i,
                "train_pairs": len(fit.pairs),
                "assessment_pairs": len(assess.pairs),
                "boundary_pairs_excluded": excluded,
                "train_entities": 4,
                "assessment_entities": 2,
                "weighted_average_precision": 1.0,
                "envelopes": {
                    "0.99": {"selection": {"threshold": 0.8}, "raw": classification_metrics(y, y)}
                },
            }
        )
    candidate = {
        "settings": {"feature_version": "3"},
        "folds": saved,
        "effective_parameters": {"fictional": True},
    }
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps(candidate))
    monkeypatch.setattr(module, "verify_control", lambda config, previous: previous)
    fits = []

    class Model:
        def __init__(self, **kwargs):
            assert kwargs == candidate["settings"]
            self.booster = SimpleNamespace(params={"fictional": True})

        def fit(self, train):
            assert train.name == "train"
            self.entities = set(train.entity_ids.values())
            fits.append(train)
            return self

        def predict(self, assess, calibrated):
            assert not calibrated and not self.entities & set(assess.entity_ids.values())
            return np.asarray(
                [
                    0.9 if assess.listings[p.left].title == assess.listings[p.right].title else 0.1
                    for p in assess.pairs
                ]
            )

    monkeypatch.setattr(module, "LearnedMatcher", Model)
    _, details = summarize_inputs(split)
    report = module.reproduce_control(config, split, details, control_path)
    assert len(fits) == report["fits"] == 3
    assert report["assessed_positives"] == 6 and report["assessed_negatives"] == 12
    assert report["unassessed_boundary_negatives"] == 48
    assert not report["thresholds_reselected"]
    assert all(r["saved_threshold"] == 0.8 for r in report["folds"])
