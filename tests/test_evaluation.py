import json

import numpy as np
import pytest

from catalog_identity.cli import main
from catalog_identity.evaluation.config import RunConfig
from catalog_identity.evaluation.metrics import classification_metrics, probability_metrics
from catalog_identity.evaluation.runner import retrieval_evaluation, run_evaluation
from catalog_identity.evaluation.synthetic import write_synthetic
from catalog_identity.ingestion.loaders import BenchmarkSplit, LabeledPair, load_wdc_pairs
from catalog_identity.matching.baseline import LogisticMatcher, feature_matrix
from catalog_identity.normalization import normalize
from catalog_identity.retrieval import CandidateIndex
from catalog_identity.schema import Listing


def test_classification_metrics_and_undefined_precision():
    report = classification_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert report["precision"] == report["recall"] == report["f1"] == 0.5
    assert classification_metrics(np.array([1, 0]), np.array([0, 0]))["precision"] is None


def test_probability_bins_include_probability_one():
    report = probability_metrics(np.array([0, 1]), np.array([0.0, 1.0]))
    assert report["brier_score"] == 0
    assert report["reliability_bins"][0]["count"] == report["reliability_bins"][9]["count"] == 1


@pytest.fixture
def synthetic_config(tmp_path):
    paths = write_synthetic(tmp_path / "fixture", seed=17)
    return RunConfig(
        dataset_name="test-fixture",
        dataset_version="1",
        synthetic=True,
        **paths,
        test_entity_disjoint=True,
        seed=17,
    )


def test_end_to_end_artifacts_and_reproducibility(tmp_path, synthetic_config):
    output = tmp_path / "run"
    report = run_evaluation(synthetic_config, output)
    repeated = run_evaluation(synthetic_config, tmp_path / "repeat")
    assert report["synthetic"] and "SYNTHETIC" in report["warning"]
    assert report["threshold_selection"] == repeated["threshold_selection"]
    assert (
        report["splits"]["test"]["pair_classification"]
        == repeated["splits"]["test"]["pair_classification"]
    )
    assert (output / "test-pairs.jsonl").read_bytes() == (
        tmp_path / "repeat/test-pairs.jsonl"
    ).read_bytes()
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["inputs"]["test"]["sha256"]) == 64
    assert manifest["source_sha256"] and manifest["dependencies"]["scikit-learn"]
    with pytest.raises(ValueError, match="already exists"):
        run_evaluation(synthetic_config, output)


def test_saved_model_parameters_reproduce_scores(tmp_path, synthetic_config):
    output = tmp_path / "run"
    run_evaluation(synthetic_config, output)
    model = json.loads((output / "model.json").read_text())
    split = load_wdc_pairs(synthetic_config.test, "test")
    x, _ = feature_matrix(split)
    logits = ((x - model["mean"]) / model["scale"]) @ np.array(model["coefficients"])[0] + model[
        "intercept"
    ][0]
    expected = 1 / (1 + np.exp(-logits))
    actual = [
        json.loads(line)["probability"]
        for line in (output / "test-pairs.jsonl").read_text().splitlines()
    ]
    np.testing.assert_allclose(actual, expected)


def test_training_only_and_protocol_validation(synthetic_config):
    with pytest.raises(ValueError, match="training split"):
        LogisticMatcher().fit(load_wdc_pairs(synthetic_config.test, "test"))
    values = synthetic_config.model_dump()
    with pytest.raises(ValueError, match="benchmark protocol"):
        RunConfig(**{**values, "synthetic": False})
    with pytest.raises(ValueError, match="100% unseen"):
        RunConfig(
            **{
                **values,
                "synthetic": False,
                "test_entity_disjoint": False,
                "protocol": {"corner_cases": 80, "development_size": "small", "test_unseen": 100},
            }
        )


def test_cli_demo_and_bad_config(tmp_path, capsys):
    assert main(["demo", "--output", str(tmp_path / "demo")]) == 0
    assert "SYNTHETIC" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "evaluate",
                "--config",
                str(tmp_path / "missing.json"),
                "--output",
                str(tmp_path / "bad"),
            ]
        )
    assert exc.value.code == 2


def test_retrieval_recall_counts_misses_and_uses_directed_denominator(tmp_path):
    a = Listing(source="test", source_id="a", title="matching lamp")
    b = Listing(source="test", source_id="b", title="matching lamp")
    c = Listing(source="test", source_id="c", title="zzzzzz")
    split = BenchmarkSplit(
        "test",
        {x.key: x for x in (a, b, c)},
        (LabeledPair(a.key, b.key, 1, "ab"), LabeledPair(a.key, c.key, 1, "ac")),
        {x.key: "one-entity" for x in (a, b, c)},
    )
    index = CandidateIndex().fit([normalize(a)])
    metrics, _ = retrieval_evaluation(index, split, [1, 10], tmp_path / "candidates.jsonl")
    assert metrics["labeled_positive_directions"] == 4
    assert metrics["labeled_pair_recall_at_k"] == {"1": 0.5, "10": 0.5}
    assert metrics["entity_positive_directions"] == 6
    assert metrics["entity_recall_at_k"]["10"] == pytest.approx(1 / 3)


def test_cli_evaluation_reads_config_relative_paths(tmp_path, synthetic_config, capsys):
    values = synthetic_config.model_dump(mode="json")
    for name in ("train", "validation", "test"):
        values[name] = str(getattr(synthetic_config, name).relative_to(tmp_path))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))
    assert main(["evaluate", "--config", str(path), "--output", str(tmp_path / "evaluated")]) == 0
    assert "Report:" in capsys.readouterr().out
