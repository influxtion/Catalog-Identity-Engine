import pytest

from catalog_identity.evaluation.cluster_showcase import cluster_showcase
from catalog_identity.evaluation.config import RunConfig
from catalog_identity.evaluation.development import develop
from catalog_identity.evaluation.synthetic import write_synthetic


def test_cluster_showcase_marks_synthetic_and_demonstrates_bridge(tmp_path):
    report = cluster_showcase(tmp_path / "clusters")
    assert report["synthetic"]
    assert report["bridge_ablation"]["connected_components"]["metrics"]["erroneous_merges"] == 1
    assert report["bridge_ablation"]["constrained"]["metrics"]["erroneous_merges"] == 0
    assert report["levels"]["variant"]["excluded_unresolved"] == 2


def test_negative_ablation_uses_only_training_and_retains_positives(tmp_path):
    paths = write_synthetic(tmp_path / "fixture")
    config = RunConfig(dataset_name="fixture", dataset_version="1", synthetic=True, **paths)
    report = develop(
        config, tmp_path / "model", learners=("logistic",), negative_policy="random_only"
    )
    selection = report["training_selection"]
    assert selection["used_pairs"] < selection["original_pairs"]
    assert selection["positives"] == 72


def test_scale_report_measures_cost_without_accuracy_claims(tmp_path):
    pytest.importorskip("psutil")
    from catalog_identity.evaluation.scaling import scaling

    report = scaling(tmp_path / "scale", sizes=(30,), queries=4)
    assert report["synthetic"]
    assert report["runs"][0]["queries"] == 4
    assert report["runs"][0]["query_p95_ms"] > 0
    assert report["runs"][0]["sampled_peak_rss_bytes"] > 0


def test_incremental_memberships_match_fresh_rebuild_and_retain_unchanged_ids(tmp_path):
    pytest.importorskip("sqlalchemy")
    from catalog_identity.evaluation.incremental import incremental_showcase

    report = incremental_showcase(tmp_path / "updates")
    assert report["synthetic"] and report["identical_reingestion_changed"] == 0
    for step in report["steps"]:
        assert step["changed_listings"] == 1
        assert step["stale_before_rebuild"] and not step["stale_after_rebuild"]
        for result in step["levels"].values():
            assert result["matches_fresh_partition"]
            assert result["unchanged_groups"] == result["unchanged_ids_retained"]
    assert report["steps"][0]["levels"]["variant"]["groups"] == 8
    assert report["steps"][1]["levels"]["variant"]["groups"] == 9
