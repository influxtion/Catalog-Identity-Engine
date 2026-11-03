import pytest

from catalog_identity.evaluation.config import RunConfig
from catalog_identity.evaluation.development import develop
from catalog_identity.evaluation.frozen import evaluate_frozen
from catalog_identity.evaluation.synthetic import write_synthetic


def test_frozen_evaluation_uses_saved_model_and_checks_inputs(tmp_path):
    paths = write_synthetic(tmp_path / "fixture")
    config = RunConfig(
        dataset_name="fixture",
        dataset_version="1",
        synthetic=True,
        **paths,
        test_entity_disjoint=True,
    )
    develop(config, tmp_path / "development", learners=("logistic",))
    model = tmp_path / "development/logistic"
    before = (model / "matcher.json").read_bytes()
    result = evaluate_frozen(config, model, tmp_path / "test-run")
    assert result["phase"] == "frozen_test_evaluation"
    assert (model / "matcher.json").read_bytes() == before
    paths["validation"].write_text(paths["validation"].read_text() + "\n")
    with pytest.raises(ValueError, match="development inputs"):
        evaluate_frozen(config, model, tmp_path / "bad")
