import numpy as np
import pytest

from catalog_identity.evaluation.config import RunConfig
from catalog_identity.evaluation.development import develop
from catalog_identity.evaluation.synthetic import write_synthetic
from catalog_identity.matching.baseline import select_match_threshold


def test_validation_development_never_needs_test_file(tmp_path):
    paths = write_synthetic(tmp_path / "fixture")
    config = RunConfig(
        dataset_name="fixture",
        dataset_version="1",
        synthetic=True,
        train=paths["train"],
        validation=paths["validation"],
        test=tmp_path / "absent",
    )
    result = develop(config, tmp_path / "run", learners=("logistic",))
    assert not result["test_opened"]
    assert (tmp_path / "run/logistic/policy.json").exists()


def test_weighted_policy_accounts_for_excluded_negatives():
    labels = np.array([1, 0, 1])
    probabilities = np.array([0.9, 0.8, 0.7])
    unweighted = select_match_threshold(labels, probabilities, 0.65, 2)
    weighted = select_match_threshold(labels, probabilities, 0.65, 2, np.array([1.0, 2.0, 1.0]))
    assert unweighted.threshold == 0.7 and weighted.threshold is None
    with pytest.raises(ValueError, match="sample weights"):
        select_match_threshold(labels, probabilities, sample_weights=np.array([-1.0, 1.0, 1.0]))
