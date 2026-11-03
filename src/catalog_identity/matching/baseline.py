from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from catalog_identity.ingestion.loaders import BenchmarkSplit
from catalog_identity.matching.features import FEATURE_NAMES, pair_features
from catalog_identity.normalization import normalize


def feature_matrix(split: BenchmarkSplit) -> tuple[np.ndarray, np.ndarray]:
    normalized = {key: normalize(listing) for key, listing in split.listings.items()}
    features = [
        list(pair_features(normalized[p.left], normalized[p.right]).values()) for p in split.pairs
    ]
    return (
        np.asarray(features, dtype=float).reshape(-1, len(FEATURE_NAMES)),
        np.asarray([p.label for p in split.pairs], dtype=int),
    )


class LogisticMatcher:
    def __init__(self, seed: int = 42, regularization: float = 1.0):
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=regularization, max_iter=2000, random_state=seed),
        )

    def fit(self, split: BenchmarkSplit) -> "LogisticMatcher":
        if split.name != "train":
            raise ValueError("matcher fitting requires the training split")
        features, labels = feature_matrix(split)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("training requires positive and negative examples")
        self.pipeline.fit(features, labels)
        return self

    def predict(self, split: BenchmarkSplit) -> np.ndarray:
        features, _ = feature_matrix(split)
        if not len(features):
            raise ValueError("cannot score an empty pair set")
        return self.pipeline.predict_proba(features)[:, 1]


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float | None
    validation_precision: float | None
    validation_recall: float
    predicted_matches: int
    target_precision: float
    minimum_matches: int


def select_match_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_precision: float = 0.99,
    minimum_matches: int = 5,
) -> ThresholdSelection:
    """Maximize validation recall subject to empirical precision and support.

    A missing threshold means abstain from automatic matches. This is an empirical
    operating point, not a statistical guarantee or probability calibration.
    """
    if not 0 < target_precision <= 1 or minimum_matches < 1:
        raise ValueError("invalid threshold selection constraints")
    if len(labels) != len(probabilities) or not len(labels):
        raise ValueError("nonempty, aligned validation arrays are required")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("invalid probabilities")
    positives = int(labels.sum())
    best = ThresholdSelection(None, None, 0.0, 0, target_precision, minimum_matches)
    # Sweep distinct scores once; tied probabilities must enter together.
    order = np.argsort(-probabilities, kind="stable")
    true_positives = 0
    for count, i in enumerate(order, 1):
        true_positives += int(labels[i])
        if count < len(order) and probabilities[order[count]] == probabilities[i]:
            continue
        precision = true_positives / count
        recall = true_positives / positives if positives else 0.0
        if (
            count >= minimum_matches
            and precision >= target_precision
            and recall > best.validation_recall
        ):
            best = ThresholdSelection(
                float(probabilities[i]), precision, recall, count, target_precision, minimum_matches
            )
    return best


def decision(probability: float, match_threshold: float | None, reject_threshold: float) -> str:
    if not 0 <= reject_threshold <= 1 or not 0 <= probability <= 1:
        raise ValueError("probabilities and reject threshold must be in [0, 1]")
    if match_threshold is not None:
        if not 0 <= match_threshold <= 1 or reject_threshold >= match_threshold:
            raise ValueError("reject threshold must be below match threshold")
        if probability >= match_threshold:
            return "match"
    if probability <= reject_threshold:
        return "reject"
    return "review"
