import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    if labels.shape != predictions.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("nonempty, aligned one-dimensional labels and predictions required")
    if not np.isin(labels, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("labels and predictions must be binary")
    positives = labels == 1
    predicted = predictions.astype(bool)
    tp = int(np.sum(positives & predicted))
    fp = int(np.sum(~positives & predicted))
    fn = int(np.sum(positives & ~predicted))
    tn = int(np.sum(~positives & ~predicted))
    return {
        "pairs": len(labels),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    if labels.shape != probabilities.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("nonempty, aligned one-dimensional labels and probabilities required")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("invalid probabilities")
    bins = []
    assignments = np.minimum((probabilities * 10).astype(int), 9)
    for i in range(10):
        mask = assignments == i
        bins.append(
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "count": int(mask.sum()),
                "mean_probability": float(probabilities[mask].mean()) if mask.any() else None,
                "positive_fraction": float(labels[mask].mean()) if mask.any() else None,
            }
        )
    curve = None
    if np.any(labels == 1):
        precision, recall, thresholds = precision_recall_curve(labels, probabilities)
        curve = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        }
    return {
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities))
        if np.any(labels == 1)
        else None,
        "reliability_bins": bins,
        "precision_recall_curve": curve,
        "probability_calibration": "none; reliability is measured, not assumed",
    }
