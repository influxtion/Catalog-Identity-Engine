"""Portable logistic/LightGBM scorers with validation-only sigmoid calibration."""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from catalog_identity.ingestion.loaders import BenchmarkSplit
from catalog_identity.matching.enriched import ENRICHED_NAMES, ENRICHED_VERSION, TextFeatures


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -700, 700)))


class LearnedMatcher:
    def __init__(self, learner="lightgbm", seed=42, num_leaves=15, n_estimators=200):
        if learner not in ("logistic", "lightgbm"):
            raise ValueError("unknown learner")
        self.settings = dict(
            learner=learner, seed=seed, num_leaves=num_leaves, n_estimators=n_estimators
        )
        self.features = TextFeatures()
        self.calibration = None
        self.parameters = None

    def fit(self, train: BenchmarkSplit):
        x = self.features.fit(train).split(train)
        y = np.array([p.label for p in train.pairs])
        if set(y.tolist()) != {0, 1}:
            raise ValueError("training requires both classes")
        if self.settings["learner"] == "lightgbm":
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                n_estimators=self.settings["n_estimators"],
                num_leaves=self.settings["num_leaves"],
                learning_rate=0.05,
                min_child_samples=20,
                reg_lambda=1.0,
                random_state=self.settings["seed"],
                n_jobs=1,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
            )
            model.fit(x, y, feature_name=list(ENRICHED_NAMES))
            self.booster = model.booster_
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            model.fit(x, y)
            scaler, logistic = model.steps[0][1], model.steps[1][1]
            self.parameters = {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "coefficients": logistic.coef_[0].tolist(),
                "intercept": float(logistic.intercept_[0]),
            }
        return self

    def score_features(self, x, calibrated=True):
        if not len(x):
            return np.empty(0)
        if self.settings["learner"] == "lightgbm":
            raw = self.booster.predict(x, num_threads=1)
        else:
            p = self.parameters
            raw = sigmoid(((x - p["mean"]) / p["scale"]) @ p["coefficients"] + p["intercept"])
        if calibrated and self.calibration is not None:
            return sigmoid(logit(raw) * self.calibration["slope"] + self.calibration["intercept"])
        return raw

    def predict(self, split: BenchmarkSplit, calibrated=True):
        return self.score_features(self.features.split(split), calibrated)

    def calibrate(self, validation: BenchmarkSplit, sample_weights=None):
        if validation.name != "validation":
            raise ValueError("calibration requires validation records")
        y = np.array([p.label for p in validation.pairs])
        if set(y.tolist()) != {0, 1}:
            raise ValueError("calibration requires both classes")
        x = logit(self.predict(validation, calibrated=False)).reshape(-1, 1)
        model = LogisticRegression(C=1.0).fit(x, y, sample_weight=sample_weights)
        if model.coef_[0, 0] <= 0:
            raise ValueError("calibration would reverse score ordering; inspect validation data")
        self.calibration = {
            "slope": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]),
        }
        return self

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=False)
        self.features.save(directory / "text-features.json")
        (directory / "matcher.json").write_text(
            json.dumps(
                {
                    "settings": self.settings,
                    "calibration": self.calibration,
                    "parameters": self.parameters,
                    "feature_version": ENRICHED_VERSION,
                    "feature_names": list(ENRICHED_NAMES),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if self.settings["learner"] == "lightgbm":
            self.booster.save_model(str(directory / "trees.txt"))

    @classmethod
    def load(cls, directory: Path):
        data = json.loads((directory / "matcher.json").read_text(encoding="utf-8"))
        if data["feature_version"] != ENRICHED_VERSION or data["feature_names"] != list(
            ENRICHED_NAMES
        ):
            raise ValueError("model feature implementation does not match saved model")
        result = cls(**data["settings"])
        result.features = TextFeatures.load(directory / "text-features.json")
        result.calibration, result.parameters = data["calibration"], data["parameters"]
        if result.settings["learner"] == "lightgbm":
            import lightgbm as lgb

            result.booster = lgb.Booster(model_file=str(directory / "trees.txt"))
        return result
