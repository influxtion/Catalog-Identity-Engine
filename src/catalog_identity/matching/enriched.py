"""Train-fitted text features; extracted title hints are evidence, never gold IDs."""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from catalog_identity.ingestion.loaders import BenchmarkSplit
from catalog_identity.matching.features import FEATURE_NAMES, jaccard, pair_features, tokens
from catalog_identity.normalization import normalize
from catalog_identity.schema import Listing

ENRICHED_VERSION = "2"
EXTRA_NAMES = (
    "title_char_cosine",
    "description_char_cosine",
    "title_token_containment",
    "title_char_jaccard",
    "model_token_jaccard",
    "model_token_conflict",
    "model_token_missing",
    "capacity_jaccard",
    "capacity_conflict",
    "capacity_missing",
    "color_jaccard",
    "color_conflict",
    "color_missing",
    "pack_jaccard",
    "pack_conflict",
    "pack_missing",
    "title_unshared_fraction",
    "title_numeric_conflict",
    "title_exact",
)
ENRICHED_NAMES = FEATURE_NAMES + EXTRA_NAMES
COLORS = frozenset(
    "black white red green blue gray grey silver gold pink purple orange yellow brown".split()
)


def title_hints(title: str) -> dict[str, set[str]]:
    model = {
        re.sub(r"[-\s]", "", t)
        for t in tokens(title)
        if re.search(r"[a-z]", t)
        and re.search(r"\d", t)
        and not re.fullmatch(r"\d+(?:\.\d+)?(?:gb|tb|mb|mm|cm|kg|g|mhz|ghz)", t)
    }
    capacity = {
        f"{float(n) * (1000 if unit == 'tb' else 1):g}gb"
        for n, unit in re.findall(r"\b(\d+(?:\.\d+)?)\s*(gb|tb)\b", title)
    }
    colors = {"gray" if c == "grey" else c for c in tokens(title) & COLORS}
    packs = set(re.findall(r"\b(\d+)\s*[- ]?\s*(?:pack|pk|rolls|sheets|photos)\b", title))
    return {"model": model, "capacity": capacity, "color": colors, "pack": packs}


def set_evidence(a: set[str], b: set[str]) -> tuple[float, float, float]:
    return jaccard(a, b), float(bool(a and b and a.isdisjoint(b))), float(not a or not b)


class TextFeatures:
    def __init__(self):
        self.vectorizers = {}

    def fit(self, train: BenchmarkSplit):
        if train.name != "train":
            raise ValueError("text vocabulary requires the training split")
        normalized = [normalize(x) for x in train.listings.values()]
        for field in ("title", "description"):
            texts = [getattr(x, field) for x in normalized]
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=60_000,
                sublinear_tf=True,
                lowercase=False,
            )
            self.vectorizers[field] = vectorizer.fit(texts) if any(texts) else None
        return self

    def transform(self, listings: dict[str, Listing], pairs: list[tuple[str, str]]) -> np.ndarray:
        if not self.vectorizers:
            raise ValueError("fit text features first")
        records = {key: normalize(value) for key, value in listings.items()}
        keys = list(records)
        positions = {key: i for i, key in enumerate(keys)}
        left = [positions[a] for a, _ in pairs]
        right = [positions[b] for _, b in pairs]
        cosines = {}
        for field, vectorizer in self.vectorizers.items():
            if vectorizer is None or not keys or not pairs:
                cosines[field] = np.zeros(len(pairs))
            else:
                matrix = vectorizer.transform([getattr(records[k], field) for k in keys])
                cosines[field] = np.asarray(
                    matrix[left].multiply(matrix[right]).sum(axis=1)
                ).ravel()
        values = []
        for i, (a_key, b_key) in enumerate(pairs):
            a, b = records[a_key], records[b_key]
            at, bt = tokens(a.title), tokens(b.title)
            ah, bh = title_hints(a.title), title_hints(b.title)
            ac = {a.title[j : j + 3] for j in range(max(0, len(a.title) - 2))}
            bc = {b.title[j : j + 3] for j in range(max(0, len(b.title) - 2))}
            an, bn = set(re.findall(r"\d+", a.title)), set(re.findall(r"\d+", b.title))
            extra = [
                cosines["title"][i],
                cosines["description"][i],
                len(at & bt) / max(1, min(len(at), len(bt))),
                jaccard(ac, bc),
                *(
                    v
                    for hint in ("model", "capacity", "color", "pack")
                    for v in set_evidence(ah[hint], bh[hint])
                ),
                len(at ^ bt) / max(1, len(at | bt)),
                float(bool(an and bn and an != bn)),
                float(bool(a.title and a.title == b.title)),
            ]
            values.append([*pair_features(a, b).values(), *extra])
        return np.asarray(values, dtype=float).reshape(-1, len(ENRICHED_NAMES))

    def split(self, split: BenchmarkSplit) -> np.ndarray:
        return self.transform(split.listings, [(p.left, p.right) for p in split.pairs])

    def save(self, path: Path):
        data = {
            field: {
                "vocabulary": {k: int(i) for k, i in v.vocabulary_.items()},
                "idf": v.idf_.tolist(),
            }
            if v is not None
            else None
            for field, v in self.vectorizers.items()
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        result = cls()
        for field, data in json.loads(path.read_text(encoding="utf-8")).items():
            vectorizer = None
            if data:
                vectorizer = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    lowercase=False,
                    vocabulary=data["vocabulary"],
                )
                vectorizer.idf_ = np.asarray(data["idf"])
            result.vectorizers[field] = vectorizer
        return result
