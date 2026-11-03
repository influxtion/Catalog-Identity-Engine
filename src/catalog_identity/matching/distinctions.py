"""Version 3 text distinctions. Hints remain learned evidence, never identity labels."""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from catalog_identity.matching.enriched import ENRICHED_NAMES, TextFeatures, set_evidence
from catalog_identity.matching.features import tokens
from catalog_identity.normalization import normalize

HINTS = ("code", "quantity", "form_factor", "edition", "audio_channels", "memory_standard")
DISTINCTION_NAMES = ENRICHED_NAMES + (
    "title_word_cosine",
    "unshared_word_weight_min",
    "unshared_word_weight_max",
    *(
        f"{hint}_v3_{stat}"
        for hint in HINTS
        for stat in ("jaccard", "conflict", "missing", "one_sided")
    ),
    "numeric_sequence_jaccard",
    "numeric_sequence_conflict",
)


def distinction_hints(title):
    words = tokens(title)
    codes = {
        re.sub(r"[-/]", "", word)
        for word in words
        if len(word) >= 5
        and re.search(r"[a-z]", word)
        and len(re.findall(r"\d", word)) >= 3
        and not re.fullmatch(r"[\d.x]+(?:gb|tb|mb|mhz|ghz|mm|cm|kg|ml|w)", word)
        and not re.match(r"(?:g?ddr|pc\d)", word)
    }
    quantities = set(
        re.findall(r"\b(\d+)\s*[- ]?\s*(?:shot|photo|sheet|roll|pack|pk|piece)s?\b", title)
    )
    forms = set()
    for name, pattern in (
        ("sodimm", r"\bso[- ]?dimm\b"),
        ("udimm", r"\budimm\b"),
        ("rdimm", r"\brdimm\b"),
        ("m2", r"\bm[. ]?2\b"),
        ("2.5inch", r"\b2\.5\s*(?:inch|in|\")"),
    ):
        if re.search(pattern, title):
            forms.add(name)
    if "sodimm" not in forms and re.search(r"\bdimm\b", title):
        forms.add("dimm")
    editions = words & {
        "pro",
        "plus",
        "ultra",
        "max",
        "lite",
        "mini",
        "super",
        "touch",
        "square",
        "wide",
    }
    if re.search(r"\bspecial\s+edition\b", title):
        editions.add("special_edition")
    channels = set()
    if words & {"mono", "monaural"}:
        channels.add("mono")
    if words & {"stereo", "binaural"}:
        channels.add("stereo")
    return {
        "code": codes,
        "quantity": quantities,
        "form_factor": forms,
        "edition": editions,
        "audio_channels": channels,
        "memory_standard": set(re.findall(r"\bg?ddr\d+[a-z]?\b", title)),
    }


class DistinctionFeatures(TextFeatures):
    names = DISTINCTION_NAMES
    version = "3"

    def fit(self, train):
        super().fit(train)
        titles = [normalize(record).title for record in train.listings.values()]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w[\w.+/-]*\b",
            max_features=60_000,
            sublinear_tf=True,
            lowercase=False,
        )
        self.word_vectorizer = vectorizer.fit(titles) if any(titles) else None
        return self

    def transform(self, listings, pairs):
        base = super().transform(listings, pairs)
        records = {key: normalize(value) for key, value in listings.items()}
        keys = list(records)
        positions = {key: i for i, key in enumerate(keys)}
        word_scores = np.zeros(len(pairs))
        weights = np.zeros((len(pairs), 2))
        if self.word_vectorizer is not None and pairs:
            matrix = self.word_vectorizer.transform([records[key].title for key in keys])
            left, right = ([positions[p[i]] for p in pairs] for i in (0, 1))
            a, b = matrix[left], matrix[right]
            word_scores = np.asarray(a.multiply(b).sum(axis=1)).ravel()
            unshared_a = np.asarray((a - a.multiply(b.sign())).sum(axis=1)).ravel()
            unshared_b = np.asarray((b - b.multiply(a.sign())).sum(axis=1)).ravel()
            weights = np.sort(np.column_stack((unshared_a, unshared_b)), axis=1)
        extra = []
        for i, (left, right) in enumerate(pairs):
            a, b = records[left].title, records[right].title
            ah, bh = distinction_hints(a), distinction_hints(b)
            an, bn = (set(re.findall(r"\d+(?:[./-]\d+)*", title)) for title in (a, b))
            extra.append(
                [
                    word_scores[i],
                    *weights[i],
                    *(
                        value
                        for hint in HINTS
                        for value in (
                            *set_evidence(ah[hint], bh[hint]),
                            float(bool(ah[hint]) != bool(bh[hint])),
                        )
                    ),
                    *set_evidence(an, bn)[:2],
                ]
            )
        return (
            np.column_stack((base, np.asarray(extra).reshape(len(pairs), -1)))
            if pairs
            else np.empty((0, len(self.names)))
        )

    def save(self, path: Path):
        super().save(path)
        vectorizer = self.word_vectorizer
        data = json.loads(path.read_text(encoding="utf-8"))
        data["word_v3"] = (
            None
            if vectorizer is None
            else {
                "vocabulary": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
                "idf": vectorizer.idf_.tolist(),
            }
        )
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        result = super().load(path)
        data = json.loads(path.read_text(encoding="utf-8"))["word_v3"]
        result.word_vectorizer = None
        if data is not None:
            result.word_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                token_pattern=r"(?u)\b\w[\w.+/-]*\b",
                sublinear_tf=True,
                lowercase=False,
                vocabulary=data["vocabulary"],
            )
            result.word_vectorizer.idf_ = np.asarray(data["idf"])
        return result
