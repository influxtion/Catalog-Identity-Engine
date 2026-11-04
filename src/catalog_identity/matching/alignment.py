"""Version 4 bounded text alignment. All extracted specifications are soft evidence.

No identifiers, benchmark labels or hard-negative flags are inspected. Cached parses
are keyed by normalized text, not listing identity. Description lists can describe
compatibility rather than identity; no feature here is a merge veto.
"""

import re
from functools import lru_cache

import numpy as np

from catalog_identity.matching.distinctions import DISTINCTION_NAMES, DistinctionFeatures
from catalog_identity.matching.enriched import set_evidence
from catalog_identity.matching.features import jaccard, tokens
from catalog_identity.normalization import normalize

TYPES = ("capacity", "module_capacity", "module_count", "frequency", "dimension", "pack")
STATS = ("jaccard", "conflict", "missing", "one_sided")
ALIGNMENT_NAMES = DISTINCTION_NAMES + (
    *(f"aligned_code_{stat}" for stat in STATS),
    "code_alpha_jaccard",
    "code_numeric_jaccard",
    "code_component_support",
    "code_unmatched_suffix",
    "code_partial_prefix",
    *(f"typed_{kind}_{stat}" for kind in TYPES for stat in STATS),
    "idf_containment_min",
    "idf_containment_max",
    *(
        f"cross_{kind}_{direction}"
        for kind in ("char", "token", "code", "spec")
        for direction in ("min", "max")
    ),
    "cross_description_missing",
    "cross_description_one_sided",
)
NUMBER = r"\d+(?:\.\d+)?"


def evidence(a, b):
    return (*set_evidence(a, b), float(bool(a) != bool(b)))


@lru_cache(maxsize=16384)
def parse_text(text):
    """Bound parsing and pair comparisons while preserving meaningful code suffixes."""
    text = text[:8192].lower()
    # Join only adjacent alpha/numeric fragments, never arbitrary word pairs.
    joined = re.sub(r"\b([a-z]{1,12})[ -]+(?=\d)", r"\1", text)
    joined = re.sub(r"(?<=\d)-(?=[a-z0-9])", "", joined)
    codes = set()
    for word in re.findall(r"[a-z0-9]+(?:[./+][a-z0-9]+)*\+?", joined):
        if re.search(r"[a-z]", word) and re.search(r"\d", word):
            if not re.fullmatch(r"[\d.x]+(?:gb|tb|mb|khz|mhz|ghz|mm|cm|kg|ml|w)", word):
                codes.add(word)
        if len(codes) >= 24:
            break
    specs = {kind: set() for kind in TYPES}
    scale = {"mb": 0.001, "gb": 1, "tb": 1000}
    modules = list(re.finditer(rf"\b(\d+)\s*[x×]\s*({NUMBER})\s*(mb|gb|tb)\b", text))
    for match in modules:
        count, size, unit = match.groups()
        capacity = float(size) * scale[unit]
        specs["module_capacity"].add(f"{capacity:g}")
        specs["module_count"].add(str(int(count)))
        specs["capacity"].add(f"{int(count) * capacity:g}")
    for match in re.finditer(rf"(?<![\w.])({NUMBER})\s*(mb|gb|tb)\b", text):
        if not any(m.start() <= match.start() < m.end() for m in modules):
            specs["capacity"].add(f"{float(match[1]) * scale[match[2]]:g}")
    for value, unit in re.findall(rf"\b({NUMBER})\s*(khz|mhz|ghz)\b", text):
        specs["frequency"].add(f"{float(value) * {'khz': 0.001, 'mhz': 1, 'ghz': 1000}[unit]:g}")
    # Dimension tuples retain axis order and unit conversion; scalar sizes are separate.
    for match in re.finditer(
        rf"\b({NUMBER}(?:\s*[x×]\s*{NUMBER}){{0,2}})\s*(mm|cm|inch|inches)\b", text
    ):
        factor = {"mm": 1, "cm": 10, "inch": 25.4, "inches": 25.4}[match[2]]
        specs["dimension"].add(
            "x".join(f"{float(v) * factor:g}" for v in re.findall(NUMBER, match[1]))
        )
    specs["pack"].update(re.findall(r"\b(\d+)\s*[- ]?(?:pack|pk|piece|pcs)s?\b", text))
    specs["pack"].update(re.findall(r"\bpack\s+of\s+(\d+)\b", text))
    if re.search(r"\bsingle\b", text):
        specs["pack"].add("1")
    if re.search(r"\b(?:twin|double)\s*(?:pack)?\b", text):
        specs["pack"].add("2")
    return frozenset(codes), {k: frozenset(v) for k, v in specs.items()}, frozenset(tokens(text))


def code_alignment(a, b):
    def alpha(codes):
        return {v for c in codes for v in re.findall(r"[a-z]+", c)}

    def numeric(codes):
        return {v for c in codes for v in re.findall(NUMBER, c)}

    support = partial = suffix = 0.0
    for left in a:
        for right in b:
            lc, rc = re.findall(r"[a-z]+|\d+|[./+]", left), re.findall(r"[a-z]+|\d+|[./+]", right)
            common = 0
            for x, y in zip(lc, rc, strict=False):
                if x != y:
                    break
                common += 1
            support = max(support, common / max(len(lc), len(rc), 1))
            if common and left != right:
                suffix = 1.0
            if common == min(len(lc), len(rc)) and len(lc) != len(rc):
                partial = 1.0
    return (
        *evidence(a, b),
        jaccard(alpha(a), alpha(b)),
        jaccard(numeric(a), numeric(b)),
        support,
        suffix,
        partial,
    )


class AlignmentFeatures(DistinctionFeatures):
    names = ALIGNMENT_NAMES
    version = "4"

    def transform(self, listings, pairs):
        base = super().transform(listings, pairs)
        if not pairs:
            return np.empty((0, len(self.names)))
        records = {key: normalize(value) for key, value in listings.items()}
        parsed = {
            key: (parse_text(r.title), parse_text(r.description)) for key, r in records.items()
        }
        keys = list(records)
        positions = {key: i for i, key in enumerate(keys)}
        left, right = ([positions[p[i]] for p in pairs] for i in (0, 1))
        cross = np.zeros((len(pairs), 2))
        vectorizer = self.vectorizers["title"]
        if vectorizer is not None:
            titles = vectorizer.transform([records[k].title for k in keys])
            descriptions = vectorizer.transform([records[k].description[:8192] for k in keys])
            cross = np.column_stack(
                (
                    np.asarray(titles[left].multiply(descriptions[right]).sum(axis=1)).ravel(),
                    np.asarray(titles[right].multiply(descriptions[left]).sum(axis=1)).ravel(),
                )
            )
        vocabulary = self.word_vectorizer.vocabulary_ if self.word_vectorizer is not None else {}
        idf = self.word_vectorizer.idf_ if self.word_vectorizer is not None else []

        def weight(words):
            return sum(float(idf[vocabulary[w]]) for w in words if w in vocabulary)

        extra = []
        for i, (a, b) in enumerate(pairs):
            (ac, asp, at), (adc, adsp, adt) = parsed[a]
            (bc, bsp, bt), (bdc, bdsp, bdt) = parsed[b]
            shared = weight(at & bt)
            containment = sorted((shared / max(weight(at), 1e-12), shared / max(weight(bt), 1e-12)))
            cross_tokens = sorted(
                (len(at & bdt) / max(1, len(at)), len(bt & adt) / max(1, len(bt)))
            )
            cross_codes = sorted((len(ac & bdc) / max(1, len(ac)), len(bc & adc) / max(1, len(bc))))
            cross_specs = sorted(
                (
                    sum(len(asp[k] & bdsp[k]) for k in TYPES)
                    / max(1, sum(len(asp[k]) for k in TYPES)),
                    sum(len(bsp[k] & adsp[k]) for k in TYPES)
                    / max(1, sum(len(bsp[k]) for k in TYPES)),
                )
            )
            extra.append(
                [
                    *code_alignment(ac, bc),
                    *(v for k in TYPES for v in evidence(asp[k], bsp[k])),
                    *containment,
                    *sorted(cross[i]),
                    *cross_tokens,
                    *cross_codes,
                    *cross_specs,
                    float(not adt or not bdt),
                    float(bool(adt) != bool(bdt)),
                ]
            )
        return np.column_stack((base, np.asarray(extra)))
