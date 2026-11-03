"""Sparse lexical retrieval combined with validated identifier lookup."""

from collections import defaultdict
from dataclasses import dataclass
from heapq import nsmallest

from sklearn.feature_extraction.text import TfidfVectorizer

from catalog_identity.normalization import NormalizedListing


def retrieval_text(listing: NormalizedListing) -> str:
    return " ".join((listing.title, listing.brand, listing.model)).strip()


def identifier_keys(listing: NormalizedListing) -> tuple[tuple[str, str], ...]:
    keys = []
    if listing.gtin:
        keys.append(("gtin", listing.gtin))
    if listing.brand and listing.model:
        keys.append(("brand_model", f"{len(listing.brand)}:{listing.brand}{listing.model}"))
    return tuple(keys)


@dataclass(frozen=True)
class Candidate:
    listing_id: str
    rank: int
    sources: tuple[str, ...]
    text_score: float
    semantic_score: float | None = None
    fusion_score: float | None = None


class CandidateIndex:
    """Fit vocabulary on training records, then index the chosen search population.

    Queries use sparse matrix products one at a time, never an all-pairs matrix.
    This is an exact lexical baseline, not a sublinear ANN index or a scale claim.
    Identifier hits precede text-only hits; ties break by stable listing ID.
    Identifier agreement supplies candidates, never an automatic match decision.
    """

    def __init__(self, max_features: int = 50_000):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=max_features,
            lowercase=False,
            sublinear_tf=True,
        )
        self._fitted = False
        self._indexed = False

    def fit(self, training_listings: list[NormalizedListing]) -> "CandidateIndex":
        if not training_listings:
            raise ValueError("training listings cannot be empty")
        self._fitted = False
        self._indexed = False
        texts = [retrieval_text(x) for x in training_listings]
        self.has_vocabulary = any(texts)
        if self.has_vocabulary:
            self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def build(self, listings: list[NormalizedListing]) -> "CandidateIndex":
        if not self._fitted:
            raise ValueError("fit the retriever on training records before building an index")
        self._indexed = False
        self.listings = sorted(listings, key=lambda x: x.key)
        if len({x.key for x in self.listings}) != len(self.listings):
            raise ValueError("duplicate listing in index")
        self.lookup: dict[tuple[str, str], set[int]] = defaultdict(set)
        for i, listing in enumerate(self.listings):
            for key in identifier_keys(listing):
                self.lookup[key].add(i)
        self.matrix = (
            self.vectorizer.transform([retrieval_text(x) for x in self.listings])
            if self.has_vocabulary and self.listings
            else None
        )
        self._indexed = True
        return self

    def query(self, listing: NormalizedListing, k: int = 20) -> list[Candidate]:
        if not self._indexed:
            raise ValueError("build an index before querying")
        if k < 1:
            raise ValueError("k must be positive")
        channels: dict[int, set[str]] = defaultdict(set)
        scores: dict[int, float] = {}
        if self.matrix is not None:
            vector = self.vectorizer.transform([retrieval_text(listing)])
            similarities = (vector @ self.matrix.T).tocsr()
            for i, value in zip(similarities.indices, similarities.data, strict=True):
                if value > 0:
                    channels[int(i)].add("tfidf")
                    scores[int(i)] = float(value)
        for key in identifier_keys(listing):
            for i in self.lookup.get(key, ()):
                channels[i].add(key[0])
        eligible = (i for i in channels if self.listings[i].key != listing.key)

        def rank_key(i: int):
            return (
                -int("gtin" in channels[i]),
                -int("brand_model" in channels[i]),
                -scores.get(i, 0.0),
                self.listings[i].key,
            )

        ranked = nsmallest(k, eligible, key=rank_key)
        return [
            Candidate(self.listings[i].key, rank, tuple(sorted(channels[i])), scores.get(i, 0.0))
            for rank, i in enumerate(ranked, 1)
        ]
