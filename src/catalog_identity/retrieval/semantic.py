"""Optional, revision-pinned sentence embeddings and a FAISS HNSW index."""

import re
from dataclasses import replace

import numpy as np

from catalog_identity.retrieval import Candidate, CandidateIndex, retrieval_text

MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class SentenceEncoder:
    def __init__(
        self,
        model="sentence-transformers/all-MiniLM-L6-v2",
        revision=MINILM_REVISION,
        allow_download=False,
    ):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("embedding revision must be an immutable commit SHA")
        from sentence_transformers import SentenceTransformer

        self.model_name, self.revision = model, revision
        self.model = SentenceTransformer(
            model,
            revision=revision,
            device="cpu",
            token=False,
            trust_remote_code=False,
            local_files_only=not allow_download,
        )

    def encode(self, texts):
        return self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )


class SemanticIndex:
    def __init__(self, encoder, connections=32, ef_search=128):
        import faiss

        self.faiss, self.encoder = faiss, encoder
        self.connections, self.ef_search = connections, ef_search
        self.matrix = None
        self.index = None
        self.cache = {}

    def fit(self, training_listings):
        # Pretrained weights are fixed; no validation vocabulary is fitted.
        return self

    def _encode(self, texts):
        vectors = np.asarray(self.encoder.encode(texts), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
            raise ValueError("encoder returned invalid vectors")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("encoder returned zero vectors")
        return np.ascontiguousarray(vectors / norms)

    def build(self, listings):
        self.listings = sorted(listings, key=lambda x: x.key)
        if len({x.key for x in self.listings}) != len(self.listings):
            raise ValueError("duplicate listing in semantic index")
        self.cache = {}
        self.index = None
        if not self.listings:
            return self
        texts = [retrieval_text(x) for x in self.listings]
        vectors = self._encode(texts)
        self.faiss.omp_set_num_threads(1)
        self.index = self.faiss.IndexHNSWFlat(
            vectors.shape[1], self.connections, self.faiss.METRIC_INNER_PRODUCT
        )
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = self.ef_search
        self.index.add(vectors)
        self.cache = {text: vector for text, vector in zip(texts, vectors, strict=True)}
        return self

    def query(self, listing, k=20):
        if k < 1:
            raise ValueError("k must be positive")
        text = retrieval_text(listing)
        if self.index is None or not text:
            return []
        vector = self.cache[text] if text in self.cache else self._encode([text])[0]
        scores, positions = self.index.search(vector.reshape(1, -1), min(k + 1, len(self.listings)))
        hits = [
            (self.listings[int(i)].key, float(s))
            for i, s in zip(positions[0], scores[0], strict=True)
            if i >= 0 and self.listings[int(i)].key != listing.key
        ]
        hits.sort(key=lambda item: (-item[1], item[0]))
        return [
            Candidate(key, rank, ("semantic",), 0.0, score)
            for rank, (key, score) in enumerate(hits[:k], 1)
        ]

    @property
    def index_bytes(self):
        return len(self.faiss.serialize_index(self.index)) if self.index is not None else 0


class HybridIndex:
    def __init__(self, encoder, max_features=50_000, depth=50, rrf_constant=60):
        if depth < 1 or rrf_constant < 1:
            raise ValueError("fusion depth and constant must be positive")
        self.lexical = CandidateIndex(max_features)
        self.semantic = SemanticIndex(encoder)
        self.depth, self.rrf_constant = depth, rrf_constant

    @property
    def matrix(self):
        return self.lexical.matrix

    def fit(self, training_listings):
        self.lexical.fit(training_listings)
        return self

    def build(self, listings):
        self.lexical.build(listings)
        self.semantic.build(listings)
        return self

    def query(self, listing, k=20):
        union = {}
        for results in (
            self.lexical.query(listing, max(k, self.depth)),
            self.semantic.query(listing, max(k, self.depth)),
        ):
            for candidate in results:
                previous = union.get(candidate.listing_id)
                score = 1 / (self.rrf_constant + candidate.rank)
                if previous:
                    union[candidate.listing_id] = replace(
                        previous,
                        sources=tuple(sorted(set(previous.sources + candidate.sources))),
                        semantic_score=candidate.semantic_score,
                        fusion_score=previous.fusion_score + score,
                    )
                else:
                    union[candidate.listing_id] = replace(candidate, fusion_score=score)
        results = sorted(
            union.values(),
            key=lambda c: (
                -int("gtin" in c.sources),
                -int("brand_model" in c.sources),
                -c.fusion_score,
                c.listing_id,
            ),
        )[:k]
        return [replace(c, rank=i) for i, c in enumerate(results, 1)]
