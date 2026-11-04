"""Optional frozen semantic features on unchanged v3; importing needs no torch."""

import hashlib
import json
import os
import uuid
import zipfile
from collections import OrderedDict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from catalog_identity.matching.distinctions import DISTINCTION_NAMES, DistinctionFeatures
from catalog_identity.normalization import NORMALIZATION_VERSION, normalize

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DIMENSION = 384
TOKEN_LIMIT = 256
PREPROCESSING = "normalized-separate-fields-v1"
DEFAULT_CACHE = Path("runs/semantic-cache")
SUMMARY_NAMES = (
    "semantic_title_cosine",
    "semantic_description_cosine",
    "semantic_cross_cosine_min",
    "semantic_cross_cosine_max",
    "semantic_title_both_available",
    "semantic_title_one_sided",
    "semantic_description_both_available",
    "semantic_description_one_sided",
    "semantic_cross_both_available",
    "semantic_cross_any_available",
)
SEMANTIC_NAMES = DISTINCTION_NAMES + SUMMARY_NAMES
INTERACTION_NAMES = (
    SEMANTIC_NAMES
    + tuple(f"semantic_title_absdiff_{i}" for i in range(DIMENSION))
    + tuple(f"semantic_title_product_{i}" for i in range(DIMENSION))
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class OfflineEncoder:
    """One lazily constructed CPU encoder, with hashes of actual cached assets."""

    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download

        from catalog_identity.retrieval.semantic import SentenceEncoder

        self.encoder = SentenceEncoder(model=MODEL, revision=REVISION, allow_download=False)
        model = self.encoder.model
        if (
            model.get_sentence_embedding_dimension() != DIMENSION
            or model.max_seq_length != TOKEN_LIMIT
        ):
            raise ValueError("cached encoder dimension/token limit differs from registered encoder")
        snapshot = Path(
            snapshot_download(MODEL, revision=REVISION, local_files_only=True, token=False)
        )
        required = (
            "model.safetensors",
            "config.json",
            "config_sentence_transformers.json",
            "sentence_bert_config.json",
            "modules.json",
            "1_Pooling/config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "README.md",
        )
        self.metadata = {
            "model": MODEL,
            "revision": REVISION,
            "dimension": DIMENSION,
            "max_tokens": TOKEN_LIMIT,
            "device": "cpu",
            "normalized": True,
            "offline": True,
            "trust_remote_code": False,
            "license": "Apache-2.0, declared by cached model card; not a data license",
            "asset_sha256": {name: digest(snapshot / name) for name in required},
            "tokenizer_class": type(model.tokenizer).__name__,
            "tokenizer_do_lower_case": bool(getattr(model.tokenizer, "do_lower_case", False)),
            "dependencies": {
                name: version(name)
                for name in ("torch", "transformers", "tokenizers", "sentence-transformers")
            },
        }
        self.runtime = {
            "batch_size": 64,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        }

    def encode(self, texts):
        return self.encoder.encode(texts)

    def token_lengths(self, texts):
        # Count before the registered 256-token truncation, including special tokens.
        return [
            len(ids)
            for ids in self.encoder.model.tokenizer(
                texts, truncation=False, padding=False, add_special_tokens=True
            )["input_ids"]
        ]


class EmbeddingCache:
    """Content-addressed .npz records, no pickle; identity and unit norms checked."""

    def __init__(self, directory=DEFAULT_CACHE, encoder=None, max_memory_entries=16384):
        if type(max_memory_entries) is not int or max_memory_entries < 1:
            raise ValueError("embedding memory limit must be a positive integer")
        self.directory = Path(directory)
        self._encoder = encoder
        self.memory = OrderedDict()
        self.max_memory_entries = max_memory_entries
        self.stats = {
            "encoded_texts": 0,
            "disk_hits": 0,
            "memory_hits": 0,
            "encode_seconds": 0.0,
            "cache_read_seconds": 0.0,
            "encoder_load_seconds": 0.0,
        }

    @property
    def encoder(self):
        if self._encoder is None:
            started = perf_counter()
            self._encoder = OfflineEncoder()
            self.stats["encoder_load_seconds"] += perf_counter() - started
        return self._encoder

    @property
    def metadata(self):
        metadata = self.encoder.metadata
        expected = {
            "model": MODEL,
            "revision": REVISION,
            "dimension": DIMENSION,
            "max_tokens": TOKEN_LIMIT,
            "device": "cpu",
            "normalized": True,
            "offline": True,
            "trust_remote_code": False,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("encoder identity differs from registered offline specification")
        if not metadata.get("asset_sha256"):
            raise ValueError("encoder asset hashes are required")
        return metadata

    def descriptor(self, field, text):
        if field not in ("title", "description"):
            raise ValueError("unknown semantic field")
        return {
            "encoder": self.metadata,
            "normalization": NORMALIZATION_VERSION,
            "preprocessing": PREPROCESSING,
            "field": field,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }

    def key(self, field, text):
        return hashlib.sha256(canonical(self.descriptor(field, text)).encode()).hexdigest()

    @staticmethod
    def validate(vector, tokens):
        if (
            vector.dtype != np.float32
            or vector.shape != (DIMENSION,)
            or not np.isfinite(vector).all()
        ):
            raise ValueError("embedding must be finite float32 with registered dimension")
        if not np.isclose(np.linalg.norm(vector), 1.0, rtol=1e-4, atol=1e-5):
            raise ValueError("nonempty embedding must have unit norm")
        if type(tokens) is not int or tokens < 1:
            raise ValueError("nonempty embedding requires a positive token count")

    def _remember(self, key, value):
        self.memory[key] = value
        self.memory.move_to_end(key)
        while len(self.memory) > self.max_memory_entries:
            self.memory.popitem(last=False)

    def vectors(self, field, texts):
        vectors = np.zeros((len(texts), DIMENSION), dtype=np.float32)
        lengths = np.zeros(len(texts), dtype=np.int64)
        pending = {}
        resolved = {}
        descriptors = {}
        keys = []
        for text in texts:
            if not text:
                keys.append(None)
                continue
            descriptor = canonical(self.descriptor(field, text))
            key = hashlib.sha256(descriptor.encode()).hexdigest()
            keys.append(key)
            if key in resolved:
                continue
            if key in self.memory:
                resolved[key] = self.memory[key]
                self.memory.move_to_end(key)
                self.stats["memory_hits"] += 1
                continue
            if key in pending:
                continue
            path = self.directory / f"{key}.npz"
            if path.exists():
                started = perf_counter()
                try:
                    with np.load(path, allow_pickle=False) as record:
                        if set(record.files) != {"descriptor", "vector", "tokens"}:
                            raise ValueError("unexpected cache fields")
                        if (
                            record["descriptor"].shape != ()
                            or record["descriptor"].item() != descriptor
                        ):
                            raise ValueError("cache identity mismatch")
                        vector = record["vector"]
                        token_array = record["tokens"]
                        if token_array.shape != () or token_array.dtype.kind not in "iu":
                            raise ValueError("invalid cached token count")
                        count = int(token_array.item())
                        self.validate(vector, count)
                        resolved[key] = (vector, count)
                        self._remember(key, resolved[key])
                except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
                    raise ValueError(f"invalid embedding cache record {path.name}: {exc}") from exc
                self.stats["disk_hits"] += 1
                self.stats["cache_read_seconds"] += perf_counter() - started
            else:
                pending[key] = text
                descriptors[key] = descriptor
        if pending:
            self.directory.mkdir(parents=True, exist_ok=True)
            items = list(pending.items())
            for start in range(0, len(items), 64):
                batch = items[start : start + 64]
                batch_texts = [text for _, text in batch]
                started = perf_counter()
                encoded = np.asarray(self.encoder.encode(batch_texts))
                counts = self.encoder.token_lengths(batch_texts)
                if encoded.shape != (len(batch), DIMENSION) or len(counts) != len(batch):
                    raise ValueError("encoder output shape/count differs from requested batch")
                self.stats["encode_seconds"] += perf_counter() - started
                self.stats["encoded_texts"] += len(batch)
                for (key, _), vector, count in zip(batch, encoded, counts, strict=True):
                    self.validate(vector, count)
                    temporary = self.directory / f".{key}.{uuid.uuid4().hex}.tmp"
                    try:
                        with temporary.open("wb") as stream:
                            np.savez(
                                stream,
                                descriptor=np.asarray(descriptors[key]),
                                vector=vector,
                                tokens=np.asarray(count, dtype=np.int64),
                            )
                        os.replace(temporary, self.directory / f"{key}.npz")
                    finally:
                        temporary.unlink(missing_ok=True)
                    resolved[key] = (vector, count)
                    self._remember(key, resolved[key])
        for index, key in enumerate(keys):
            if key is not None:
                vectors[index], lengths[index] = resolved[key]
        return vectors, lengths

    def prepare(self, listings):
        records = [normalize(listing) for listing in listings.values()]
        report = {"listings": len(records), "fields": {}}
        for field in ("title", "description"):
            texts = [getattr(record, field) for record in records]
            _, lengths = self.vectors(field, texts)
            report["fields"][field] = {
                "missing": sum(not text for text in texts),
                "truncated": int((lengths > TOKEN_LIMIT).sum()),
                "available": sum(bool(text) for text in texts),
            }
        return report


class SemanticFeatures(DistinctionFeatures):
    names = SEMANTIC_NAMES
    version = "5"

    def __init__(self, embedding_cache=None):
        super().__init__()
        self.embeddings = embedding_cache if embedding_cache is not None else EmbeddingCache()
        self.required_encoder = None

    def check_encoder(self):
        current = self.embeddings.metadata
        if self.required_encoder is not None and self.required_encoder != current:
            raise ValueError("saved matcher encoder requirements differ from local encoder")

    def transform(self, listings, pairs):
        base = super().transform(listings, pairs)
        if not pairs:
            return np.empty((0, len(self.names)))
        self.check_encoder()
        records = [normalize(value) for value in listings.values()]
        positions = {key: i for i, key in enumerate(listings)}
        title, _ = self.embeddings.vectors("title", [record.title for record in records])
        desc, _ = self.embeddings.vectors("description", [record.description for record in records])
        left, right = ([positions[p[i]] for p in pairs] for i in (0, 1))
        ta, tb, da, db = title[left], title[right], desc[left], desc[right]

        def available(matrix):
            return np.any(matrix != 0, axis=1)

        at, bt, ad, bd = (available(m) for m in (ta, tb, da, db))

        def dot(a, b):
            return np.clip(np.sum(a * b, axis=1), -1.0, 1.0)

        cross = np.sort(np.column_stack((dot(ta, db), dot(tb, da))), axis=1)
        cross_available = np.sort(np.column_stack((at & bd, bt & ad)), axis=1)
        summary = np.column_stack(
            (dot(ta, tb), dot(da, db), cross, at & bt, at ^ bt, ad & bd, ad ^ bd, cross_available)
        )
        if self.version == "6":
            # Missing title vectors are zero; availability flags distinguish missing evidence.
            return np.column_stack((base, summary, np.abs(ta - tb), ta * tb))
        return np.column_stack((base, summary))

    def save(self, path):
        self.check_encoder()
        super().save(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["semantic_encoder"] = self.embeddings.metadata
        data["semantic_preprocessing"] = PREPROCESSING
        data["semantic_normalization"] = NORMALIZATION_VERSION
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path):
        result = super().load(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data["semantic_preprocessing"] != PREPROCESSING
            or data["semantic_normalization"] != NORMALIZATION_VERSION
        ):
            raise ValueError("saved semantic preprocessing differs from implementation")
        result.required_encoder = data["semantic_encoder"]
        return result


class SemanticInteractionFeatures(SemanticFeatures):
    names = INTERACTION_NAMES
    version = "6"
