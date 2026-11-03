import numpy as np
import pytest

from catalog_identity.normalization import normalize
from catalog_identity.retrieval.semantic import HybridIndex, SemanticIndex
from catalog_identity.schema import Listing

pytest.importorskip("faiss")


class FixedEncoder:
    def encode(self, texts):
        return np.array(
            [[1, 0] if "lamp" in t or "light" in t else [0, 1] for t in texts], dtype=np.float32
        )


def records():
    return [
        normalize(Listing(source="test", source_id=str(i), title=title))
        for i, title in enumerate(("desk lamp", "office light", "kitchen kettle"))
    ]


def test_semantic_self_exclusion_and_nearest_neighbor():
    a, b, c = records()
    index = SemanticIndex(FixedEncoder()).build([c, b, a])
    assert index.query(a, 1)[0].listing_id == b.key
    assert index.query(a, 1)[0].semantic_score == pytest.approx(1)
    assert index.index_bytes > 0
    assert index.build([]).query(a) == []


def test_hybrid_union_ranks_and_deduplication():
    a, b, c = records()
    index = HybridIndex(FixedEncoder()).fit([a, b, c]).build([a, b, c])
    result = index.query(a, 10)
    assert len({x.listing_id for x in result}) == len(result) == 2
    assert all(x.listing_id != a.key for x in result)
    assert [x.rank for x in result] == [1, 2]
    assert "semantic" in result[0].sources


def test_invalid_embeddings_are_rejected():
    class BrokenEncoder:
        def encode(self, texts):
            return np.zeros((len(texts), 3))

    with pytest.raises(ValueError, match="zero vectors"):
        SemanticIndex(BrokenEncoder()).build(records())
