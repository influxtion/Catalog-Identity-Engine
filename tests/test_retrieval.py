import pytest

from catalog_identity.normalization import normalize
from catalog_identity.retrieval import CandidateIndex
from catalog_identity.schema import Listing


def record(id, title="", **kwargs):
    return normalize(Listing(source="test", source_id=id, title=title, **kwargs))


def test_identifier_union_deduplication_and_self_exclusion():
    a = record("a", "Aurora headset black", brand="Aurora", model="H1", gtin="036000291452")
    b = record("b", "Aurora black headset", brand="Aurora", model="H-1", gtin="0036000291452")
    c = record("c", "Aurora headset silver", brand="Aurora", model="H1")
    index = CandidateIndex().fit([a, b, c]).build([c, b, a])
    candidates = index.query(a, 10)
    assert [x.listing_id for x in candidates] == [b.key, c.key]
    assert candidates[0].sources == ("brand_model", "gtin", "tfidf")
    assert [x.rank for x in candidates] == [1, 2]


def test_invalid_identifiers_and_unknown_brands_do_not_retrieve():
    a = record("a", gtin="bad", model="same")
    b = record("b", gtin="bad", model="same")
    # Matching models may still retrieve through text, but never an identifier channel.
    candidates = CandidateIndex().fit([a, b]).build([b]).query(a)
    assert all(x.sources == ("tfidf",) for x in candidates)


def test_train_only_vocabulary_and_deterministic_ties():
    train = record("train", "training lamp")
    a, b = record("a", "training lamp"), record("b", "training lamp")
    unseen = record("unseen", "zzzzzz")
    index = CandidateIndex().fit([train])
    vocabulary = index.vectorizer.vocabulary_.copy()
    index.build([b, unseen, a])
    assert index.vectorizer.vocabulary_ == vocabulary
    assert [x.listing_id for x in index.query(train)] == sorted([a.key, b.key])
    assert len(index.query(train, 1)) == 1
    assert index.query(unseen) == []


def test_empty_text_can_retrieve_by_gtin():
    a = record("a", gtin="036000291452")
    b = record("b", gtin="036000291452")
    index = CandidateIndex().fit([a]).build([b])
    assert index.query(a)[0].sources == ("gtin",)
    assert index.build([]).query(a) == []


def test_invalid_lifecycle_and_duplicate_input():
    a = record("a", "lamp")
    with pytest.raises(ValueError, match="build"):
        CandidateIndex().query(a)
    with pytest.raises(ValueError, match="fit"):
        CandidateIndex().build([a])
    with pytest.raises(ValueError, match="duplicate"):
        CandidateIndex().fit([a]).build([a, a])
    with pytest.raises(ValueError, match="positive"):
        CandidateIndex().fit([a]).build([a]).query(a, 0)
