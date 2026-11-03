import pytest

from catalog_identity.clustering.engine import Edge, cluster, structured_edges
from catalog_identity.clustering.policy import compare_identity
from catalog_identity.clustering.registry import IdentityRegistry
from catalog_identity.evaluation.clusters import cluster_metrics
from catalog_identity.normalization import normalize
from catalog_identity.schema import Listing


def record(id, model="H1", color="black", category="headphones"):
    return normalize(
        Listing(
            source="showcase",
            source_id=id,
            title="Fictional headset",
            brand="Aster",
            model=model,
            category=category,
            attributes={"color": color},
        )
    )


def test_bridge_edge_does_not_merge_conflicting_endpoints():
    a, b, c = record("a", "H1"), record("b", None), record("c", "H2")
    listings = {x.key: x for x in (a, b, c)}
    edges = [Edge(a.key, b.key, 0.99), Edge(b.key, c.key, 0.98)]
    assert len(cluster(listings, edges, constrained=False).clusters) == 1
    result = cluster(listings, edges)
    assert len(result.clusters) == 2
    assert result.rejected[0]["reason"] in ("identity_conflict", "missing_cross_cluster_support")
    assert not any(a.key in group and c.key in group for group in result.clusters)


def test_family_and_variant_are_separate():
    a, b, c = record("a"), record("b", color="silver"), record("c")
    listings = {x.key: x for x in (a, b, c)}
    family, variant = structured_edges(listings)
    assert len(cluster(listings, family, "family").clusters) == 1
    assert len(cluster(listings, variant, "variant").clusters) == 2
    assert compare_identity(a, b).relation == "same_family_different_variant"
    assert compare_identity(a, c).relation == "same_exact_variant"


def test_unknown_variant_attributes_and_categories_abstain():
    a, b = record("a"), record("b", color=None)
    assert compare_identity(a, b).variant == "unknown"
    result = cluster({a.key: a, b.key: b}, [Edge(a.key, b.key, 1)], "variant")
    assert len(result.clusters) == 2
    assert compare_identity(a, record("c", category="unknown")).variant == "unknown"


def test_complete_cross_cluster_support_is_required():
    a, b, c = record("a"), record("b"), record("c")
    result = cluster(
        {x.key: x for x in (a, b, c)}, [Edge(a.key, b.key, 1), Edge(b.key, c.key, 0.99)]
    )
    assert len(result.clusters) == 2
    assert result.rejected[0]["reason"] == "missing_cross_cluster_support"


def test_stable_ids_and_merge_split_history():
    registry = IdentityRegistry()
    first = registry.reconcile([("a", "b"), ("c",)])
    ab = next(key for key, members in first.items() if "a" in members)
    assert registry.reconcile([("a", "b"), ("c",)]) == first
    assert registry.revision == 1
    merged = registry.reconcile([("a", "b", "c")])
    assert next(iter(merged)) == ab
    split = registry.reconcile([("a", "b"), ("c",)])
    assert split[ab] == ("a", "b")
    assert any(event["event"] == "split" for event in registry.history)
    restored = IdentityRegistry.from_dict(registry.to_dict())
    assert restored.active == registry.active
    with pytest.raises(ValueError, match="partition"):
        registry.reconcile([("a",), ("a",)])


def test_cluster_metrics_penalize_bridge_and_fragmentation():
    truth = {"a": "x", "b": "x", "c": "y"}
    merged = cluster_metrics([("a", "b", "c")], truth)
    assert merged["pair_precision"] == pytest.approx(1 / 3)
    assert merged["erroneous_merges"] == 1
    fragmented = cluster_metrics([("a",), ("b",), ("c",)], truth)
    assert fragmented["fragmentation"] == 1 and fragmented["pair_recall"] == 0
    perfect = cluster_metrics([("a", "b"), ("c",)], truth)
    assert perfect["pair_f1"] == perfect["b_cubed_recall"] == perfect["purity"] == 1
