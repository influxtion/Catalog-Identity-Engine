from dataclasses import asdict
from pathlib import Path

from catalog_identity.clustering.engine import Edge, cluster, structured_edges
from catalog_identity.evaluation.clusters import cluster_metrics
from catalog_identity.evaluation.provenance import implementation_manifest
from catalog_identity.evaluation.runner import write_json
from catalog_identity.normalization import normalize
from catalog_identity.showcase import showcase


def cluster_showcase(output: Path):
    if output.exists():
        raise ValueError("output already exists")
    records = showcase()
    normalized = {x.key: normalize(x) for x in records}
    family_edges, variant_edges = structured_edges(normalized)
    # Explicit fixture labels; ambiguous records are excluded from variant truth.
    family_groups = [(0, 1, 2, 3, 4), (5,), (6, 7, 8), (9, 10)]
    variant_groups = [(0, 1), (2, 3), (5,), (6, 7), (8,), (9,), (10,)]
    report = {
        "synthetic": True,
        "warning": "Original fictional fixture, not real-world or human-labeled accuracy.",
        "implementation": implementation_manifest(),
        "levels": {},
    }
    for level, edges, truth_groups in (
        ("family", family_edges, family_groups),
        ("variant", variant_edges, variant_groups),
    ):
        truth = {
            records[i].key: str(group)
            for group, members in enumerate(truth_groups)
            for i in members
        }
        result = cluster(normalized, edges, level)
        labeled_groups = [
            tuple(key for key in members if key in truth) for members in result.clusters
        ]
        labeled_groups = [group for group in labeled_groups if group]
        report["levels"][level] = {
            "labeled_listings": len(truth),
            "excluded_unresolved": len(records) - len(truth),
            "metrics": cluster_metrics(labeled_groups, truth),
        }
    a, b, c = (records[i] for i in (0, 1, 5))
    b = b.model_copy(update={"model": None})
    listings = {x.key: normalize(x) for x in (a, b, c)}
    edges = [Edge(a.key, b.key, 0.99), Edge(b.key, c.key, 0.98)]
    truth = {a.key: "first", b.key: "first", c.key: "second"}
    report["bridge_ablation"] = {}
    for name, constrained in (("connected_components", False), ("constrained", True)):
        result = cluster(listings, edges, constrained=constrained)
        report["bridge_ablation"][name] = {
            "metrics": cluster_metrics(result.clusters, truth),
            "graph": asdict(result),
        }
    output.mkdir(parents=True)
    write_json(output / "clusters.json", report)
    return report
