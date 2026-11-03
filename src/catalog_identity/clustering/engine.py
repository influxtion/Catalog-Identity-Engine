from dataclasses import asdict, dataclass
from itertools import combinations

from catalog_identity.clustering.policy import compare_identity


@dataclass(frozen=True)
class Edge:
    left: str
    right: str
    score: float


@dataclass(frozen=True)
class ClusterResult:
    clusters: tuple[tuple[str, ...], ...]
    accepted: tuple[Edge, ...]
    rejected: tuple[dict, ...]


def cluster(listings, edges, level="family", threshold=0.95, constrained=True):
    """Complete cross-cluster support avoids unsupported transitive bridges.

    Scores must have the requested identity semantics. Binary WDC probabilities are
    not family/variant probabilities. Missing cross edges cause abstention.
    """
    if level not in ("family", "variant") or not 0 <= threshold <= 1:
        raise ValueError("invalid clustering policy")
    scores = {}
    for edge in edges:
        if edge.left not in listings or edge.right not in listings or edge.left == edge.right:
            raise ValueError("edge must reference two distinct known listings")
        if not 0 <= edge.score <= 1:
            raise ValueError("edge score must be in [0, 1]")
        key = tuple(sorted((edge.left, edge.right)))
        if key in scores:
            raise ValueError("duplicate edge")
        scores[key] = edge.score
    groups = {key: {key} for key in listings}
    owners = {key: key for key in listings}
    accepted, rejected = [], []
    for (left, right), score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        edge = Edge(left, right, score)
        if score < threshold:
            rejected.append({**asdict(edge), "reason": "below_threshold"})
            continue
        a, b = owners[left], owners[right]
        if a == b:
            continue
        reason = None
        witness = None
        if constrained:
            for x in sorted(groups[a]):
                for y in sorted(groups[b]):
                    evidence = compare_identity(listings[x], listings[y])
                    if getattr(evidence, level) == "different":
                        reason, witness = (
                            "identity_conflict",
                            {"left": x, "right": y, **asdict(evidence)},
                        )
                        break
                    if scores.get(tuple(sorted((x, y))), -1) < threshold:
                        reason, witness = "missing_cross_cluster_support", {"left": x, "right": y}
                        break
                    # Variant merges require explicit category-specific agreement.
                    if level == "variant" and evidence.variant != "same":
                        reason, witness = (
                            "unknown_variant_attributes",
                            {"left": x, "right": y, **asdict(evidence)},
                        )
                        break
                if reason:
                    break
        if reason:
            rejected.append({**asdict(edge), "reason": reason, "witness": witness})
            continue
        keep, remove = sorted((a, b))
        groups[keep].update(groups.pop(remove))
        for key in groups[keep]:
            owners[key] = keep
        accepted.append(edge)
    return ClusterResult(
        tuple(sorted(tuple(sorted(group)) for group in groups.values())),
        tuple(accepted),
        tuple(rejected),
    )


def structured_edges(listings):
    """Generate only within known brand/model blocks for the structured showcase.

    These are deterministic rule confidences, not calibrated model probabilities.
    Unknown variant attributes never create exact-variant edges.
    """
    blocks = {}
    for key, listing in listings.items():
        if listing.brand and listing.model:
            blocks.setdefault((listing.brand, listing.model), []).append(key)
    family, variant = [], []
    for members in blocks.values():
        for a, b in combinations(sorted(members), 2):
            evidence = compare_identity(listings[a], listings[b])
            if evidence.family == "same":
                family.append(Edge(a, b, 1.0))
            if evidence.variant == "same":
                variant.append(Edge(a, b, 1.0))
    return family, variant
