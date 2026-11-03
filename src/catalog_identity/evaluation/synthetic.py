"""Original fictional fixture; useful for plumbing, not benchmark accuracy."""

import json
import random
from itertools import combinations
from pathlib import Path

SYNTHETIC_VERSION = "1"


def write_synthetic(directory: Path, seed: int = 42) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    paths = {}
    for split_number, (split, families) in enumerate(
        (("train", 12), ("validation", 6), ("test", 6))
    ):
        offers = []
        for family in range(families):
            generation = 10 + split_number * 20 + family
            brand = ("Aster", "Lumen", "Morrow")[family % 3]
            category = ("music player", "tablet", "media console")[family % 3]
            for capacity in (64, 128):
                entity = f"{split}-{family}-{capacity}"
                for store in range(3):
                    titles = [
                        f"{brand} Arc-{generation} {category} {capacity}GB black",
                        f"{category} ARC {generation} by {brand} - {capacity} GB charcoal",
                        f"{brand} Arc{generation} {capacity}GB {category}",
                    ]
                    offers.append(
                        {
                            "id": f"{entity}-store{store}",
                            "cluster_id": entity,
                            "title": titles[store],
                            "brand": brand if store != 1 else None,
                            "description": f"Portable {category} with rechargeable battery.",
                            "price": str(60 + generation + capacity / 4 + store),
                            "priceCurrency": "USD",
                        }
                    )
        rows = []
        for a, b in combinations(offers, 2):
            positive = a["cluster_id"] == b["cluster_id"]
            hard = a["description"] == b["description"]
            if not positive and rng.random() > (0.18 if hard else 0.025):
                continue
            rows.append(
                {
                    **{f"{key}_left": value for key, value in a.items()},
                    **{f"{key}_right": value for key, value in b.items()},
                    "pair_id": f"{a['id']}#{b['id']}",
                    "label": int(positive),
                    "is_hard_negative": not positive and hard,
                }
            )
        path = directory / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        paths[split] = path
    return paths
