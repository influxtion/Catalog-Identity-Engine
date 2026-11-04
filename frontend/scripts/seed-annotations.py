"""Original fictional annotation cases for browser tests, never benchmark annotations."""

import json
import os
import sys
from pathlib import Path

from catalog_identity.annotations.store import AnnotationStore
from catalog_identity.evaluation.training_audit import fingerprint
from catalog_identity.storage.database import Database

folder = Path(sys.argv[1])
packet = folder / "fictional-packet.jsonl"
rows = [
    {
        "review_id": fingerprint(["browser-fiction", index]),
        "left": {
            "title": f"Fictional Aster {index} blue desk lamp",
            "brand": "Aster",
            "description": "A fictional lamp with a blue finish for workflow testing.",
        },
        "right": {
            "title": f"Fictional Aster {index} amber desk lamp",
            "brand": "Aster",
            "description": "A fictional lamp with an amber finish for workflow testing.",
        },
        "judgment": {
            "input_sufficiency": None,
            "benchmark_identity": None,
            "family_relation": None,
            "variant_relation": None,
            "evidence_fields": [],
            "reason": None,
        },
    }
    for index in range(2)
]
packet.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
database = Database(os.environ["DATABASE_URL"])
access = AnnotationStore(database).initialize(
    packet, "Fictional evidence review", ["alice", "bob"], "carol"
)
output = Path("runs/annotation-e2e/access.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(access), encoding="utf-8")
database.close()
