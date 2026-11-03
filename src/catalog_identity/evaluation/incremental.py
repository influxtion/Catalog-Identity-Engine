"""Compare incremental and fresh memberships using original synthetic records."""

from pathlib import Path
from time import perf_counter

from catalog_identity.evaluation.provenance import implementation_manifest
from catalog_identity.evaluation.runner import write_json
from catalog_identity.showcase import showcase
from catalog_identity.storage.catalog import Catalog
from catalog_identity.storage.database import Database


def memberships(catalog, level):
    return {frozenset(group["members"]): group["id"] for group in catalog.identities(level)}


def incremental_showcase(output: Path):
    if output.exists():
        raise ValueError("output already exists")
    report = {
        "synthetic": True,
        "scope": "12 fictional listings, SQLite; correctness check, not production scaling.",
        "implementation": implementation_manifest(),
        "steps": [],
    }
    database = Database("sqlite:///:memory:")
    catalog = Catalog(database)
    records = showcase()
    try:
        catalog.ingest(records)
        catalog.rebuild()
        report["identical_reingestion_changed"] = catalog.ingest(records)["changed"]
        for name, index, color in (
            ("resolve_unknown_variant", 4, "black"),
            ("split_variant", 1, "blue"),
        ):
            before = {level: memberships(catalog, level) for level in ("family", "variant")}
            records[index] = records[index].model_copy(
                update={"attributes": {**records[index].attributes, "color": color}}
            )
            started = perf_counter()
            result = catalog.ingest([records[index]])
            stale_before_rebuild = catalog.overview()["identities_stale"]
            catalog.rebuild()
            elapsed = perf_counter() - started
            fresh_database = Database("sqlite:///:memory:")
            try:
                fresh = Catalog(fresh_database)
                fresh.ingest(records)
                fresh.rebuild()
                levels = {}
                for level in before:
                    after = memberships(catalog, level)
                    unchanged = before[level].keys() & after.keys()
                    levels[level] = {
                        "matches_fresh_partition": after.keys() == memberships(fresh, level).keys(),
                        "unchanged_groups": len(unchanged),
                        "unchanged_ids_retained": sum(
                            before[level][key] == after[key] for key in unchanged
                        ),
                        "groups": len(after),
                    }
                report["steps"].append(
                    {
                        "change": name,
                        "changed_listings": result["changed"],
                        "stale_before_rebuild": stale_before_rebuild,
                        "stale_after_rebuild": catalog.overview()["identities_stale"],
                        "ingest_and_rebuild_seconds": elapsed,
                        "levels": levels,
                    }
                )
            finally:
                fresh_database.close()
        report["history_events"] = [event["event"] for event in catalog.overview()["history"]]
    finally:
        database.close()
    output.mkdir(parents=True)
    write_json(output / "incremental.json", report)
    return report
