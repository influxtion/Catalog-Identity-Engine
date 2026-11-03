import os
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from sqlalchemy import select

from catalog_identity.api.app import create_app
from catalog_identity.showcase import seed, showcase
from catalog_identity.storage import database as tables
from catalog_identity.storage.catalog import Catalog, Conflict
from catalog_identity.storage.database import Database
from catalog_identity.storage.worker import run_one


@pytest.fixture
def client():
    app = create_app("sqlite:///:memory:")
    with TestClient(app) as client:
        yield client
    app.state.catalog.db.close()


def test_ingestion_is_idempotent_and_changed_records_invalidate_scores(client):
    records = showcase()[:2]
    payload = {"listings": [x.model_dump() for x in records]}
    first = client.post("/listings", json=payload)
    assert first.status_code == 201 and first.json()["changed"] == 2
    assert client.post("/listings", json=payload).json()["changed"] == 0
    pair = client.post(
        "/pairs/inspect", json={"left": records[0].key, "right": records[1].key}
    ).json()
    assert pair["hierarchy"]["relation"] == "same_exact_variant"
    assert pair["probability"] is None
    payload["listings"][0]["attributes"]["color"] = "silver"
    assert client.post("/listings", json=payload).json()["changed"] == 1
    catalog = client.app.state.catalog
    with catalog.db.transaction() as connection:
        assert connection.execute(select(tables.pairs.c.stale)).scalar_one()
    assert client.get(f"/listings/{records[0].key}").json()["revision"] == 2
    assert client.get("/overview").json()["identities_stale"]


def test_worker_rebuilds_persistent_family_and_variant_membership(client, tmp_path):
    records = showcase()
    client.post("/listings", json={"listings": [x.model_dump() for x in records]})
    catalog = client.app.state.catalog
    assert run_one(catalog, tmp_path, tmp_path / "runs")
    families = client.get("/identities?level=family").json()
    variants = client.get("/identities?level=variant").json()
    assert len(families) == 5 and len(variants) == 9
    family_ids = {x["id"] for x in families}
    assert all(x["parent_id"] in family_ids for x in variants)
    assert not client.get("/overview").json()["identities_stale"]
    catalog.rebuild()
    assert {x["id"] for x in catalog.identities()} == family_ids


def test_reviews_reject_stale_revisions_and_hard_conflicts(client):
    records = showcase()
    seed(client.app.state.catalog)
    request = {
        "left": records[0].key,
        "right": records[5].key,
        "level": "family",
        "decision": "match",
        "note": "Reviewed model identifiers",
        "left_revision": 1,
        "right_revision": 1,
    }
    assert client.post("/reviews", json=request).status_code == 409
    request.update(decision="reject", left_revision=2)
    assert client.post("/reviews", json=request).status_code == 409
    request["left_revision"] = 1
    assert client.post("/reviews", json=request).status_code == 201
    assert client.get("/jobs").json()[0]["status"] == "pending"


def test_human_rejection_prevents_indirect_merge(client):
    records = showcase()
    catalog = client.app.state.catalog
    seed(catalog)
    catalog.submit_review(
        records[0].key, records[1].key, "family", "reject", "Verified separate models", 1, 1
    )
    catalog.rebuild()
    assert not any(
        records[0].key in group["members"] and records[1].key in group["members"]
        for group in catalog.identities()
    )
    assert any(
        edge["reason"] == "human_rejection" for edge in catalog.overview()["graph"]["rejected"]
    )


def test_jobs_are_idempotent_and_failures_are_durable(client, tmp_path):
    catalog = client.app.state.catalog
    job = catalog.enqueue("evaluate", {"config_name": "missing.json"}, "once")
    assert catalog.enqueue("evaluate", {"config_name": "missing.json"}, "once")["id"] == job["id"]
    with pytest.raises(Conflict):
        catalog.enqueue("rebuild", idempotency_key="once")
    run_one(catalog, tmp_path, tmp_path / "runs")
    assert catalog.jobs()[0]["status"] == "failed"
    assert catalog.jobs()[0]["error"]
    assert catalog.claim_job() is None
    with pytest.raises(Conflict):
        catalog.finish_job(job["id"], result={})


def test_family_review_keeps_unresolved_variant_in_queue(client):
    records = showcase()
    catalog = client.app.state.catalog
    seed(catalog)
    left, right = records[0].key, records[4].key
    catalog.inspect_pair(left, right)
    catalog.submit_review(left, right, "family", "match", "Family verified; color absent", 1, 1)
    assert any({row["left"], row["right"]} == {left, right} for row in catalog.review_queue())
    catalog.submit_review(left, right, "variant", "reject", "Supplier confirmed other color", 1, 1)
    assert not any({row["left"], row["right"]} == {left, right} for row in catalog.review_queue())


def test_variant_acceptance_cannot_override_active_family_rejection(client):
    records = showcase()
    catalog = client.app.state.catalog
    seed(catalog)
    left, right = records[0].key, records[1].key
    catalog.submit_review(left, right, "family", "reject", "Supplier model incorrect", 1, 1)
    with pytest.raises(Conflict, match="family rejection"):
        catalog.submit_review(left, right, "variant", "match", "Same supplied attributes", 1, 1)
    catalog.submit_review(left, right, "family", "match", "Verified the supplier correction", 1, 1)
    catalog.submit_review(left, right, "variant", "match", "Variant details now verified", 1, 1)


def test_search_retrieval_and_bad_requests(client):
    records = showcase()
    seed(client.app.state.catalog)
    assert len(client.get("/listings?q=aster").json()) == 6
    hits = client.get(f"/listings/{records[0].key}/candidates?k=3").json()
    assert len(hits) == 3 and all(x["listing_id"] != records[0].key for x in hits)
    assert client.get("/listings/missing").status_code == 404
    assert client.get("/listings?limit=-1").status_code == 422
    assert (
        client.post("/evaluations", json={"config_name": "../../secrets.json"}).status_code == 422
    )
    assert client.get("/reviews").json()
    assert client.get("/openapi.json").status_code == 200


def test_database_survives_reopen(tmp_path):
    url = "sqlite:///" + str(tmp_path / "catalog.db")
    db = Database(url)
    catalog = Catalog(db)
    seed(catalog)
    ids = {x["id"] for x in catalog.identities()}
    db.close()
    other = Database(url)
    assert {x["id"] for x in Catalog(other).identities()} == ids
    other.close()


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"), reason="PostgreSQL integration URL not configured"
)
def test_postgres_ingestion_and_worker():
    db = Database(os.environ["TEST_DATABASE_URL"])
    catalog = Catalog(db)
    seed(catalog)
    assert len(catalog.search()) == len(showcase())
    catalog.enqueue("rebuild")
    assert run_one(catalog, Path("configs"), Path("runs"))
    assert catalog.jobs()[0]["status"] == "succeeded"
    db.close()
