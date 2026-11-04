import json
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from sqlalchemy import select

from catalog_identity.annotations.schema import GUIDELINE_VERSION, SaveRequest
from catalog_identity.annotations.store import AnnotationStore, members
from catalog_identity.api.app import create_app
from catalog_identity.evaluation.training_audit import fingerprint
from catalog_identity.storage.catalog import Conflict
from catalog_identity.storage.database import Database


def packet(tmp_path):
    path = tmp_path / "blind.jsonl"
    rows = [
        {
            "review_id": fingerprint(["fiction", i]),
            "left": {"title": f"Fictional Aster {i} blue lamp", "brand": "Aster"},
            "right": {"title": f"Fictional Aster {i} amber lamp", "brand": "Aster"},
            "judgment": {
                "input_sufficiency": None,
                "benchmark_identity": None,
                "family_relation": None,
                "variant_relation": None,
                "evidence_fields": [],
                "reason": None,
            },
        }
        for i in range(2)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path, rows


@pytest.fixture
def workspace(tmp_path):
    path, rows = packet(tmp_path)
    app = create_app("sqlite:///" + str(tmp_path / "catalog.db"))
    store = app.state.annotations
    access = store.initialize(path, "Fictional review", ["alice", "bob"], "carol")
    headers = {r["name"]: {"Authorization": "Bearer " + r["access_key"]} for r in access["members"]}
    with TestClient(app) as client:
        yield client, store, access, headers, rows
    store.db.close()


def payload(case, benchmark="uncertain"):
    return {
        "case_hash": case["case_hash"],
        "revision": case["revision"],
        "guideline_version": GUIDELINE_VERSION,
        "submit": True,
        "judgment": {
            "input_sufficiency": "sufficient",
            "benchmark_identity": benchmark,
            "family_relation": "same",
            "variant_relation": "different",
            "evidence": [
                {
                    "side": "left",
                    "field": "title",
                    "quote": "blue",
                    "role": "variant_attribute",
                    "interpretation": "Observed color",
                }
            ],
            "sources": [],
            "reason": "Colors differ; benchmark identity is not inferred.",
        },
    }


def test_blinding_locking_adjudication_and_export(workspace):
    client, store, access, headers, rows = workspace
    url = f"/annotations/cases/{rows[0]['review_id']}"
    assert client.get("/annotations/session").status_code == 401
    assert (
        client.get("/annotations/session", headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )
    assert client.get(url, headers=headers["carol"]).status_code == 409
    assert client.get("/annotations/cases", headers=headers["carol"]).json() == []
    a = client.get(url, headers=headers["alice"])
    assert a.headers["cache-control"] == "no-store"
    request = payload(a.json())
    assert client.post(url, headers=headers["alice"], json=request).status_code == 200
    assert client.post(url, headers=headers["alice"], json=request).json()["revision"] == 1
    request["judgment"]["reason"] = "A different retrospective explanation."
    assert client.post(url, headers=headers["alice"], json=request).status_code == 409
    b = client.get(url, headers=headers["bob"]).json()
    assert b["judgment"]["reason"] == "" and "independent_reviews" not in b
    assert "resolution" not in b
    assert client.post(url, headers=headers["bob"], json=payload(b, "different")).status_code == 200
    adjudication = client.get(url, headers=headers["carol"]).json()
    assert adjudication["resolution"] == "needs_adjudication"
    assert len(adjudication["independent_reviews"]) == 2
    assert client.post(url, headers=headers["carol"], json=payload(adjudication)).status_code == 200
    exported = store.export(access["batch_id"])
    assert exported["summary"]["both_reviewed"] == 1
    assert exported["summary"]["disagreement_rate"] == 1
    completed = next(r for r in exported["cases"] if r["case"]["review_id"] == rows[0]["review_id"])
    assert completed["status"] == "adjudicated"
    assert completed["resolved_decisions"]["benchmark_identity"] == "uncertain"
    assert len(completed["submissions"]) == 3
    assert not completed["automatically_training_eligible"]
    assert not exported["official_labels_modified"]
    assert "token_hash" not in json.dumps(exported) and "access_key" not in json.dumps(exported)
    assert client.get("/annotations/export", headers=headers["alice"]).status_code == 404
    request["reviewer"] = "carol"
    assert client.post(url, headers=headers["bob"], json=request).status_code == 422


def test_drafts_stale_requests_and_evidence_validation(workspace):
    client, _, _, headers, rows = workspace
    url = "/annotations/cases/" + rows[0]["review_id"]
    case = client.get(url, headers=headers["alice"]).json()
    draft = {**payload(case), "submit": False, "judgment": {"reason": "Partial thought"}}
    assert client.post(url, headers=headers["alice"], json=draft).status_code == 200
    draft["judgment"]["reason"] = "Stale browser tab"
    assert client.post(url, headers=headers["alice"], json=draft).status_code == 409
    current = client.get(url, headers=headers["alice"]).json()
    assert current["judgment"]["reason"] == "Partial thought" and current["revision"] == 1
    bad = payload(current)
    bad["judgment"]["evidence"][0]["quote"] = "invented observation"
    assert client.post(url, headers=headers["alice"], json=bad).status_code == 422
    bad = payload(current)
    bad["judgment"]["family_relation"] = "different"
    bad["judgment"]["variant_relation"] = "same"
    assert client.post(url, headers=headers["alice"], json=bad).status_code == 422
    bad = payload(current)
    bad["case_hash"] = "0" * 64
    assert client.post(url, headers=headers["alice"], json=bad).status_code == 409
    bad = payload(current)
    bad["judgment"]["evidence"] = []
    assert client.post(url, headers=headers["alice"], json=bad).status_code == 422
    for name in ("benchmark_identity", "family_relation", "variant_relation"):
        bad["judgment"][name] = "uncertain"
    bad["judgment"]["input_sufficiency"] = "insufficient"
    assert client.post(url, headers=headers["alice"], json=bad).status_code == 200


def test_rotation_duplicate_import_and_packet_validation(workspace, tmp_path):
    client, store, access, headers, _ = workspace
    replacement = store.rotate(access["batch_id"], "alice")
    assert client.get("/annotations/session", headers=headers["alice"]).status_code == 401
    assert (
        client.get(
            "/annotations/session", headers={"Authorization": "Bearer " + replacement["access_key"]}
        ).status_code
        == 200
    )
    with store.db.transaction() as c:
        saved = c.execute(select(members.c.token_hash)).scalars().all()
    assert replacement["access_key"] not in saved
    path, rows = packet(tmp_path)
    with pytest.raises(Conflict, match="already imported"):
        store.initialize(path, "duplicate", ["alice", "bob"], "carol")
    with pytest.raises(ValueError, match="different reviewers"):
        store.initialize(path, "bad", ["alice", "ALICE"], "carol")
    rows[0]["benchmark_label"] = 1
    path.write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError, match="Extra inputs"):
        store.initialize(path, "leaky", ["a", "b"], "c")


def test_concurrent_drafts_and_persistence(workspace):
    _, store, access, _, rows = workspace
    case_id = rows[0]["review_id"]
    actor = store.authenticate(access["members"][0]["access_key"])
    body = payload(store.case(actor, case_id))
    body["submit"] = False
    other_db = Database(str(store.db.engine.url))
    other = AnnotationStore(other_db)

    def save(service):
        try:
            service.save(actor, case_id, SaveRequest.model_validate(body))
            return "saved"
        except Conflict:
            return "stale"

    with ThreadPoolExecutor(2) as executor:
        assert sorted(executor.map(save, (store, other))) == ["saved", "stale"]
    assert other.case(actor, case_id)["revision"] == 1
    assert other.export(access["batch_id"])["summary"]["both_reviewed"] == 0
    other_db.close()


def test_cli_import_export_keep_keys_out_of_logs(tmp_path, monkeypatch, capsys):
    from catalog_identity.cli import main

    path, _ = packet(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + str(tmp_path / "cli.db"))
    output = tmp_path / "keys"
    assert (
        main(
            [
                "annotation-import",
                "--packet",
                str(path),
                "--reviewers",
                "a",
                "b",
                "--adjudicator",
                "c",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    access = json.loads((output / "access-keys.json").read_text())
    captured = capsys.readouterr().out
    assert all(m["access_key"] not in captured for m in access["members"])
    assert (output / ".gitignore").read_text().strip() == "*"
    assert (
        main(
            [
                "annotation-export",
                "--batch",
                access["batch_id"],
                "--output",
                str(tmp_path / "export"),
            ]
        )
        == 0
    )
    summary = json.loads((tmp_path / "export/summary.json").read_text())
    assert summary["summary"]["status_counts"] == {"awaiting_reviews": 2}
    assert summary["summary"]["disagreement_rate"] is None
