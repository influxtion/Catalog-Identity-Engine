import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from uuid import uuid4

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from sqlalchemy import select

from catalog_identity.annotations.report import analyze
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


def test_agreement_role_conflicts_and_operator_analysis(workspace, tmp_path):
    client, store, access, headers, rows = workspace
    key_path = tmp_path / "key.jsonl"
    keys = [
        {
            "review_id": row["review_id"],
            "benchmark_label": 1,
            "primary_stratum": "reference",
            "tags": ["code_disagreement_positive"],
        }
        for row in rows
    ]
    key_path.write_text("\n".join(json.dumps(k) for k in keys))
    with pytest.raises(ValueError, match="resolve every case"):
        analyze(store.export(access["batch_id"]), key_path)
    for index, row in enumerate(rows):
        url = "/annotations/cases/" + row["review_id"]
        for name in ("alice", "bob"):
            body = payload(client.get(url, headers=headers[name]).json())
            if index == 0 and name == "bob":
                body["judgment"]["evidence"][0]["role"] = "own_model"
            assert client.post(url, headers=headers[name], json=body).status_code == 200
    exported = store.export(access["batch_id"])
    assert exported["summary"]["status_counts"] == {"needs_adjudication": 1, "agreed": 1}
    with pytest.raises(ValueError, match="resolve every case"):
        analyze(exported, key_path)
    url = "/annotations/cases/" + rows[0]["review_id"]
    body = payload(client.get(url, headers=headers["carol"]).json(), "different")
    body["judgment"]["input_sufficiency"] = "needs_source_check"
    assert client.post(url, headers=headers["carol"], json=body).status_code == 200
    exported = store.export(access["batch_id"])
    report = analyze(exported, key_path)
    assert report["overall"]["cases"] == 2
    assert report["overall"]["disagreement_rate"] == 0.5
    assert report["overall"]["needs_more_evidence_rate"] == 0.5
    assert report["overall"]["benchmark_label_comparison"] == {"differs": 1, "uncertain": 1}
    assert report["primary_strata"]["reference"]["cases"] == 2
    assert report["overlapping_tags"]["code_disagreement_positive"]["cases"] == 2
    assert not report["official_labels_modified"]
    assert key_path.read_text() == "\n".join(json.dumps(k) for k in keys)
    key_path.write_text(json.dumps(keys[0]))
    with pytest.raises(ValueError, match="exactly the batch"):
        analyze(exported, key_path)


def test_source_provenance_and_batch_isolation(workspace, tmp_path):
    client, store, _, headers, rows = workspace
    url = "/annotations/cases/" + rows[0]["review_id"]
    body = payload(client.get(url, headers=headers["alice"]).json())
    body["judgment"]["evidence"] = []
    body["judgment"]["sources"] = [
        {
            "url": "https://example.org/fictional",
            "checked_on": (date.today() + timedelta(days=1)).isoformat(),
            "note": "Fictional source for testing date validation.",
        }
    ]
    assert client.post(url, headers=headers["alice"], json=body).status_code == 422
    body["judgment"]["sources"][0]["checked_on"] = date.today().isoformat()
    body["judgment"]["sources"][0]["url"] = "javascript:alert(1)"
    assert client.post(url, headers=headers["alice"], json=body).status_code == 422
    body["judgment"]["sources"][0]["url"] = "https://example.org/fictional"
    assert client.post(url, headers=headers["alice"], json=body).status_code == 200
    path, second_rows = packet(tmp_path)
    second_rows[0]["left"]["title"] = "A separate fictional packet"
    second_rows[1]["review_id"] = fingerprint("second-batch-only")
    path.write_text("\n".join(json.dumps(row) for row in second_rows))
    second = store.initialize(path, "Other batch", ["alice", "bob"], "carol")
    second_headers = {"Authorization": "Bearer " + second["members"][0]["access_key"]}
    assert client.get(url, headers=second_headers).json()["judgment"]["reason"] == ""
    assert client.post(url, headers=second_headers, json=body).status_code == 409
    assert (
        client.get(
            "/annotations/cases/" + second_rows[1]["review_id"], headers=headers["alice"]
        ).status_code
        == 404
    )


def test_audit_binding_rejects_changed_packets_and_heldout_inputs(tmp_path):
    packet_path, rows = packet(tmp_path)
    audit = tmp_path / "audit.json"
    report = {
        "schema_version": "training-input-audit-v1",
        "complete": True,
        "dataset_name": "fiction",
        "dataset_version": "1",
        "seed": 42,
        "synthetic": True,
        "identity_level": "benchmark_entity",
        "inputs": {"train": "a" * 64},
        "validation_opened": False,
        "test_opened": False,
        "review_sampling": {"selected_pairs": len(rows)},
        "review_artifacts_sha256": {
            "review-blind.jsonl": hashlib.sha256(packet_path.read_bytes()).hexdigest()
        },
    }
    audit.write_text(json.dumps(report))
    database = Database("sqlite:///" + str(tmp_path / "bound.db"))
    store = AnnotationStore(database)
    access = store.initialize(packet_path, "Bound fixture", ["a", "b"], "c", audit)
    provenance = store.export(access["batch_id"])["batch"]["provenance"]
    assert provenance["packet_audit_binding"] == "checksum_verified"
    assert provenance["sampling_seed"] == 42
    packet_path.write_text(packet_path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum differs"):
        store.initialize(packet_path, "Changed", ["a", "b"], "c", audit)
    report["test_opened"] = True
    audit.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="training-only"):
        store.initialize(packet_path, "Heldout", ["a", "b"], "c", audit)
    database.close()


def test_repeated_quotes_have_distinct_unicode_spans(tmp_path):
    path, rows = packet(tmp_path)
    text = "💡 Own model R7; compatible with R7 hubs."
    rows[0]["left"]["title"] = text
    path.write_text("\n".join(json.dumps(row) for row in rows))
    database = Database("sqlite:///" + str(tmp_path / "spans.db"))
    store = AnnotationStore(database)
    access = store.initialize(path, "Spans", ["a", "b"], "c")
    actor = store.authenticate(access["members"][0]["access_key"])
    case_id = rows[0]["review_id"]
    body = payload(store.case(actor, case_id))
    evidence = body["judgment"]["evidence"][0]
    evidence.update(quote="R7", role="compatibility")
    with pytest.raises(ValueError, match="occurs more than once"):
        store.save(actor, case_id, SaveRequest.model_validate(body))
    evidence["start"] = text.rindex("R7") + 1  # UTF-16 offset would be wrong after the emoji.
    with pytest.raises(ValueError, match="quote must occur"):
        store.save(actor, case_id, SaveRequest.model_validate(body))
    evidence["start"] = text.rindex("R7")
    body["judgment"]["evidence"].append(
        {**evidence, "start": text.index("R7"), "role": "own_model"}
    )
    store.save(actor, case_id, SaveRequest.model_validate(body))
    own = store.case(actor, case_id)["judgment"]
    assert own["evidence"][0]["start"] == text.rindex("R7")
    from catalog_identity.annotations.schema import disagreement

    assert not disagreement(own, own)
    database.close()


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="PostgreSQL URL not configured")
def test_postgres_annotation_locking(tmp_path):
    path, rows = packet(tmp_path)
    for row in rows:
        row["review_id"] = fingerprint(["postgres-fiction", uuid4().hex])
    path.write_text("\n".join(json.dumps(row) for row in rows))
    database = Database(os.environ["TEST_DATABASE_URL"])
    other_database = Database(os.environ["TEST_DATABASE_URL"])
    first, second = AnnotationStore(database), AnnotationStore(other_database)
    access = first.initialize(path, "Postgres fixture", ["a", "b"], "c")
    actor = first.authenticate(access["members"][0]["access_key"])
    case_id = rows[0]["review_id"]
    body = payload(first.case(actor, case_id))
    body["submit"] = False

    def save(service):
        try:
            service.save(actor, case_id, SaveRequest.model_validate(body))
            return "saved"
        except Conflict:
            return "stale"

    try:
        with ThreadPoolExecutor(2) as executor:
            assert sorted(executor.map(save, (first, second))) == ["saved", "stale"]
        request = SaveRequest.model_validate(payload(second.case(actor, case_id)))
        second.save(actor, case_id, request)
        assert first.case(actor, case_id)["status"] == "submitted"
        assert len(first.export(access["batch_id"])["history"]) == 3
    finally:
        database.close()
        other_database.close()
