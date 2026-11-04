import hashlib
import json
import secrets
from collections import Counter
from datetime import UTC, datetime

from sqlalchemy import JSON, Column, Integer, MetaData, String, Table, insert, select, update

from catalog_identity.annotations.schema import (
    GUIDELINE_VERSION,
    Draft,
    Judgment,
    PacketCase,
    decision,
    disagreement,
)
from catalog_identity.evaluation.training_audit import fingerprint
from catalog_identity.storage.catalog import Conflict, NotFound

metadata = MetaData()
batches = Table(
    "annotation_batches",
    metadata,
    Column("id", String, primary_key=True),
    Column("data", JSON, nullable=False),
)
members = Table(
    "annotation_members",
    metadata,
    Column("batch_id", String, primary_key=True),
    Column("name", String, primary_key=True),
    Column("role", String, nullable=False),
    Column("token_hash", String, nullable=False, unique=True),
)
cases = Table(
    "annotation_cases",
    metadata,
    Column("batch_id", String, primary_key=True),
    Column("id", String, primary_key=True),
    Column("digest", String, nullable=False),
    Column("data", JSON, nullable=False),
)
reviews = Table(
    "annotation_reviews",
    metadata,
    Column("batch_id", String, primary_key=True),
    Column("case_id", String, primary_key=True),
    Column("reviewer", String, primary_key=True),
    Column("role", String, nullable=False),
    Column("revision", Integer, nullable=False),
    Column("status", String, nullable=False),
    Column("data", JSON, nullable=False),
    Column("updated_at", String, nullable=False),
)
events = Table(
    "annotation_events",
    metadata,
    Column("id", String, primary_key=True),
    Column("batch_id", String, nullable=False),
    Column("data", JSON, nullable=False),
)


def now():
    return datetime.now(UTC).isoformat()


def token_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def where_case(table, batch, case):
    column = table.c.id if table is cases else table.c.case_id
    return (table.c.batch_id == batch) & (column == case)


class AnnotationStore:
    def __init__(self, database):
        self.db = database
        # Additive namespace; the catalog's existing schema and records are unchanged.
        metadata.create_all(database.engine)

    def initialize(self, packet, name, reviewers, adjudicator, audit=None):
        people = [n.strip().casefold() for n in [*reviewers, adjudicator]]
        if (
            len(reviewers) != 2
            or len(set(people)) != 3
            or any(not n or len(n) > 80 for n in people)
        ):
            raise ValueError("assign two different reviewers and a separate adjudicator")
        raw = packet.read_bytes()
        rows = [PacketCase.model_validate_json(line) for line in raw.splitlines() if line.strip()]
        if not rows or len({r.review_id for r in rows}) != len(rows):
            raise ValueError("packet must contain unique, nonempty review cases")
        packet_hash = hashlib.sha256(raw).hexdigest()
        batch_id = fingerprint([packet_hash, GUIDELINE_VERSION])[:24]
        provenance = {"purpose": "training_diagnostic_annotations", "synthetic": None}
        if audit is not None:
            report = json.loads(audit.read_text(encoding="utf-8"))
            if report.get("schema_version") != "training-input-audit-v1" or not report.get(
                "complete"
            ):
                raise ValueError("a completed training input audit is required")
            if report["review_sampling"]["selected_pairs"] != len(rows):
                raise ValueError("packet size differs from audit")
            provenance.update(
                {
                    "audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
                    "training_inputs": report["inputs"],
                    "synthetic": report["synthetic"],
                    "dataset_name": report["dataset_name"],
                }
            )
        credentials = []
        with self.db.transaction() as c:
            if c.execute(select(batches.c.id).where(batches.c.id == batch_id)).first():
                raise Conflict("packet already imported; rotate a lost access key instead")
            c.execute(
                insert(batches).values(
                    id=batch_id,
                    data={
                        "schema_version": 1,
                        "name": name,
                        "created_at": now(),
                        "guideline_version": GUIDELINE_VERSION,
                        "packet_sha256": packet_hash,
                        "case_count": len(rows),
                        "provenance": provenance,
                    },
                )
            )
            for index, person in enumerate(people):
                role = "reviewer" if index < 2 else "adjudicator"
                key = secrets.token_urlsafe(32)
                c.execute(
                    insert(members).values(
                        batch_id=batch_id, name=person, role=role, token_hash=token_hash(key)
                    )
                )
                credentials.append({"name": person, "role": role, "access_key": key})
            for row in rows:
                data = row.model_dump(mode="json", exclude={"judgment"})
                digest = fingerprint(data)
                c.execute(
                    insert(cases).values(
                        batch_id=batch_id, id=row.review_id, digest=digest, data=data
                    )
                )
                for member in credentials:
                    c.execute(
                        insert(reviews).values(
                            batch_id=batch_id,
                            case_id=row.review_id,
                            reviewer=member["name"],
                            role=member["role"],
                            revision=0,
                            status="pending",
                            data={},
                            updated_at=now(),
                        )
                    )
            self._event(
                c,
                batch_id,
                {
                    "action": "import",
                    "packet_sha256": packet_hash,
                    "case_count": len(rows),
                    "members": people,
                },
            )
        return {"batch_id": batch_id, "members": credentials}

    def _event(self, c, batch_id, data):
        c.execute(
            insert(events).values(
                id=secrets.token_hex(16), batch_id=batch_id, data={"at": now(), **data}
            )
        )

    def rotate(self, batch_id, name):
        name = name.strip().casefold()
        key = secrets.token_urlsafe(32)
        with self.db.transaction() as c:
            result = c.execute(
                update(members)
                .where((members.c.batch_id == batch_id) & (members.c.name == name))
                .values(token_hash=token_hash(key))
            )
            if not result.rowcount:
                raise NotFound("unknown batch member")
            self._event(c, batch_id, {"action": "key_rotated", "reviewer": name})
        return {"batch_id": batch_id, "name": name, "access_key": key}

    def authenticate(self, key):
        with self.db.transaction() as c:
            row = (
                c.execute(select(members).where(members.c.token_hash == token_hash(key)))
                .mappings()
                .first()
            )
            if not row:
                return None
            batch = c.execute(
                select(batches.c.data).where(batches.c.id == row["batch_id"])
            ).scalar_one()
            if batch["schema_version"] != 1 or batch["guideline_version"] != GUIDELINE_VERSION:
                raise ValueError("unsupported annotation batch version")
            return {
                "batch_id": row["batch_id"],
                "name": row["name"],
                "role": row["role"],
                "batch": batch,
            }

    def _reviews(self, c, batch_id, case_id):
        return (
            c.execute(select(reviews).where(where_case(reviews, batch_id, case_id)))
            .mappings()
            .all()
        )

    def _state(self, rows):
        independent = [r for r in rows if r["role"] == "reviewer" and r["status"] == "submitted"]
        adjudicated = next(
            (r for r in rows if r["role"] == "adjudicator" and r["status"] == "submitted"), None
        )
        if len(independent) < 2:
            return "awaiting_reviews", independent, None
        if adjudicated:
            return "adjudicated", independent, adjudicated["data"]
        if disagreement(independent[0]["data"], independent[1]["data"]):
            return "needs_adjudication", independent, None
        return "agreed", independent, independent[0]["data"]

    def queue(self, actor):
        result = []
        with self.db.transaction() as c:
            all_reviews = (
                c.execute(select(reviews).where(reviews.c.batch_id == actor["batch_id"]))
                .mappings()
                .all()
            )
            grouped = {}
            for row in all_reviews:
                grouped.setdefault(row["case_id"], []).append(row)
            for case in c.execute(
                select(cases).where(cases.c.batch_id == actor["batch_id"]).order_by(cases.c.id)
            ).mappings():
                rows = grouped[case["id"]]
                own = next(r for r in rows if r["reviewer"] == actor["name"])
                state, _, _ = self._state(rows)
                if actor["role"] == "adjudicator" and state == "awaiting_reviews":
                    continue
                result.append(
                    {
                        "id": case["id"],
                        "left_title": case["data"]["left"]["title"],
                        "right_title": case["data"]["right"]["title"],
                        "status": own["status"],
                        "resolution": state if actor["role"] == "adjudicator" else None,
                    }
                )
        return result

    def case(self, actor, case_id):
        with self.db.transaction() as c:
            row = (
                c.execute(select(cases).where(where_case(cases, actor["batch_id"], case_id)))
                .mappings()
                .first()
            )
            if not row:
                raise NotFound("case is not assigned to this batch")
            rows = self._reviews(c, actor["batch_id"], case_id)
            state, independent, _ = self._state(rows)
            if actor["role"] == "adjudicator" and state == "awaiting_reviews":
                raise Conflict("both independent reviews must be locked before adjudication")
            own = next(r for r in rows if r["reviewer"] == actor["name"])
            result = {
                **row["data"],
                "case_hash": row["digest"],
                "guideline_version": GUIDELINE_VERSION,
                "revision": own["revision"],
                "status": own["status"],
                "judgment": own["data"] or Draft().model_dump(mode="json"),
            }
            if actor["role"] == "adjudicator":
                result.update(
                    resolution=state,
                    independent_reviews=[
                        {"reviewer": r["reviewer"], "judgment": r["data"]} for r in independent
                    ],
                )
            return result

    def save(self, actor, case_id, request):
        judgment = (Judgment if request.submit else Draft).model_validate(
            request.judgment.model_dump()
        )
        data = judgment.model_dump(mode="json")
        with self.db.transaction() as c:
            case = (
                c.execute(select(cases).where(where_case(cases, actor["batch_id"], case_id)))
                .mappings()
                .first()
            )
            if not case:
                raise NotFound("case is not assigned to this batch")
            if (
                case["digest"] != request.case_hash
                or request.guideline_version != GUIDELINE_VERSION
            ):
                raise Conflict("case or guidelines changed; reload before saving")
            for item in judgment.evidence:
                text = case["data"][item.side].get(item.field) or ""
                if item.quote not in text:
                    raise ValueError("evidence quote must occur in the selected listing field")
            rows = self._reviews(c, actor["batch_id"], case_id)
            state, _, _ = self._state(rows)
            if actor["role"] == "adjudicator" and state == "awaiting_reviews":
                raise Conflict("both independent reviews must be locked before adjudication")
            own = next(r for r in rows if r["reviewer"] == actor["name"])
            if own["status"] == "submitted":
                if request.submit and data == own["data"]:
                    return {"revision": own["revision"], "status": "submitted"}
                raise Conflict("submitted judgments are immutable")
            if request.revision != own["revision"]:
                raise Conflict("a newer draft exists; reload before saving")
            revision = own["revision"] + 1
            status = "submitted" if request.submit else "draft"
            c.execute(
                update(reviews)
                .where(
                    where_case(reviews, actor["batch_id"], case_id)
                    & (reviews.c.reviewer == actor["name"])
                )
                .values(revision=revision, status=status, data=data, updated_at=now())
            )
            self._event(
                c,
                actor["batch_id"],
                {
                    "action": status,
                    "case_id": case_id,
                    "case_hash": case["digest"],
                    "reviewer": actor["name"],
                    "role": actor["role"],
                    "revision": revision,
                    "guideline_version": GUIDELINE_VERSION,
                    "judgment": data,
                },
            )
        return {"revision": revision, "status": status}

    def export(self, batch_id):
        with self.db.transaction() as c:
            batch = c.execute(
                select(batches.c.data).where(batches.c.id == batch_id)
            ).scalar_one_or_none()
            if batch is None:
                raise NotFound("unknown annotation batch")
            output, counts, sufficient = [], Counter(), Counter()
            both, disagreements = 0, 0
            for case in c.execute(
                select(cases).where(cases.c.batch_id == batch_id).order_by(cases.c.id)
            ).mappings():
                rows = self._reviews(c, batch_id, case["id"])
                status, independent, resolved = self._state(rows)
                counts[status] += 1
                if len(independent) == 2:
                    both += 1
                    disagreements += int(
                        disagreement(independent[0]["data"], independent[1]["data"])
                    )
                if resolved:
                    sufficient[resolved["input_sufficiency"]] += 1
                output.append(
                    {
                        "case": case["data"],
                        "case_hash": case["digest"],
                        "status": status,
                        "resolved_decisions": decision(resolved) if resolved else None,
                        "submissions": [
                            {
                                "reviewer": r["reviewer"],
                                "role": r["role"],
                                "revision": r["revision"],
                                "submitted_at": r["updated_at"],
                                "judgment": r["data"],
                            }
                            for r in rows
                            if r["status"] == "submitted"
                        ],
                        "automatically_training_eligible": False,
                    }
                )
            history = (
                c.execute(select(events.c.data).where(events.c.batch_id == batch_id))
                .scalars()
                .all()
            )
        return {
            "schema_version": 1,
            "batch_id": batch_id,
            "batch": batch,
            "exported_at": now(),
            "summary": {
                "cases": len(output),
                "status_counts": dict(counts),
                "both_reviewed": both,
                "independent_disagreements": disagreements,
                "disagreement_rate": disagreements / both if both else None,
                "resolved_sufficiency_counts": dict(sufficient),
            },
            "cases": output,
            "history": sorted(history, key=lambda e: (e["at"], e["action"])),
            "official_labels_modified": False,
            "warning": "Diagnostic training cases, not independent evaluation data. "
            "Consensus is not automatically ground truth. Preserve uncertain decisions.",
        }
