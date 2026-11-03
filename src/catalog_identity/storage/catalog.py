import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import delete, insert, or_, select, update

from catalog_identity.clustering.engine import Edge, cluster, structured_edges
from catalog_identity.clustering.policy import compare_identity
from catalog_identity.clustering.registry import IdentityRegistry
from catalog_identity.matching.enriched import ENRICHED_NAMES
from catalog_identity.matching.features import pair_features
from catalog_identity.normalization import normalize, normalize_text
from catalog_identity.retrieval import CandidateIndex
from catalog_identity.schema import Listing
from catalog_identity.storage import database as tables


class NotFound(ValueError):
    pass


class Conflict(ValueError):
    pass


def timestamp():
    return datetime.now(UTC).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


class Catalog:
    def __init__(self, database, model_directory: Path | None = None):
        self.db = database
        self.model = None
        self.model_id = "structured-evidence-v1"
        self.policy = None
        if model_directory:
            from catalog_identity.evaluation.runner import file_hash
            from catalog_identity.matching.learned import LearnedMatcher

            self.model = LearnedMatcher.load(model_directory)
            self.policy = json.loads((model_directory / "policy.json").read_text(encoding="utf-8"))
            self.model_id = digest(
                {p.name: file_hash(p) for p in sorted(model_directory.iterdir()) if p.is_file()}
            )
        with self.db.transaction() as connection:
            if (
                connection.execute(
                    select(tables.models).where(tables.models.c.id == self.model_id)
                ).first()
                is None
            ):
                connection.execute(
                    insert(tables.models).values(
                        id=self.model_id,
                        data={
                            "kind": "learned_benchmark_entity"
                            if self.model
                            else "structured_rules",
                            "policy": self.policy,
                        },
                    )
                )

    def _state(self, connection):
        return dict(
            connection.execute(
                select(tables.state.c.data).where(tables.state.c.key == "catalog")
            ).scalar_one()
        )

    def _save_state(self, connection, state):
        connection.execute(
            update(tables.state).where(tables.state.c.key == "catalog").values(data=state)
        )

    def ingest(self, records):
        records = [
            Listing.model_validate(record) if isinstance(record, dict) else record
            for record in records
        ]
        if len({x.key for x in records}) != len(records):
            raise ValueError("duplicate listing in batch")
        changed = []
        with self.db.transaction() as connection:
            state = self._state(connection)
            for record in records:
                normalized = normalize(record)
                payload = record.model_dump(mode="json")
                fingerprint = digest(payload)
                existing = (
                    connection.execute(
                        select(tables.listings).where(tables.listings.c.id == record.key)
                    )
                    .mappings()
                    .first()
                )
                if existing and existing["digest"] == fingerprint:
                    continue
                values = dict(
                    source=record.source,
                    source_id=record.source_id,
                    title_search=normalized.title,
                    revision=existing["revision"] + 1 if existing else 1,
                    digest=fingerprint,
                    data=payload,
                )
                if existing:
                    connection.execute(
                        update(tables.listings)
                        .where(tables.listings.c.id == record.key)
                        .values(**values)
                    )
                else:
                    connection.execute(insert(tables.listings).values(id=record.key, **values))
                changed.append(record.key)
            if changed:
                for table in (tables.pairs, tables.reviews):
                    connection.execute(
                        update(table)
                        .where(or_(table.c.left.in_(changed), table.c.right.in_(changed)))
                        .values(stale=True)
                    )
                connection.execute(
                    delete(tables.candidates).where(
                        or_(
                            tables.candidates.c.query_id.in_(changed),
                            tables.candidates.c.candidate_id.in_(changed),
                        )
                    )
                )
                state.update(revision=state["revision"] + 1, identities_stale=True)
                self._save_state(connection, state)
            return {
                "changed": len(changed),
                "unchanged": len(records) - len(changed),
                "listing_ids": [x.key for x in records],
                "catalog_revision": state["revision"],
            }

    def listing(self, key):
        with self.db.transaction() as connection:
            row = (
                connection.execute(select(tables.listings).where(tables.listings.c.id == key))
                .mappings()
                .first()
            )
            if row is None:
                raise NotFound("listing not found")
            membership = (
                connection.execute(
                    select(tables.memberships).where(tables.memberships.c.listing_id == key)
                )
                .mappings()
                .all()
            )
            return {
                **dict(row),
                "identities": [dict(x) for x in membership],
                "identities_stale": self._state(connection)["identities_stale"],
            }

    def search(self, query="", limit=50, offset=0):
        with self.db.transaction() as connection:
            statement = select(tables.listings)
            if query:
                statement = statement.where(
                    tables.listings.c.title_search.contains(normalize_text(query), autoescape=True)
                )
            rows = (
                connection.execute(
                    statement.order_by(tables.listings.c.id).limit(limit).offset(offset)
                )
                .mappings()
                .all()
            )
            return [dict(row) for row in rows]

    def snapshot(self):
        with self.db.transaction() as connection:
            rows = connection.execute(select(tables.listings)).mappings().all()
            return (
                self._state(connection),
                {row["id"]: Listing.model_validate(row["data"]) for row in rows},
                {row["id"]: row["revision"] for row in rows},
            )

    def retrieve(self, key, k=20):
        state, listings, _ = self.snapshot()
        if key not in listings:
            raise NotFound("listing not found")
        normalized = {key: normalize(value) for key, value in listings.items()}
        index = CandidateIndex().fit(list(normalized.values())).build(list(normalized.values()))
        result = [asdict(c) for c in index.query(normalized[key], k)]
        with self.db.transaction() as connection:
            if self._state(connection)["revision"] != state["revision"]:
                raise Conflict("catalog changed during retrieval; retry")
            connection.execute(delete(tables.candidates).where(tables.candidates.c.query_id == key))
            for candidate in result:
                connection.execute(
                    insert(tables.candidates).values(
                        query_id=key,
                        candidate_id=candidate["listing_id"],
                        catalog_revision=state["revision"],
                        data=candidate,
                    )
                )
        return result

    def inspect_pair(self, left, right):
        if left == right:
            raise ValueError("choose two distinct listings")
        left, right = sorted((left, right))
        _, listings, revisions = self.snapshot()
        if left not in listings or right not in listings:
            raise NotFound("listing not found")
        a, b = normalize(listings[left]), normalize(listings[right])
        hierarchy = asdict(compare_identity(a, b))
        probability, model_decision = None, "not_scored"
        features = pair_features(a, b)
        if self.model:
            x = self.model.features.transform(
                {left: listings[left], right: listings[right]}, [(left, right)]
            )
            features = dict(zip(ENRICHED_NAMES, x[0].tolist(), strict=True))
            probability = float(self.model.score_features(x)[0])
            match, reject = self.policy["match"]["threshold"], self.policy["reject_threshold"]
            model_decision = (
                "match"
                if match is not None and probability >= match
                else "reject"
                if reject is not None and probability <= reject
                else "review"
            )
        result = {
            "id": digest([left, right, self.model_id]),
            "left": left,
            "right": right,
            "left_revision": revisions[left],
            "right_revision": revisions[right],
            "model_id": self.model_id,
            "probability": probability,
            "model_identity_level": "benchmark_entity" if self.model else None,
            "model_decision": model_decision,
            "hierarchy": hierarchy,
            "features": features,
            "needs_review": hierarchy["relation"] == "uncertain" or model_decision == "review",
        }
        with self.db.transaction() as connection:
            for key in (left, right):
                current = connection.execute(
                    select(tables.listings.c.revision).where(tables.listings.c.id == key)
                ).scalar_one()
                if current != revisions[key]:
                    raise Conflict("listing changed during scoring; retry")
            connection.execute(delete(tables.pairs).where(tables.pairs.c.id == result["id"]))
            connection.execute(
                insert(tables.pairs).values(
                    **{
                        key: result[key]
                        for key in (
                            "id",
                            "left",
                            "right",
                            "left_revision",
                            "right_revision",
                            "model_id",
                            "probability",
                        )
                    },
                    stale=False,
                    data=result,
                )
            )
        return result

    def submit_review(self, left, right, level, decision, note, left_revision, right_revision):
        if (
            left == right
            or level not in ("family", "variant")
            or decision not in ("match", "reject", "uncertain")
        ):
            raise ValueError("invalid review")
        if len(note.strip()) < 3:
            raise ValueError("review note must explain the decision")
        with self.db.transaction() as connection:
            observed = {}
            for key, revision in ((left, left_revision), (right, right_revision)):
                row = (
                    connection.execute(select(tables.listings).where(tables.listings.c.id == key))
                    .mappings()
                    .first()
                )
                if row is None:
                    raise NotFound("listing not found")
                if row["revision"] != revision:
                    raise Conflict("review refers to an outdated listing revision")
                observed[key] = normalize(Listing.model_validate(row["data"]))
            evidence = compare_identity(observed[left], observed[right])
            if decision == "match" and (
                getattr(evidence, level) == "different"
                or (level == "variant" and evidence.variant != "same")
            ):
                raise Conflict(
                    "resolve conflicting or missing defining attributes before accepting this merge"
                )
            pair_keys = or_(
                (tables.reviews.c.left == left) & (tables.reviews.c.right == right),
                (tables.reviews.c.left == right) & (tables.reviews.c.right == left),
            )
            connection.execute(
                update(tables.reviews)
                .where(pair_keys & (tables.reviews.c.level == level))
                .values(stale=True)
            )
            review = dict(
                id=str(uuid4()),
                left=left,
                right=right,
                level=level,
                decision=decision,
                note=note.strip(),
                left_revision=left_revision,
                right_revision=right_revision,
                created_at=timestamp(),
                stale=False,
            )
            connection.execute(insert(tables.reviews).values(**review))
            state = self._state(connection)
            state.update(revision=state["revision"] + 1, identities_stale=True)
            self._save_state(connection, state)
            return review

    def rebuild(self):
        with self.db.transaction() as connection:
            state = self._state(connection)
            rows = connection.execute(select(tables.listings)).mappings().all()
            listings = {row["id"]: normalize(Listing.model_validate(row["data"])) for row in rows}
            family_edges, variant_edges = structured_edges(listings)
            active_reviews = (
                connection.execute(select(tables.reviews).where(tables.reviews.c.stale.is_(False)))
                .mappings()
                .all()
            )
            by_level = {
                "family": {tuple(sorted((e.left, e.right))): e for e in family_edges},
                "variant": {tuple(sorted((e.left, e.right))): e for e in variant_edges},
            }
            forbidden = {"family": set(), "variant": set()}
            for review in active_reviews:
                pair = tuple(sorted((review["left"], review["right"])))
                level = review["level"]
                if review["decision"] == "reject":
                    forbidden[level].add(pair)
                    if level == "family":
                        forbidden["variant"].add(pair)
                elif review["decision"] == "match":
                    by_level[level][pair] = Edge(*pair, 1.0)
            family = cluster(
                listings, by_level["family"].values(), "family", forbidden_pairs=forbidden["family"]
            )
            owner = {key: i for i, group in enumerate(family.clusters) for key in group}
            variant = cluster(
                listings,
                [e for e in by_level["variant"].values() if owner[e.left] == owner[e.right]],
                "variant",
                forbidden_pairs=forbidden["variant"],
            )
            registries = {}
            for level, result in (("family", family), ("variant", variant)):
                saved = state[f"{level}_registry"]
                registry = IdentityRegistry.from_dict(saved) if saved else IdentityRegistry()
                old_history = len(registry.history)
                registry.reconcile(result.clusters)
                for event in registry.history[old_history:]:
                    connection.execute(
                        insert(tables.history).values(
                            id=str(uuid4()),
                            catalog_revision=state["revision"],
                            data={"level": level, **event},
                        )
                    )
                state[f"{level}_registry"] = registry.to_dict()
                registries[level] = registry
            connection.execute(delete(tables.memberships))
            connection.execute(delete(tables.clusters))
            family_owner = {
                key: identity
                for identity, group in registries["family"].active.items()
                for key in group
            }
            for level, registry in registries.items():
                for identity, group in registry.active.items():
                    connection.execute(
                        insert(tables.clusters).values(
                            id=identity,
                            level=level,
                            parent_id=family_owner[group[0]] if level == "variant" else None,
                            members=list(group),
                        )
                    )
                    for key in group:
                        connection.execute(
                            insert(tables.memberships).values(
                                listing_id=key, level=level, cluster_id=identity
                            )
                        )
            state["identities_stale"] = False
            state["graph"] = {
                "accepted": [
                    {"level": level, **asdict(edge)}
                    for level, result in (("family", family), ("variant", variant))
                    for edge in result.accepted
                ],
                "rejected": [
                    {"level": level, **edge}
                    for level, result in (("family", family), ("variant", variant))
                    for edge in result.rejected
                ],
            }
            self._save_state(connection, state)
            return {
                "families": len(family.clusters),
                "variants": len(variant.clusters),
                "catalog_revision": state["revision"],
            }

    def identities(self, level="family"):
        with self.db.transaction() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    select(tables.clusters)
                    .where(tables.clusters.c.level == level)
                    .order_by(tables.clusters.c.id)
                ).mappings()
            ]

    def identity(self, key):
        with self.db.transaction() as connection:
            row = (
                connection.execute(select(tables.clusters).where(tables.clusters.c.id == key))
                .mappings()
                .first()
            )
            if row is None:
                raise NotFound("active identity not found; inspect history for retired identities")
            return {**dict(row), "stale": self._state(connection)["identities_stale"]}

    def overview(self):
        with self.db.transaction() as connection:
            state = self._state(connection)
            return {
                "catalog_revision": state["revision"],
                "identities_stale": state["identities_stale"],
                "graph": state["graph"],
                "model_id": self.model_id,
                "model_loaded": self.model is not None,
                "history": [row[0] for row in connection.execute(select(tables.history.c.data))],
            }

    def review_queue(self):
        with self.db.transaction() as connection:
            current_reviews = (
                connection.execute(select(tables.reviews).where(tables.reviews.c.stale.is_(False)))
                .mappings()
                .all()
            )
            resolved = {
                tuple(sorted((r["left"], r["right"])))
                for r in current_reviews
                if r["decision"] != "uncertain"
            }
            scored = (
                connection.execute(select(tables.pairs).where(tables.pairs.c.stale.is_(False)))
                .mappings()
                .all()
            )
            return [
                row["data"]
                for row in scored
                if row["data"]["needs_review"]
                and tuple(sorted((row["left"], row["right"]))) not in resolved
            ]

    def enqueue(self, kind, payload=None, idempotency_key=None):
        if kind not in ("rebuild", "evaluate"):
            raise ValueError("unknown job kind")
        payload = payload or {}
        with self.db.transaction() as connection:
            if idempotency_key:
                existing = (
                    connection.execute(
                        select(tables.jobs).where(tables.jobs.c.idempotency_key == idempotency_key)
                    )
                    .mappings()
                    .first()
                )
                if existing:
                    if existing["kind"] != kind or existing["payload"] != payload:
                        raise Conflict("idempotency key already used for different work")
                    return dict(existing)
            job = dict(
                id=str(uuid4()),
                kind=kind,
                payload=payload,
                status="pending",
                idempotency_key=idempotency_key,
                created_at=timestamp(),
                result=None,
                error=None,
            )
            connection.execute(insert(tables.jobs).values(**job))
            return job

    def claim_job(self):
        with self.db.transaction() as connection:
            row = (
                connection.execute(
                    select(tables.jobs)
                    .where(tables.jobs.c.status == "pending")
                    .order_by(tables.jobs.c.created_at, tables.jobs.c.id)
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            connection.execute(
                update(tables.jobs).where(tables.jobs.c.id == row["id"]).values(status="running")
            )
            return {**dict(row), "status": "running"}

    def finish_job(self, key, result=None, error=None):
        with self.db.transaction() as connection:
            job = (
                connection.execute(select(tables.jobs).where(tables.jobs.c.id == key))
                .mappings()
                .first()
            )
            if job is None or job["status"] != "running":
                raise Conflict("only a running job can finish")
            connection.execute(
                update(tables.jobs)
                .where(tables.jobs.c.id == key)
                .values(status="failed" if error else "succeeded", result=result, error=error)
            )
            if not error and job["kind"] == "evaluate":
                connection.execute(
                    insert(tables.evaluations).values(id=key, created_at=timestamp(), data=result)
                )

    def jobs(self):
        with self.db.transaction() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    select(tables.jobs).order_by(tables.jobs.c.created_at.desc()).limit(100)
                ).mappings()
            ]

    def evaluation_reports(self):
        with self.db.transaction() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    select(tables.evaluations)
                    .order_by(tables.evaluations.c.created_at.desc())
                    .limit(20)
                ).mappings()
            ]
