from contextlib import contextmanager
from threading import RLock

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
)
from sqlalchemy.pool import StaticPool

metadata = MetaData()
listings = Table(
    "listings",
    metadata,
    Column("id", String, primary_key=True),
    Column("source", String, nullable=False),
    Column("source_id", String, nullable=False),
    Column("title_search", String, nullable=False),
    Column("revision", Integer, nullable=False),
    Column("digest", String, nullable=False),
    Column("data", JSON, nullable=False),
)
state = Table(
    "catalog_state",
    metadata,
    Column("key", String, primary_key=True),
    Column("data", JSON, nullable=False),
)
pairs = Table(
    "pair_scores",
    metadata,
    Column("id", String, primary_key=True),
    Column("left", String, nullable=False, index=True),
    Column("right", String, nullable=False, index=True),
    Column("left_revision", Integer, nullable=False),
    Column("right_revision", Integer, nullable=False),
    Column("model_id", String, nullable=False),
    Column("probability", Float),
    Column("stale", Boolean, nullable=False),
    Column("data", JSON, nullable=False),
)
candidates = Table(
    "candidate_evidence",
    metadata,
    Column("query_id", String, primary_key=True),
    Column("candidate_id", String, primary_key=True),
    Column("catalog_revision", Integer, nullable=False),
    Column("data", JSON, nullable=False),
)
clusters = Table(
    "clusters",
    metadata,
    Column("id", String, primary_key=True),
    Column("level", String, nullable=False),
    Column("parent_id", String),
    Column("members", JSON, nullable=False),
)
memberships = Table(
    "memberships",
    metadata,
    Column("listing_id", String, primary_key=True),
    Column("level", String, primary_key=True),
    Column("cluster_id", String, nullable=False),
)
history = Table(
    "identity_history",
    metadata,
    Column("id", String, primary_key=True),
    Column("catalog_revision", Integer, nullable=False),
    Column("data", JSON, nullable=False),
)
reviews = Table(
    "reviews",
    metadata,
    Column("id", String, primary_key=True),
    Column("left", String, nullable=False),
    Column("right", String, nullable=False),
    Column("left_revision", Integer, nullable=False),
    Column("right_revision", Integer, nullable=False),
    Column("level", String, nullable=False),
    Column("decision", String, nullable=False),
    Column("note", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("stale", Boolean, nullable=False),
)
jobs = Table(
    "jobs",
    metadata,
    Column("id", String, primary_key=True),
    Column("kind", String, nullable=False),
    Column("status", String, nullable=False),
    Column("idempotency_key", String, unique=True),
    Column("created_at", String, nullable=False),
    Column("payload", JSON, nullable=False),
    Column("result", JSON),
    Column("error", String),
)
models = Table(
    "model_versions",
    metadata,
    Column("id", String, primary_key=True),
    Column("data", JSON, nullable=False),
)
evaluations = Table(
    "evaluations",
    metadata,
    Column("id", String, primary_key=True),
    Column("created_at", String, nullable=False),
    Column("data", JSON, nullable=False),
)


class Database:
    def __init__(self, url="sqlite:///catalog.db"):
        options = {}
        if url.startswith("sqlite"):
            options["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                options["poolclass"] = StaticPool
        self.engine = create_engine(url, **options)
        self.lock = RLock()
        metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            current = (
                connection.execute(select(state).where(state.c.key == "catalog")).mappings().first()
            )
            if current is None:
                connection.execute(
                    insert(state).values(
                        key="catalog",
                        data={
                            "schema_version": 1,
                            "revision": 0,
                            "identities_stale": False,
                            "family_registry": None,
                            "variant_registry": None,
                            "graph": {},
                        },
                    )
                )
            elif current["data"].get("schema_version") != 1:
                raise ValueError("unsupported database schema version; migrate before opening")

    @contextmanager
    def transaction(self):
        # PostgreSQL serializes catalog writes across API and worker processes.
        # SQLite is a single-process development/test fallback.
        with self.lock, self.engine.connect() as connection:
            if self.engine.dialect.name == "sqlite":
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                connection.begin()
                connection.execute(
                    select(state).where(state.c.key == "catalog").with_for_update()
                ).first()
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def close(self):
        self.engine.dispose()
