# Local catalog service

Install `uv sync --locked --extra dev --extra models --extra server`. Start PostgreSQL
with `docker compose up -d` when Docker is available. The Compose credentials are local
development defaults and the port is bound to loopback. Keep this service local; authentication,
tenant isolation, deployment hardening, and backups are not implemented.

PowerShell setup (use `export NAME=value` on Linux/macOS):

```powershell
$env:DATABASE_URL = "postgresql+psycopg://catalog:catalog_local@localhost:5432/catalog"
# Optional: load the validation-selected benchmark matcher for pair inspection.
$env:CATALOG_MODEL_DIR = "runs/development/logistic"
uv run catalog-identity seed-showcase
uv run catalog-identity serve
```

The default, when `DATABASE_URL` is unset, is a persistent SQLite file `catalog.db`.
SQLite supports the local demo with serialized transactions; PostgreSQL is the target
for a shared service. Schema version 2 stores normalized attributes and their transformation
history separately from the original listing, linked to its revision and normalization version.
Opening a version 1 database creates that table and backfills it transactionally without
changing source records. Unknown schema versions fail closed before creating new tables.
Future schema changes require a corresponding migration; `create_all` alone is not a migration.

In another terminal with the same environment:

```sh
uv run catalog-identity worker
```

The worker claims database-backed jobs and records success or failure. API calls enqueue
expensive rebuild/evaluation work and return immediately. A worker crash leaves a job in
`running` for operator inspection; automatic lease expiry/retry is not implemented.
`worker --once` processes at most one pending job. Evaluation jobs accept a configuration
filename under the configured directory, not arbitrary filesystem paths. Do not expose
this developer API to untrusted users.

New evaluation requests default to `mode: "validation"`, which trains both matchers and
exports validation diagnostics without opening test data. `feature_version` selects `"2"`
or `"3"`; the worker needs the `models` extra. The dashboard defaults to this workflow.
`mode: "baseline"` explicitly runs the original three-split baseline. Legacy queued jobs
without a mode keep their original baseline behavior. Completed summaries are available from
`GET /evaluations`; full artifacts and local pair evidence remain in the worker's run directory.

API documentation: <http://127.0.0.1:8000/docs>. Endpoints cover listing ingestion/search,
candidate retrieval, pair inspection, family/variant identities, reviews, evaluations and jobs.

The separate [annotation workflow](annotation-workflow.md) adds bearer-authenticated
`GET /annotations/session`, `GET /annotations/cases`, `GET /annotations/cases/{id}` and
`POST /annotations/cases/{id}`. The server derives the reviewer and batch from the access
key. Saves require the case hash, guideline version and current review revision. Import,
key rotation and exports are local operator CLI commands; no administration or private
review-key endpoint is exposed. Annotation tables are additive and do not alter catalog
labels, memberships, model versions or ordinary catalog review decisions.

## Identity and update behavior

Ingestion is idempotent by source ID and payload hash. Changed records increment their
revision, invalidate stored pair/candidate evidence and reviews, and mark identities stale.
Rebuilds reconcile identities against their previous memberships. Unchanged groups keep
their opaque IDs. Merges retain the oldest nominated ID; split children use maximum member
overlap to retain one ID and allocate new IDs to other children. History retains prior IDs.
The current incremental strategy recomputes the small catalog graph; it is not a streaming
100k-listing implementation.

The hierarchy is driven by explicit structured brand/model identity and category-specific
variant rules. Benchmark scores remain separate evidence. They are **not** family/variant
probabilities and never drive automatic hierarchical merges. All showcase products and labels
are original synthetic examples, not a human-annotated real-world evaluation.

Identity policy v2 also compares pack count, region and compatibility when either listing
supplies them. A conflict separates variants; a one-sided value abstains. If neither supplies
these fields, the category's configured requirements apply; this is not proof that every
possible real-world distinction is known. Upgrading the policy atomically marks old pair
evidence and variant acceptances stale and queues a rebuild, preserving human rejections.
API ingestion and review changes commit with their rebuild job in one transaction. Direct
Python callers can request the same behavior with `enqueue_rebuild=True`.

Unknown variant attributes abstain. A review cannot override conflicting or missing
variant-defining evidence; correct the listing first. A rejection is a cannot-link constraint
checked across proposed merged groups, so an indirect path cannot bypass it. Reviews require
the listing revisions the reviewer saw and become stale when either record changes.
A family match leaves an unresolved variant in the review queue. Accepting an exact variant
requires resolving any active family rejection first.

Structured generation is bounded to 500 listings per brand/model block, with an explicit
error beyond that size. Supply a bounded candidate graph for larger blocks. Live lexical
search fits a process-local index on the current catalog and caches it by database revision.
Writes from other API/worker processes invalidate that cache on the next request. A concurrent
change during retrieval rejects the stale result. Benchmark vocabulary still uses training
records only. Cold builds are synchronous and each process owns its cache; this is not a
persistent distributed index. Pair inspection reads only the two requested records, and
review queues exclude scores from other model versions.
