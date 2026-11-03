# Local catalog service

Install `uv sync --locked --extra dev --extra models --extra server`. Start PostgreSQL
with `docker compose up -d` when Docker is available. The Compose credentials are local
development defaults and the port is bound to loopback. Keep this service local; authentication,
tenant isolation, deployment hardening, and backups are not implemented.

PowerShell setup (use `export NAME=value` on Linux/macOS):

```powershell
$env:DATABASE_URL = "postgresql+psycopg://catalog:catalog_local@localhost:5432/catalog"
# Optional: load the validation-selected benchmark matcher for pair inspection.
$env:CATALOG_MODEL_DIR = "runs/development-v3/logistic"
uv run catalog-identity seed-showcase
uv run catalog-identity serve
```

The default, when `DATABASE_URL` is unset, is a persistent SQLite file `catalog.db`.
SQLite supports the local demo with serialized transactions; PostgreSQL is the target
for a shared service. Schema version 1 is created on an empty database. Unknown schema
versions fail closed. Future schema changes require an explicit migration; `create_all`
does not migrate existing tables.

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

API documentation: <http://127.0.0.1:8000/docs>. Endpoints cover listing ingestion/search,
candidate retrieval, pair inspection, family/variant identities, reviews, evaluations and jobs.

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

Unknown variant attributes abstain. A review cannot override conflicting or missing
variant-defining evidence; correct the listing first. A rejection is a cannot-link constraint
checked across proposed merged groups, so an indirect path cannot bypass it. Reviews require
the listing revisions the reviewer saw and become stale when either record changes.

Structured generation is bounded to 500 listings per brand/model block, with an explicit
error beyond that size. Supply a bounded candidate graph for larger blocks. Live lexical
search currently rebuilds its vocabulary from the current catalog per request; benchmark
evaluation continues to fit vocabulary only on training records. This live route is a
correctness prototype, not a low-latency scale service.
