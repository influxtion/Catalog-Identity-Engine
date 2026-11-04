# Catalog dashboard

Next.js, TypeScript and Tailwind/PostCSS. The dashboard uses the local API through a
same-origin server proxy, so database credentials never enter the browser.

Start the backend and worker from the repository root first; see
[the service instructions](../docs/service.md). Seed the original synthetic showcase
with `uv run catalog-identity seed-showcase`.

From this directory:

```sh
npm ci
npm run dev
```

Open <http://127.0.0.1:3000>. The default API is <http://127.0.0.1:8000>.
Set the server environment variable `CATALOG_API_URL` to change it.

Features: listing search and source provenance, family/variant membership, ranked
candidates, side-by-side inspection, revision-aware reviews, accepted/rejected graph
links, identity history, measured benchmark results, and queued evaluation jobs.
The evaluation screen separates recorded test results from validation-only development
comparisons. Validation jobs select an official development split and feature version, leave
test files unopened, and display their completed summaries. The retrieval cost table includes
the measured 100k synthetic run; it is not a full-service throughput claim.
The explorer loads up to 200 records and filters that loaded view. Large-catalog
pagination and graph virtualization are future work. No real product images are used.

`/annotate` provides a separate blinded evidence workspace with assignment keys, draft
resumption, exact quote locations, source checks, immutable submissions and third-person
adjudication. Follow the [annotation guide](../docs/annotation-workflow.md) to import the
145-case packet and collect independent reviews. The same-origin proxy forwards bearer
credentials only to annotation routes. The private review key is never served to the UI.

## Checks

```sh
npm run build
npm run typecheck
npx playwright install chromium
npm run test:e2e
```

End-to-end tests start an isolated temporary SQLite catalog and a production Next.js
server on ports 8011 and 3011. They do not modify the normal development database.
They exercise search, pair review, graph/evaluation navigation, invalid/valid ingestion,
mobile layout, independent annotation/adjudication, and repeated-code span selection.
Chromium screenshots are generated under ignored `test-results/`. Annotation tests use
fictional cases and temporary access keys under ignored `runs/annotation-e2e/`.
The API and worker are required for queued rebuilds; the UI explicitly marks old
identity memberships as stale while a rebuild is pending.
