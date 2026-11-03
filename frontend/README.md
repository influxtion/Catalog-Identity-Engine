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
The explorer loads up to 200 records and filters that loaded view. Large-catalog
pagination and graph virtualization are future work. No real product images are used.

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
and mobile layout. Chromium screenshots are generated under ignored `test-results/`.
The API and worker are required for queued rebuilds; the UI explicitly marks old
identity memberships as stale while a rebuild is pending.
