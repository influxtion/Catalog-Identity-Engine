# Catalog Identity Engine

Resolve product listings across stores, with reproducible experiments and inspectable evidence.
Independent project inspired by public catalog-matching research; not affiliated with Shopify.

The local research system includes typed ingestion, lexical and semantic retrieval, calibrated
logistic/LightGBM experiments, constrained identities, persistence, a FastAPI service, and a
Next.js inspection dashboard. Its measured matcher accuracy does not yet support the intended
high-precision automatic matching.
Product-family and exact-variant identity are distinct concepts. WDC's binary entity labels must
not be treated as hierarchical family/variant annotations.

## Development

Python 3.11 or newer. Using [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked --extra dev
uv run catalog-identity demo --output runs/demo
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Alternatively, create a virtual environment and run `python -m pip install -e ".[dev]"`.
The committed lockfile is the reproducible dependency specification for uv installations.

On Windows without uv, the equivalent commands are:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m catalog_identity.cli demo --output runs/demo
.venv\Scripts\python -m pytest
```

Choose a fresh output directory for every run. Existing runs are never overwritten.
The demo generates fictional products with entity-disjoint splits. Its metrics verify
the pipeline; **they are not real benchmark results**.

## Run WDC Products locally

1. Review [the data contract and licensing status](docs/datasets.md). Obtain the pairwise
   archive from the official WDC Products page and keep it under `data/`. For the example,
   extract `80pair.zip` into `data/wdc-products-80/`.
2. Extract one complete official train/validation/test configuration. Do not reshuffle rows.
3. Copy `configs/wdc-products.example.json` and set the three file paths and protocol fields
   to match your chosen files. Paths resolve relative to the configuration file.
4. Run:

```sh
uv run catalog-identity evaluate --config configs/wdc-products.example.json --output runs/wdc-baseline
```

The example uses 80% corner cases, small development size, and 100% unseen test entities.
Its paths match the original archive filenames. The loader enforces disjoint offer
IDs; the 100% unseen configuration additionally requires a disjoint test entity set.

## Improve and evaluate the matcher

```sh
uv sync --locked --extra dev --extra models --extra semantic --extra server --extra benchmarks
uv run catalog-identity develop --config configs/wdc-products.example.json --output runs/development
uv run catalog-identity retrieval-ablation --config configs/wdc-products.example.json --output runs/retrieval --allow-model-download
# Freeze the model and retrieval choice using validation before opening test.
uv run catalog-identity evaluate-frozen --config configs/wdc-products.example.json --model runs/development/logistic --hybrid --output runs/frozen
```

`develop` compares enriched logistic regression and LightGBM without reading test data. It
partitions validation entities between calibration and operating-threshold selection. The
semantic command explicitly permits a pinned MiniLM weight download; subsequent runs use the
local cache. No product images are downloaded. Semantic dependencies include PyTorch and are
optional for the service and lexical baseline.

See [the development protocol](docs/development.md) for partition weighting, model selection,
negative-pair ablations, and the limits of these scores.

## Run the local dashboard

Install the `server` extra above, then use three terminals from the repository root:

```sh
# First terminal; defaults to catalog.db with original fictional records.
uv run catalog-identity seed-showcase
uv run catalog-identity serve
```

```sh
# Second terminal; processes rebuilds and evaluation jobs.
uv run catalog-identity worker
```

```sh
# Third terminal; Node 24 and npm are used in CI.
cd frontend
npm ci
npm run dev
```

Open [the dashboard](http://localhost:3000) and [API documentation](http://127.0.0.1:8000/docs).
Search listings or identity IDs, select a family or variant group, inspect pair evidence,
record reviews, and view graph links and measured evaluation costs. The default service uses
structured identity rules and shows no invented model probability. To load the trained
benchmark matcher or use PostgreSQL, follow [service setup](docs/service.md). Benchmark scores
never supply hierarchical family/variant probabilities.

[Frontend instructions](frontend/README.md) cover the proxy, build, and isolated browser tests.
This developer service has no authentication or tenant isolation; keep it local.

## What an original baseline run produces

| File | Contents |
| --- | --- |
| `report.json` | Retrieval recall@k, pair precision/recall/F1, PR curve, reliability bins, errors and timings |
| `manifest.json` | Input/source checksums, Git revision and dirty state, dependency versions and platform |
| `config.json` | Effective dataset protocol, seed, model and retrieval settings |
| `model.json` | Feature order, scaling statistics, logistic coefficients and selected thresholds |
| `*-pairs.jsonl` | Every labeled pair's probability, decision, features, conflicts and retrieval ranks |
| `*-candidates.jsonl` | Ranked candidates with retrieval channels and lexical scores |

The matcher and TF-IDF vocabulary fit training records only. The automatic-match threshold
is chosen on validation pairs to maximize recall subject to empirical precision and a minimum
support count. If no threshold qualifies, automatic matching is disabled. This is not a
precision guarantee or probability calibration. Scores between the rejection and match
thresholds go to review; rejection uses a fixed, recorded threshold.

Classification evaluates every supplied labeled pair, independently of retrieval. Retrieval
indexes all listings in the evaluated split and reports directed recall against labeled pairs
and, when complete, benchmark entity metadata. The combined measure treats a pair as retrieved
if either direction finds it. Unknown candidate labels are never assumed negative. See
[the evaluation protocol](docs/evaluation.md) for limitations and error analysis.

## Ingest your own listings

Canonical CSV and JSONL require string `source` and `source_id` fields. Optional fields include
`title`, `description`, `brand`, `category`, `model`, `gtin`, `attributes`, `price`, `currency`,
`product_url`, `image_url`, and `observed_at`. Prices and identifiers are strings to preserve
source formatting and leading zeroes. CSV attributes are a JSON object in a quoted cell.

```sh
uv run catalog-identity profile path/to/listings.jsonl
```

The Python `load_listings` adapter returns validated records with source provenance. Normalization
retains the original record and a transformation log. Invalid GTIN checksums are flagged and
excluded from identifier lookup. Model punctuation is handled conservatively; unknown values
are not agreements. Brand aliases are explicit configuration of the normalization function.

## Measured results and limitations

On WDC Products with 80% corner cases, small development size, and 100% unseen test entities:

| Measure | Original baseline | Enriched model + hybrid retrieval |
| --- | ---: | ---: |
| Test average precision | 0.46231 | 0.52345 |
| Test automatic-match precision | Undefined (no matches) | 80.0% |
| Test automatic-match recall | 0.0% | 8.0% |
| Directed retrieval recall@50 | 99.4% | 99.6% |

The enriched model produced 40 true and 10 false matches at its frozen validation-selected
threshold. **The 99% precision aspiration is not met.** The test has 1,000 offers, not a full
commercial catalog. [Experiment records](experiments/README.md) include counts, hashes,
model comparisons, calibration, retrieval ablations, and reproduction commands.

Hierarchical clustering uses category-specific rules, complete cross-cluster support, and
hard-conflict/human-rejection constraints. Unknown variant attributes abstain. Its synthetic
results are not real-world cluster accuracy. Incremental changes invalidate evidence and
reconcile opaque IDs after a small-catalog rebuild; this is not a streaming implementation.

```sh
uv run catalog-identity cluster-showcase --output runs/clusters
uv run catalog-identity incremental-showcase --output runs/updates
uv run catalog-identity benchmark-scale --sizes 1000 10000 --queries 100 --output runs/scaling
uv run catalog-identity develop --config configs/wdc-products.example.json --negative-policy random_only --output runs/random-negatives
```

Scaling measurements cover lexical retrieval on synthetic populations through 10,000 listings.
The live candidate endpoint caches its index by catalog revision; the dashboard loads at most 200
listings. Structured graph generation is capped at 500 listings per brand/model block.
There is no 100k service claim, real human-labeled hierarchical evaluation, image experiment,
or cross-encoder result. Larger-corpus ingestion and production indexing remain future work.

## Verification

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
cd frontend
npm run build
npm run typecheck
npx playwright install chromium
npm run test:e2e
```

CI runs Python 3.11/3.12 on Windows/Linux, PostgreSQL integration, and Chromium dashboard tests.
The PostgreSQL test skips locally unless `TEST_DATABASE_URL` names a disposable test database.
Browser tests start their own API and production dashboard with an isolated synthetic database.

Read [docs/datasets.md](docs/datasets.md) for benchmark semantics and data restrictions.

## License

Project code and original synthetic fixtures are MIT licensed. Third-party datasets and
product images are not included and are not covered by this license.
