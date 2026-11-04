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

For train-only model selection, the fixed v3/v4 LightGBM screening matrix uses three
entity-disjoint folds and never opens or hashes validation/test files:

```sh
uv run catalog-identity screen-matchers --config configs/wdc-products-large.json --output runs/screening-large
```

Its precision envelopes are threshold-selection diagnostics, not precision guarantees.
Official large train/validation share all 500 entities despite disjoint offers; subsequent
validation comparison is seen-entity development evidence. See the
[registered experiment](experiments/matcher-v4-preregistration.md) for selection and stopping rules.
The completed four-candidate screen retained the v3 control: v4 improved mean ranking AP
but reduced recall at the 99% empirical precision envelope. All unsuccessful candidates
and the unchanged 22-correct/zero-false validation replay are preserved in
[the experiment results](experiments/README.md#train-only-entity-screening-of-matcher-v4).

A separate optional semantic experiment uses cached, frozen MiniLM pair features on v3:

```sh
uv run catalog-identity screen-semantic --config configs/wdc-products-large.json --output runs/semantic-screening
```

It requires previously cached model assets, runs offline, and uses the stricter
[phase 2 selection rule](experiments/matcher-semantic-preregistration.md). Cache records stay
under ignored `runs/`; the CLI and service defaults remain unchanged.
The completed semantic screen also retained the control: summaries slightly improved AP
but reduced recall at the target precision, and vector interactions performed worse.
No further official-validation run followed that failed selection. See
[the complete phase 2 outcomes](experiments/README.md#frozen-semantic-pair-features-phase-2).

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

A later **validation-only** comparison uses the official large development split and stronger
specification features. On the same 4,500 validation pairs, LightGBM's false matches fell from
four to zero; correct matches fell from 25 to 22 (recall 5.0% to 4.4%). This is a precision
tradeoff observed on development data, not a new test result or a 99% precision guarantee.
The earlier frozen test results above remain unchanged.

```sh
# Extract the official *_train_large.json.gz and *_valid_large.json.gz files locally first.
uv run catalog-identity develop --config configs/wdc-products-large.json --feature-version 3 --output runs/development-large
uv run catalog-identity validation-errors --config configs/wdc-products-large.json --model runs/development-large/lightgbm --output runs/validation-errors-large
```

The dashboard can queue these validation-only comparisons and display completed summaries.
The API's default evaluation mode leaves test data unopened; baseline test evaluation is an
explicit separate action. Version 2 features remain the CLI default for reproducibility.

Hierarchical clustering uses category-specific rules, complete cross-cluster support, and
hard-conflict/human-rejection constraints. Unknown variant attributes abstain. Its synthetic
results are not real-world cluster accuracy. Incremental changes invalidate evidence and
reconcile opaque IDs after a small-catalog rebuild; this is not a streaming implementation.

```sh
uv run catalog-identity cluster-showcase --output runs/clusters
uv run catalog-identity incremental-showcase --output runs/updates
uv run catalog-identity benchmark-scale --sizes 1000 10000 100000 --queries 100 --output runs/scaling
uv run catalog-identity develop --config configs/wdc-products.example.json --negative-policy random_only --output runs/random-negatives
```

Scaling measurements cover lexical retrieval on synthetic populations through 100,000 listings.
The latest run measured 99 ms median / 132 ms p95 queries at 100k, with about 742 MiB sampled
process RSS. At 10k, median query time fell from 133 ms to 8.6 ms while preserving ranking.
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
