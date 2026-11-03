# Catalog Identity Engine

Resolve product listings across stores, with reproducible experiments and inspectable evidence.
Independent project inspired by public catalog-matching research; not affiliated with Shopify.

The working baseline includes typed ingestion, conservative normalization, candidate retrieval,
logistic pair scoring, and reproducible evaluation with an offline demo.
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

## What a run produces

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

## Scope and next steps

Implemented: ingestion and profiling, official split loading, normalization, identifier/TF-IDF
candidate union, logistic baseline, threshold selection, reports, tests and CI.

Next: evaluate hard negatives and retrieval misses on validation data, compare semantic retrieval
and LightGBM, calibrate probabilities, then build conflict-aware family/variant clustering with
its own labeled fixtures. Persistence, API, review dashboard and scale benchmarks follow those
steps. No clustering, live service, dashboard, or production-quality accuracy is claimed yet.

The current retriever uses sparse per-query products, avoiding a dense all-pairs matrix. It is
an exact baseline for benchmark-sized populations; it is not an ANN index or a 100k-listing
scaling result. Evidence features describe observations, not causal explanations of a match.

Read [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full design and
[docs/datasets.md](docs/datasets.md) for benchmark semantics and data restrictions.

## License

Project code and original synthetic fixtures are MIT licensed. Third-party datasets and
product images are not included and are not covered by this license.
