# Catalog Identity Engine — Project Specification

## What we are building
An open-source, reproducible machine-learning and search system that resolves messy product listings from multiple ecommerce stores into stable identities for the same real-world products. It combines structured attribute extraction, hybrid candidate retrieval, learned pairwise scoring, and constrained graph clustering. It exposes an API and an inspection dashboard so developers can understand *why* listings match, where uncertainty remains, and how well the system performs.

This is inspired by public Shopify engineering work on large-scale catalog clustering. It is an independent project, not an official Shopify project or a claim about Shopify's internal requirements.

## Core example and identity semantics
- `Sony WH-1000XM5 Wireless Black` and `Sony XM5 Black Noise Cancelling Headphones` should share a product family and, if all defining attributes agree, an exact variant.
- `Sony WH-1000XM5 Silver` belongs to the same family but a different exact variant.
- `Sony WH-1000XM4 Black` belongs to a different family.

Model two identity levels:
1. **Product family**: the underlying model/product family, allowing variant differences such as color when appropriate.
2. **Exact variant**: the same sellable product configuration, including category-dependent variant-defining attributes (e.g., color, size, capacity, pack count, region, compatibility).

Do not assume every category uses the same variant rules. Distinguish unknown from conflicting attributes. Store stable opaque internal IDs, not title-derived IDs. Keep source listing IDs and provenance.

## Inputs and outputs
**Input listing fields** (optional unless otherwise required by a dataset): source/store ID, source listing ID, title, description, brand, category, model/MPN, GTIN/UPC/EAN, attributes, price and currency, image URL or permitted image asset, product URL, timestamp.

**Outputs**:
- Candidate listings with retrieval sources and scores.
- Pairwise match probability, model version, evidence features, and predicted relation: same exact variant, same family/different variant, different product, or uncertain.
- Stable family/variant cluster IDs with member listings and provenance.
- Conflicts, human-review items, and cluster merge/split history.
- Evaluation reports and latency/throughput metrics.

## Data strategy
Start with the WDC Products benchmark (https://webdatacommons.org/largescaleproductcorpus/wdc-products/index.html) and its documented splits/label semantics. For scale experiments, inspect the larger WDC Product Data Corpus (https://webdatacommons.org/largescaleproductcorpus/v2/index.html). Use a separately curated and labeled showcase dataset for family-versus-variant distinctions if the benchmark does not support both. Verify licensing and attribution before redistribution. Do not assume the benchmark includes usable images or category-specific variant labels.

Maintain three clearly separated data uses:
- Benchmark evaluation with official labels and splits.
- Larger unlabeled corpus for retrieval/throughput testing (not accuracy claims without ground truth).
- Curated human-labeled examples for family/variant visualization and error analysis.

Prevent leakage through near-duplicate listings and shared product entities where the evaluation protocol requires entity-disjoint splits.

## Architecture

```text
Product feeds / benchmark data
          |
  Schema validation + normalization
          |
 Attribute extraction + provenance
          |
  +-------------------------------+
  | Exact identifiers / BM25      |
  | Semantic embedding ANN        |
  +-------------------------------+
          |
   Candidate union + deduplication
          |
  Pairwise features + LightGBM
          |
 Optional cross-encoder for uncertain pairs
          |
  Calibrated pair probabilities
          |
  Conflict-aware graph clustering
          |
 Stable family / variant identities
          |
 PostgreSQL + FastAPI + review dashboard
```

### 1. Ingestion and normalization
Implement a canonical typed listing schema and adapters for the benchmark and CSV/JSONL. Normalize Unicode, whitespace, punctuation, units, brand aliases, model-number formatting, and common attribute values without destroying distinguishing information. Preserve raw fields and transformation provenance. Handle missing/ambiguous data explicitly. Add profiling and validation reports.

### 2. Baseline retrieval and matching
Build deterministic candidate rules for trustworthy identifiers and a TF-IDF/BM25 text retriever. Train a logistic-regression baseline with title/token similarity, brand agreement, model/identifier agreement, category, attribute agreement/conflict, and optional price features. Do not assume identical identifiers are always sufficient if data quality is questionable. Evaluate retrieval and matching independently.

### 3. Hybrid candidate retrieval
Use sentence-transformer text embeddings and FAISS for approximate nearest-neighbor retrieval. Combine ANN candidates with BM25 and identifier-based candidates; deduplicate and record which channels retrieved each pair. Measure candidate recall@k, query latency, index build time, and memory usage. Avoid O(N²) exhaustive comparisons at scale.

### 4. Pairwise matching
Train a LightGBM classifier on candidate pairs, including hard negatives such as different generations, capacities, sizes, colors, and incompatible specifications. Compare with the baseline. Optionally apply a cross-encoder to only ambiguous pairs. Calibrate probabilities on validation data; choose thresholds for automatic matches, automatic rejections, and review. Avoid presenting feature scores as definitive causal explanations. For hierarchical labels, implement a coherent family/variant decision scheme and validate it on a suitable labeled dataset.

### 5. Conflict-aware clustering
Construct weighted candidate graphs. Implement connected components as a baseline and demonstrate the bridge-edge failure: A matches B, B matches C, but A conflicts with C. Build constrained agglomerative clustering or another well-defined method that scores cross-cluster support and rejects hard conflicts (e.g., incompatible product model or variant-defining attributes). Support two-level family/variant identities, incremental assignment, merge/split history, and stable IDs. Prefer correctness over merging aggressively.

### 6. Persistence and API
Use PostgreSQL for listings, normalized attributes, candidate evidence, pair scores, clusters, membership, model versions, review decisions, and change history. Provide documented FastAPI endpoints for ingest, search, get listing, retrieve candidates, inspect pair decision, get cluster, trigger evaluation, and submit human review. Design idempotent ingestion and a strategy for re-scoring changed listings. Keep expensive jobs asynchronous where needed.

### 7. Frontend
Build a Next.js/TypeScript dashboard with:
- Searchable listing and identity explorer.
- Family and variant views with source-store provenance.
- Side-by-side pair inspection with evidence and conflicts.
- Graph view for cluster edges and rejected merges.
- Human-review queue for uncertain pairs/clusters.
- Evaluation dashboard for accuracy, retrieval, latency, throughput, and costs.

### 8. Evaluation
Measure:
- Retrieval: candidate recall@10/@20/@50, latency, index build cost.
- Pair classification: precision, recall, F1, PR curves, calibration, false-positive/negative breakdown.
- Clustering: pairwise cluster precision/recall/F1, B-cubed precision/recall, cluster purity, erroneous merges, fragmentation.
- System: throughput, p50/p95 latency, peak memory, model/index size, incremental update behavior.

Treat target numbers (e.g., 98% candidate recall@50, 99% pair precision, 90% recall, 99% cluster purity, 100k listings) as aspirations, never pre-established results. Set appropriate acceptance criteria after validating label quality and tradeoffs. Report benchmark protocol and actual measured values.

Run ablations: exact/BM25 vs embeddings vs hybrid; logistic regression vs LightGBM vs cross-encoder; random vs hard negatives; connected components vs constrained clustering; text vs permitted images; static vs incremental identities. Preserve a held-out test set.

## Suggested repository structure
```text
catalog-identity-engine/
├── AGENTS.md
├── PROJECT_SPEC.md
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── configs/
├── data/                 # no unlicensed data committed
├── src/catalog_identity/
│   ├── ingestion/
│   ├── normalization/
│   ├── retrieval/
│   ├── matching/
│   ├── clustering/
│   ├── evaluation/
│   ├── storage/
│   └── api/
├── experiments/
├── tests/
└── frontend/
```

## Definition of done
A developer can set up the repository from README instructions, ingest a supported dataset, reproduce baseline and improved-model evaluations, inspect matching evidence and conflict-aware clusters, query stable product identities through an API, and explore uncertain decisions through the dashboard. All reported metrics must come from actual runs, with versioned configs and a documented evaluation protocol.

## Implementation guidance for Codex
First inspect the repository, verify dataset schema and licensing, scaffold the Python package, implement ingestion/normalization and tests, then build a reproducible baseline retrieval + pairwise matching evaluation. Do not begin with the frontend or advanced models. After the baseline is tested, proceed through the architecture in dependency order and report test results after each substantial change.
