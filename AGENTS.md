# Codex Instructions — Catalog Identity Engine

Read `PROJECT_SPEC.md` before changing code. This repository implements an open-source product entity-resolution system across heterogeneous ecommerce listings.

## Working principles
- Implement a reproducible, testable end-to-end system; do not build a generic chatbot or merely wrap an LLM.
- Preserve the distinction between product-family identity and exact-variant identity. Respect each benchmark's actual label semantics; do not silently reinterpret labels.
- Build a functioning simple baseline before advanced models, clustering, UI, or scaling work.
- Prefer deterministic rules and measurable ML components where appropriate. Treat LLM or image models as optional, experimentally justified additions.
- Never invent performance numbers, citations, labels, data licensing permissions, or benchmark outcomes.
- Avoid leakage: honor published dataset splits and use entity-disjoint evaluation when claiming unseen-product generalization.
- Measure candidate-generation recall separately from pairwise matching and end-to-end clustering quality.
- Optimize for high precision; erroneous cross-product merges are especially costly.
- Add automated tests for every major module, especially normalization, conflicting identifiers, candidate retrieval, and non-transitive cluster edges.
- Record model/data/config versions, seeds, thresholds, and evaluation commands. Ensure the README gives reproducible instructions.
- Do not introduce Kubernetes, Kafka, or a graph database without demonstrated need.

## Stack
Python 3.11+, uv or pip, Polars/DuckDB, scikit-learn, sentence-transformers, FAISS, LightGBM, optional PyTorch cross-encoder, NetworkX for prototyping, FastAPI, PostgreSQL, Next.js/TypeScript/Tailwind, pytest, Docker, GitHub Actions, MLflow if useful.

## Implementation order (dependency order, not a schedule)
1. Inspect repository and benchmark schema; document dataset label definitions and license.
2. Build typed listing schema, ingestion, normalization, profiling, and split-aware loaders.
3. Build rule-based and TF-IDF/BM25 candidate retrieval and a logistic-regression pairwise baseline.
4. Establish evaluation scripts and error analysis before adding complexity.
5. Add hybrid semantic retrieval, LightGBM matching, calibrated thresholds, and optional cross-encoder reranking.
6. Implement conflict-aware clustering, stable IDs, and cluster evaluation.
7. Add persistence, incremental ingestion, APIs, review workflows, and UI.
8. Run ablations and scaling benchmarks; document actual results.

## Execution rules
- Before a substantial change, state the intended scope and the tests to run.
- Keep changes modular and avoid unnecessary rewrites.
- Run the relevant tests and report pass/fail honestly.
- If dataset access or credentials are unavailable, provide a small clearly labeled synthetic fixture and document how to run the real benchmark; never represent synthetic metrics as real results.
- Do not auto-download or redistribute copyrighted product images without checking rights.
- If an operation is destructive or requires secrets or external credentials, ask first.
