# Measured experiments

## Original baseline

The first real run used code commit `9b089f4f99d86bfc2709cb962488ae9be96e7e2a`, with a
clean working tree, seed 42, and the official WDC Products pairwise archive for 80%
corner cases. Development size: small. Test unseen entities: 100%.

Train and validation each contain 2,500 pairs; test contains 4,500. Each split has
1,000 offers. The audit found no shared test entities with train or validation.

| Measure | Validation | Test |
| --- | ---: | ---: |
| Directed candidate recall@10 | 92.9% | 92.7% |
| Directed candidate recall@20 | 97.2% | 97.6% |
| Directed candidate recall@50 | 99.4% | 99.4% |
| Pair average precision | 0.53899 | 0.46231 |
| Brier score | 0.12334 | 0.07941 |
| Automatic matches | 0 | 0 |
| Review decisions | 1,206 | 1,931 |
| Rejection decisions | 1,294 | 2,569 |

No validation threshold met 99% empirical precision with at least five predicted
matches. Automatic matching therefore remained disabled. Automatic-match recall
and F1 are zero; precision is undefined. Zero false positives in this run is not
evidence of a useful high-precision matcher. Average precision summarizes the ranking
across thresholds; it is not precision at the operating point.

The retrieval population contains only the 1,000 test offers. This is not a claim
about a larger catalog. The published pair files lack explicit GTIN/model/category
fields, so this run retrieves through TF-IDF alone. Structured feature paths are
covered by separate synthetic tests but receive no benchmark accuracy claim here.

[The aggregate run record](wdc-80-small-unseen100-baseline.json) contains the exact
configuration, input hashes, source hashes, versions, hardware-dependent timings,
and scalar/reliability metrics. No product text or images are redistributed. Full
local reports, curves and pair evidence remain in the ignored `runs/` directory.

Reproduce from the repository root after supplying the original archive files:

```sh
uv sync --locked --extra dev
uv run catalog-identity evaluate --config configs/wdc-products.example.json --output runs/wdc-baseline-reproduction
```

## Enriched matching and hybrid retrieval

[improved-v3.json](improved-v3.json) preserves the validation-selected model and frozen test
result. Features were developed using validation errors; test was not opened by the development
runner. The original baseline above had already used this test once. The improved model was
selected before the second test evaluation and has not been retuned against that result.

Validation entities are split between calibration (772 pairs) and policy selection (717).
Another 1,011 cross-partition pairs are excluded. Calibration and policy selection weight
negatives by two to correct unequal inclusion probabilities; diagnostic AP below describes the
retained policy subset. See [the detailed protocol](../docs/development.md).

| Model, same enriched features | Policy AP | Correct matches | False matches |
| --- | ---: | ---: | ---: |
| Logistic | 0.76413 | 19 | 0 |
| LightGBM | 0.81170 | 13 | 0 |

Logistic was selected for recall at the requested empirical precision, despite LightGBM's
higher ranking AP. The frozen logistic threshold was 0.8597742461487005. Zero errors among
19 policy predictions did not generalize to the requested 99% precision.

| Retrieval channel, validation | Recall@10 | Recall@20 | Recall@50 |
| --- | ---: | ---: | ---: |
| Lexical | 92.9% | 97.2% | 99.4% |
| MiniLM HNSW | 88.3% | 95.5% | 99.2% |
| Hybrid | 93.5% | 97.8% | 99.6% |

The cached-embedding retrieval comparison was run before the expanded implementation-manifest
helper was added; its record retains the metadata actually recorded, including encoder revision
and input hashes. No missing historical provenance has been invented. Reproduction commands
now emit the expanded manifest.

The frozen hybrid/logistic test run used clean commit `dd5656c`. Classification across all
4,500 supplied pairs gave **40 TP, 10 FP, 460 FN, 3,990 TN**: 80% precision, 8% recall,
14.55% F1, and AP 0.52345. Hybrid directed retrieval recall was 94.7%/98.4%/99.6% at 10/20/50.
These are separate retrieval and pair-classification measures, not full-catalog clustering
accuracy. The service consequently keeps benchmark scores separate from hierarchical merges.

Aggregate records correct a legacy reporting field that said `none` even for sigmoid-calibrated
scores; no probabilities or numeric metrics were changed. The combined retrieval-plus-matching
result is separately derived from the original saved pair decisions and ranks, with its input
hash recorded. No model was refitted or rescored for this reporting correction.

```sh
uv sync --locked --extra dev --extra models --extra semantic --extra server --extra benchmarks
uv run catalog-identity develop --config configs/wdc-products.example.json --output runs/development-reproduction
uv run catalog-identity retrieval-ablation --config configs/wdc-products.example.json --output runs/retrieval-reproduction --allow-model-download
uv run catalog-identity evaluate-frozen --config configs/wdc-products.example.json --model runs/development-reproduction/logistic --hybrid --output runs/frozen-reproduction
```

The selected logistic artifact directory contains 2,922,385 bytes, including vocabulary,
numeric model parameters and policy. Aggregate metadata records file sizes and hashes; model
vocabulary and raw benchmark evidence remain local.

## Ablations and synthetic system checks

[ablations-v1.json](ablations-v1.json) records actual runs with source hashes and environment
metadata. No real product text or images are included.

Removing the 1,000 published hard-negative training pairs retained 1,500 pairs, including all
500 positives. Validation policy AP fell to 0.60926 for logistic and 0.67383 for LightGBM;
neither produced a qualifying automatic-match threshold. This comparison changes training
size and class prior, so it does not isolate negative difficulty from those effects. Test was
not opened for this ablation.

The synthetic hierarchy fixture has 11 resolved family labels and 10 resolved variant labels;
one/two unresolved listings are excluded respectively. All measured cluster metrics were
perfect on this small fixture. This demonstrates implementation behavior, not realistic
accuracy. The separate three-node bridge test produced one erroneous merge with connected
components and zero with constrained clustering.

Two synthetic updates resolved a missing color, then split a variant by changing a color.
Both incremental results matched fresh membership partitions. Each update retained all five
unchanged family IDs and all seven unchanged variant IDs. Identical reingestion changed zero
records, and updates exposed stale identities until rebuilding. Timings cover only 12 records
on SQLite; they provide no PostgreSQL throughput claim.

| Synthetic lexical population | Build incl. vocabulary | Query p50 / p95 | Queries/sec | Sampled process RSS |
| --- | ---: | ---: | ---: | ---: |
| 1,000 | 0.072 s | 9.0 / 10.2 ms | 107.7 | 141.6 MiB |
| 10,000 | 0.825 s | 132.9 / 186.4 ms | 7.3 | 199.9 MiB |

Scaling used clean commit `4602529`, seed 42, 100 queries per population, and k=50. Sparse
matrix storage was 1,138,076 and 11,742,032 bytes. RSS includes the interpreter and imports,
is sampled every 5 ms, and may retain allocations between sizes. Query latency excludes API,
serialization, and fresh index construction. Build throughput excludes fixture generation.
These are local hardware measurements through 10k synthetic records, not a 100k service result.

```sh
uv run catalog-identity develop --config configs/wdc-products.example.json --negative-policy random_only --output runs/random-negative-reproduction
uv run catalog-identity cluster-showcase --output runs/cluster-reproduction
uv run catalog-identity incremental-showcase --output runs/incremental-reproduction
uv run catalog-identity benchmark-scale --sizes 1000 10000 --queries 100 --output runs/scaling-reproduction
```

Cross-encoder, image, real human-labeled hierarchy and larger-corpus experiments remain unrun.
The current evidence supports a reproducible inspection prototype, not production acceptance.
