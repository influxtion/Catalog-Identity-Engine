# Measured baseline

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

Next experiments should use validation errors to improve pair discrimination and
probability reliability, preserving this result as the original held-out baseline.
Do not lower the target or choose new features by optimizing against this test run.
