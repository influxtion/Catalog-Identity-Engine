# Matcher development and hierarchical evaluation

The original baseline protocol remains in [evaluation.md](evaluation.md). The `develop`
command is a separate validation-only workflow. It does not open the test path.

## Features and learners

The enriched feature space adds train-fitted title/description character TF-IDF similarities,
token containment, character overlap, and model-token/capacity/color/pack hints to the original
structured comparisons. Text hints are weak features; they are not extracted ground truth,
valid identifiers, or unconditional vetoes. Both learners share this feature space.

Logistic regression standardizes the features. LightGBM uses a recorded deterministic seed
and bounded tree complexity. Source IDs, gold entity IDs, pair IDs and published hard-negative
flags are never model inputs. Portable JSON parameters and LightGBM model text avoid loading
executable pickle artifacts. Tests verify round-trip predictions.

## Calibration and operating thresholds

Validation entities are assigned to two deterministic hash buckets using seed 42. Only pairs
whose endpoints occupy the same bucket are retained. One bucket calibrates a sigmoid on model
logits; the other selects match/reject thresholds. The buckets have disjoint entities.
Cross-bucket negatives are excluded, never relabeled.

A positive pair has probability 1/2 of entering either bucket, while a negative connecting
different entities has probability 1/4. We use weights 1 for positives and 2 for negatives in
calibration and threshold selection, restoring the original sampling prior in expectation.
Raw classification counts and AP/Brier/reliability diagnostics still describe the retained
subset. They are not directly comparable to full-validation baseline metrics.

The measured split contains 772 calibration pairs, 717 policy pairs, and 1,011 excluded
cross-bucket pairs. Match selection targets 99% weighted empirical precision with at least
five predictions. Rejection selection applies the same procedure to nonmatches. If either
policy lacks support, that action is disabled. Small support and repeated model comparisons
mean these operating points offer no statistical precision guarantee.

The enriched logistic model was selected before the new test run because it recovered 19
policy positives with zero false positives, versus 13 for LightGBM. LightGBM's ranking AP was
higher, but recall at the requested precision took priority. The resulting test precision was
only 80%, exposing limited generalization. Do not retune against test errors. Test had already
been used once to report the original baseline; it is not a newly collected blind benchmark.

`--negative-policy random_only` removes published hard-negative training pairs while retaining
all positives. Validation files and partitions stay identical. This changes training size and
prior as well as difficulty, so it is not a size-matched causal experiment. Saved metadata
records original and retained pair counts.

## Semantic retrieval

Optional MiniLM embeddings use `sentence-transformers/all-MiniLM-L6-v2`, pinned to revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The model card declares Apache-2.0; that does not
license benchmark data. Weights require an explicit initial download flag. Remote model code
and authentication tokens are disabled.

FAISS HNSW indexes normalized embeddings. Single-threaded construction and stable listing
order make experiments repeatable within the recorded environment. Hybrid retrieval combines
lexical and semantic ranks using reciprocal-rank fusion, preserving identifier priority.
Candidate records expose each contributing channel. Indexed embeddings are cached; reported
query time excludes fresh text encoding, while build includes encoding. Vocabulary fitting
uses training listings only during benchmark evaluation.

## Frozen test and identity boundaries

`evaluate-frozen` checks original train/validation hashes against the saved policy and loads
the model without fitting. It audits all split boundaries and applies the recorded policy to
test. Retrieval quality, labeled-pair classification and their combined labeled-pair measure
remain separate. Unknown candidate labels are not negatives.

WDC labels describe benchmark entities, not family/variant relationships. The service never
uses these learned probabilities to merge hierarchical identities. Its versioned structured
policy requires brand/model agreement for a family and category-specific defining attributes
for an exact variant. Missing defining evidence abstains; explicit conflicts reject merges.
Human rejections constrain indirect paths, and variant acceptance cannot contradict an active
family rejection. A family match alone does not resolve an unknown variant review.

`cluster-showcase` reports pairwise, B-cubed, purity, merge and fragmentation metrics on an
original synthetic fixture, excluding unresolved labels explicitly. Its three-node comparison
demonstrates the connected-components bridge failure. `incremental-showcase` compares updates
against fresh memberships and checks unchanged-group ID retention. Neither is a human-labeled
real-world clustering result.

## Costs and remaining experiments

`benchmark-scale` measures lexical vocabulary/index construction, p50/p95 query latency,
query throughput, sparse matrix bytes, and process RSS sampled every 5 ms. Build throughput
excludes synthetic record generation. Memory includes interpreter/imports and may retain
allocations from earlier sizes; sampling can miss brief peaks. These are local-process costs,
not end-to-end API latency or exact peak memory. Manifests record source/lock hashes and
environment information. Model artifact sizes accompany the published aggregates.

Cross-encoder reranking, permitted-image comparison, larger-corpus/100k experiments and a real
human-annotated hierarchical set have not been run. They need evidence or suitable data before
adding complexity. No performance or licensing result is implied for these paths.
