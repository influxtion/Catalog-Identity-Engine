# Matcher development and hierarchical evaluation

The original baseline protocol remains in [evaluation.md](evaluation.md). The `develop`
command is a separate validation-only workflow. It does not open the test path.

## Train-only screening and v4 alignment

`screen-matchers` runs the [preregistered four-candidate matrix](../experiments/matcher-v4-preregistration.md)
using only the training file. Neither validation nor test is loaded or hashed. Three seed-42
entity folds separately fit vocabularies and models; training pairs require both endpoints
outside the assessment fold and assessment pairs require both inside. Boundary pairs are
excluded and counted. Complete, consistent entity metadata is required. All assessment
positive entities and matched positive entities are counted per fold.

Assessment weights are 1 for positives and 3 for negatives, correcting the expected loss of
cross-fold negatives across the three assessments. Training pair restrictions also alter
class balance. This is an approximate training-source screening protocol, not a guarantee
for deployment data. Raw and weighted AP and precision envelopes are reported separately.
The 95/98/99% envelopes select their own thresholds on the reported population and require
five predictions: they are optimistic diagnostics, not independently validated policies.
Fold scores are never pooled into one threshold. The fixed selection rule includes support
in two folds and a numerical AP guard; this does not itself establish improvements in two folds.

Version 4 extends v3 with separator-aware model-code alignment, preserved numeric/suffix
distinctions, typed capacity/frequency/dimension/pack evidence, and separate module count,
module capacity and total capacity. For example, `2x8GB` and `16GB` agree on total capacity
but the latter's module count remains unknown. Missing and one-sided information remain
separate from disagreement. Symmetric title-to-other-description evidence includes character,
token, code and specification support; compatibility lists never become hard identity rules.
IDF weights and vocabularies use training listings only. Unknown words receive the largest
training unigram IDF so novel distinguishing terms cannot vanish from containment denominators.
Additional text parsing is cached
and bounded to 8,192 characters and 24 codes per field. Existing v2/v3 feature implementations
and artifacts remain supported; v2 remains the CLI default.

```sh
uv run catalog-identity screen-matchers --config configs/wdc-products-large.json --output runs/screening-reproduction
# After selection, pass the selected feature version, num_leaves and n_estimators to develop.
uv run catalog-identity develop --config configs/wdc-products-large.json --feature-version 3 --learners lightgbm --num-leaves 15 --n-estimators 200 --output runs/selected-reproduction
```

Official large training and validation have disjoint listing IDs but share all 500 entities.
The calibration/policy buckets are disjoint from each other, not from training entities.
Official validation therefore measures seen-entity development. Reports now expose this
overlap explicitly without rejecting the benchmark's legitimate protocol. Full-validation
replays reuse calibration and policy-selection observations and are not independent evidence.
No new frozen test run or automatic production promotion is part of this experiment.
The completed screen selected the existing v3 control; v4's mean ranking improvement did
not translate into better high-precision recall. See [all measured outcomes](../experiments/README.md#train-only-entity-screening-of-matcher-v4).

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

`develop --feature-version 3` adds train-fitted word/bigram similarity and symmetric hints for
quantities, form factors, editions, audio channels and specific model codes. One-sided hints
remain distinct from explicit conflicts. These are experimental learned features, not hard
benchmark label rules. Version 2 remains the default for reproduction and existing saved
models remain loadable. `validation-errors --config ... --model ... --output ...` exports local
pair evidence without opening test data; its full-validation summary is diagnostic, not an
independent performance estimate.

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

Cross-encoder reranking, permitted-image comparison, larger-corpus/end-to-end 100k experiments and a real
human-annotated hierarchical set have not been run. They need evidence or suitable data before
adding complexity. No performance or licensing result is implied for these paths.

The later synthetic lexical benchmark reaches 100k listings; its scope excludes the live
service. The process-local service cache is revision-aware but cold builds are synchronous.
Normalized listing evidence now persists separately from raw source observations, with an
additive schema migration. API writes and rebuild jobs are atomic. Interrupted running jobs
still need operator recovery; leases/retries, persistent indexes, large-catalog dashboard
pagination, and independently annotated hierarchical evaluation remain open work.
