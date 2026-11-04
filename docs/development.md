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

## Supervised joint-text experiment and resource gate

The separately [registered phase 3](../experiments/cross-encoder-preregistration.md) trains
only the final two layers of the cached six-layer MiniLM BERT backbone, plus a small head
combining final CLS with fold-fitted standardized v3 evidence. The first four layers and
embeddings remain frozen and in evaluation mode. Balanced 256-token pairs preserve both
titles, account for field labels/special tokens, and allocate remaining capacity to brand
and description. Inference averages both orientation logits before sigmoid.

Training is fixed at three epochs, all retained pairs, float32 CPU with 14 threads,
microbatch 8 and accumulation 4. Epochs, learning rates, freezing, warmup/decay, gradient
clipping and final-checkpoint selection are registered before results. Missing data and
benchmark identity semantics are unchanged. No serving adapter or default promotion is
enabled by this experiment.

```sh
uv run catalog-identity pilot-joint --config configs/wdc-products-large.json --validation-report experiments/matcher-v4-validation.json --output runs/joint-timing
# Run only if BOTH registered resource budgets pass.
uv run catalog-identity screen-joint --config configs/wdc-products-large.json --timing runs/joint-timing/timing.json --control experiments/matcher-v4-screening.json --output runs/joint-screening
# Run only if the completed screen qualifies under the fixed selection rule.
uv run catalog-identity develop-joint --config configs/wdc-products-large.json --screening runs/joint-screening/screening.json --output runs/joint-selected
```

The timing-only pilot uses a deterministic spread of training text lengths with synthetic
alternating labels, reports no accuracy and saves no pilot model. Four warmup microbatches
precede 16 measured forward/backward microbatches; two-orientation inference is timed
separately. Projection includes all three epochs, actual fold counts, fixed 256-token padding,
measured model/feature/tokenization costs, serialization reserve and a 1.5 safety factor.
Conditional validation pair counts come from the prior committed aggregate; the pilot does
not open or hash validation/test data. Screening must project within 90 minutes and the
conditional full fit/validation within 45 minutes. A failed gate stops the phase without
changing epochs, token budgets, layers or data. Timing is hardware-specific, not accuracy.
The completed pilot failed both budgets: projected screening 167.88 minutes and conditional
full fit/validation 100.82 minutes. No joint-model accuracy run followed. The
[timing report](../experiments/cross-encoder-timing.json) and
[measured limitations](../experiments/README.md#supervised-joint-text-model-phase-3-resource-gate)
are preserved; a different compute plan would require separate authorization and registration.

Screening verifies the saved control's training checksum, seed, weighting, numeric features,
loader, fold and threshold-selection implementation. Models reset from original cached
weights for each fold. Reused development folds and empirical precision envelopes retain
their earlier limitations. Resource-aborted partial screens remain explicitly incomplete.

Model artifacts use safetensors plus JSON v3 vocabulary/scaler/architecture/training metadata,
the exported backbone configuration and tokenizer. Loading is local and self-contained;
it verifies config/tokenizer hashes, tensor keys/shapes/dtypes/finiteness and calibration.
No pickle or original-cache dependency is needed to reload a saved artifact. Input/source
provenance accompanies actual selected-model artifacts. Optional runtime tests skip when
PyTorch/Transformers are absent; pure token-budget and existing model tests remain available.

## Execution-only GPU addendum

The original CPU-only Torch environment did not expose the machine's physical RTX 4060
Laptop GPU. Its failed timing report remains published. The separately registered
[GPU execution addendum](../experiments/cross-encoder-gpu-execution-addendum.md) uses the
same architecture, training schedule and selection rule in isolated `.venv-gpu`; only the
Torch wheel differs from the original environment's pinned package versions. No drivers,
global Python settings or model assets are changed. GPU runtime installation is separate
from model-weight downloads, which remain disabled.

To reproduce the isolated runtime after creating the original CPU environment:

```powershell
.venv\Scripts\python -m venv .venv-gpu
.venv-gpu\Scripts\python -m pip install --no-deps -r experiments/gpu-runtime-requirements.txt
.venv-gpu\Scripts\python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu132 torch==2.15.0+cu132
.venv-gpu\Scripts\python -m pip install --no-deps -e .
.venv-gpu\Scripts\python -m pip check
```

The package pins and [environment/wheel record](../experiments/gpu-execution-environment.json)
preserve the actual Windows/Python 3.12 setup; another platform needs an explicitly
compatible runtime. Installing dependencies does not authorize model-weight downloads.

Explicit `--device cuda` enables this path; CPU remains the default. The execution manifest
records exact dependencies/hardware, IEEE float32 precision, deterministic algorithms,
MATH attention, autocast off, and non-fused/non-foreach AdamW. It is fingerprinted and must
match the timing gate, screening and conditional full fit. CUDA intervals synchronize around
training and symmetric inference, include transfers, and report allocated/reserved GPU peaks
after warmup separately from CPU RSS. CUDA failures do not silently switch execution modes.

```powershell
# Existing runtime preparation records exact pins and official wheel hashes under ignored runs/.
.venv-gpu\Scripts\python -m catalog_identity.cli pilot-joint --config configs/wdc-products-large.json --validation-report experiments/matcher-v4-validation.json --device cuda --output runs/joint-gpu-timing
# Only after the new GPU gate passes:
.venv-gpu\Scripts\python -m catalog_identity.cli screen-joint --config configs/wdc-products-large.json --timing runs/joint-gpu-timing/timing.json --control experiments/matcher-v4-screening.json --device cuda --output runs/joint-gpu-screening
```

New model artifacts preserve training execution metadata but still load on CPU by default.
Legacy artifacts without execution flags remain loadable without inventing their history.
CPU/CUDA results need not be bitwise equal; the fictional interoperability smoke uses a
declared 1e-5 float32 tolerance. The [official Torch CUDA precision documentation](https://docs.pytorch.org/docs/2.15/notes/cuda.html)
requires using the new precision API without mixing legacy TF32 setters; runtime flags
follow that guidance. The wheel comes from the [official CUDA 13.2 index](https://download.pytorch.org/whl/cu132/torch/).

The completed GPU screen failed the fixed accuracy-selection rule: mean weighted AP
0.88199 and 99% envelope recall 5.71%, versus control 0.91120 and 22.78%. All three folds
completed without execution failure, but none improved recall. The phase stopped without
conditional full fitting, official validation or historical-test evaluation. See
[the complete negative result](../experiments/README.md#completed-supervised-joint-model-comparison).

## Earlier feature experiments

Feature versions 5 and 6 are optional semantic experiments on unchanged v3, separately
[preregistered after the negative v4 screen](../experiments/matcher-semantic-preregistration.md).
Version 5 adds title/title, description/description and symmetric cross-field cosine
summaries. Availability flags describe both/one-sided titles and descriptions, and
both/any cross comparisons; they do not correspond to the ordering of cosine minima/maxima.
Version 6 adds 384 absolute differences and 384 elementwise products of title vectors.
Missing fields are zero vectors plus availability, never encoded empty strings.

The existing pinned MiniLM encoder runs offline on CPU with 256-token truncation and
384-dimensional normalized output. These features require the optional `semantic` extra
and previously cached model assets. They never enable downloads. Versions 2–4 do not
import or load the optional runtime. No fine-tuning or corpus-fitted embedding transform
is performed. Normalized title and description are encoded separately; description
compatibility language remains learned evidence, not an identity constraint.

`runs/semantic-cache` holds content-addressed numeric `.npz` records, loaded without pickle.
Keys include exact weight/tokenizer/config asset hashes, model revision and dependency
versions, normalization/preprocessing version, field type and text digest. Records validate
identity, shape, float32 type, finiteness, unit norm and token count before use. Atomic writes
avoid partial records; corrupt records fail explicitly. The in-memory cache keeps at most
16,384 entries with least-recently-used eviction; evicted entries reload from validated disk
records without re-encoding. Cache location is not part of model identity. Portable matcher
metadata records exact encoder requirements and verifies them
against the offline local encoder on load. Caches and embeddings must stay uncommitted.

```sh
# Requires the already cached pinned MiniLM assets; this command never downloads them.
uv run catalog-identity screen-semantic --config configs/wdc-products-large.json --output runs/semantic-screening --cache runs/semantic-cache
```

This command runs exactly nine fits across the existing three training entity folds. One
encoder and training-only vector cache serve all fits, while vocabularies and classifiers
fit separately per fold. Reports include missing/truncated fields, asset hashes, runtime
versions, encoding/thread settings, cache/encoding/fit timing, and every candidate outcome.
The stricter registered rule requires gains in two folds, at least .02 absolute mean recall
gain, support in all three folds and mean weighted AP within 1% of control. Reusing exposed
folds makes this development selection, not independent generalization evidence. Official
validation is run only if a semantic candidate qualifies; otherwise the phase stops.
The completed registered experiment selected the control and stopped without official
validation: neither v5 nor v6 improved the target high-precision recall. All outcomes are
preserved in [the phase 2 report](../experiments/README.md#frozen-semantic-pair-features-phase-2).

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
