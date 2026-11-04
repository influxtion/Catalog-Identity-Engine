# Measured experiments

## Completed supervised joint-model comparison

The single registered joint model completed all three folds from clean commit `7fe7579`
using the frozen GPU execution profile. [Complete aggregates](cross-encoder-gpu-screening.json)
and [the stop decision](cross-encoder-gpu-selection.json) preserve the negative outcome.

| Candidate | Mean weighted AP | Mean recall at 99% weighted empirical precision | Improved folds |
| --- | ---: | ---: | ---: |
| Saved v3 control | 0.91120 | 22.78% | — |
| Joint MiniLM + v3 evidence | 0.88199 | 5.71% | 0 of 3 |

The joint model had a supported 99% envelope in every fold but failed the recall and AP
requirements. Its per-fold envelopes selected 162/134/172 true positives, zero false
positives, and 35/39/33 distinct matched positive entities. These thresholds were selected
on the reported development fold; zero errors at those points is not a precision guarantee.

**The fixed rule retained the existing v3 control.** Actual fold times totaled 503.80 seconds
(8.40 minutes). No conditional full-train fit, new official-validation replay, historical-test
evaluation, serving promotion or default change followed. No pair errors were inspected for
retuning. This tested configuration underperformed; it does not establish that all joint
models or other training protocols would underperform.

Across the three registered phases, 24 accuracy-screening fits completed (12 + 9 + 3).
Only phase 1 ran its previously reported selected/control full fit and validation replay.
The earlier 22-correct/zero-false seen-entity validation result and historical test results
remain unchanged. The work added tested representations, reproducible gates and an isolated
GPU runtime, but **did not demonstrate a stronger matcher at the target precision**.

Final regression checks passed: 120 tests in the CUDA environment (one PostgreSQL
integration skip), and 119 in the unchanged CPU environment (PostgreSQL and CUDA skips).
Ruff lint and formatting checks passed for all 80 Python files. The real fictional-data
CUDA artifact roundtrip was exact on GPU; default CPU loading differed by at most
2.16e-8, within the predeclared 1e-5 tolerance. Those checks establish implementation
behavior, not benchmark accuracy.

## GPU execution gate for the unchanged joint model

After the published CPU gate failed, a physical RTX 4060 Laptop GPU was identified.
The [execution-only addendum](cross-encoder-gpu-execution-addendum.md) retains the same
single architecture, training schedule and accuracy-selection rule. The isolated
[environment record](gpu-execution-environment.json) verifies that only Torch differs
from the original package versions/lock: the official CUDA 13.2 build replaces CPU Torch.
The original environment and drivers were not modified; existing cached model weights
were reused offline.

The [final synchronized GPU pilot](cross-encoder-gpu-timing.json), from clean source
`e96d21a`, passed the unchanged budgets: **16.25 minutes projected screening** and
**8.03 minutes projected conditional full fit/validation**, including the 1.5 safety factor.
It used deterministic IEEE float32 execution, MATH attention, no autocast or TF32,
microbatch 8 / accumulation 4, and the same training-only sample/synthetic compute labels.
This is a resource result, not model accuracy.

An [initial GPU pilot](cross-encoder-gpu-timing-initial.json) also passed (16.98 / 8.38 min),
but was superseded after correctness-only changes added a conditional source-hash guard
and portable decision-policy artifacts. No model, feature or training parameter changed,
and no accuracy run preceded the replacement pilot. Both timings remain preserved.
The final timing was frozen before the single registered three-fold accuracy comparison;
its completed negative outcome is reported above.

## Supervised joint-text model: phase 3 resource gate

The final separately [registered architecture](cross-encoder-preregistration.md) combines
the cached MiniLM backbone's jointly encoded listing pair with unchanged standardized v3
evidence. Only its final two transformer layers and small prediction head train. Before
accuracy work, a timing-only pilot ran from clean commit `af186b0` with the exact registered
256-token padding, float32 CPU, 14 Torch threads, microbatch 8 and accumulation 4.

[The timing aggregate](cross-encoder-timing.json) records the failed resource gate:

| Measurement | Observed / projected | Registered limit |
| --- | ---: | ---: |
| Training throughput | 16.03 pairs/sec | — |
| Two-orientation inference throughput | 30.94 pairs/sec | — |
| Three-fold screening, including 1.5 safety factor | 167.88 min | 90 min |
| Conditional full fit + validation, including safety factor | 100.82 min | 45 min |
| Sampled process RSS | 893 MiB | Not an exact peak |

**Both time budgets failed, so accuracy training stopped before it started.** No epochs,
layers, token budgets or data were reduced to force a pass. No joint-model accuracy result
is claimed, no pilot model was saved, and no validation/test data was opened or hashed.
The conditional validation counts came from the existing committed phase 1 aggregate.

The pilot sampled 160 training pairs spanning combined text-length quantiles and replaced
their labels with synthetic alternating labels solely for computation. Four warmup and 16
measured training microbatches were followed by two-orientation inference on all 160 pairs.
Actual nonpadding length averaged 158.625 tokens (maximum 256); computation always used
256-token padded tensors. Each orientation truncated 60 description sides and zero title
or brand sides. These are sample diagnostics, not full-dataset truncation counts.

Projection accounts for three epochs on 8,858/11,866/11,536 training pairs, their assessment
sets, full training on 19,835 pairs if qualified, and 1,289 calibration / 1,178 policy /
4,500 full-replay validation pairs. Model load, full numeric-feature fitting, tokenization,
symmetric inference and serialization reserve are included. Timings are measurements and
conservative estimates on this machine, not universal CPU or GPU performance claims.

The implementation and safe local artifacts are covered by tests, including frozen-layer
gradient checks, balanced tokenization, symmetric scoring, train-only vocabularies/scalers,
corrupted tensor/tokenizer rejection and missing optional runtimes. A separate eight-pair
fictional smoke trained three epochs and verified exact predictions after self-contained
safetensors/config/tokenizer reload with the original-cache loader disabled. That smoke
verifies mechanics only. A real-tokenizer API incompatibility was caught and fixed before
the timing run; the registered BERT input representation was preserved.

Phases 1 and 2 remain published negative accuracy experiments. Phase 3 is a measured
compute stop, not an accuracy failure or success. Continuing this architecture requires
a separately approved compute plan, such as suitable GPU execution; no GPU speed or
matching improvement is assumed. The existing v3 control and all defaults remain unchanged.

```sh
uv run catalog-identity pilot-joint --config configs/wdc-products-large.json --validation-report experiments/matcher-v4-validation.json --output runs/joint-timing-reproduction
```

## Frozen semantic pair features: phase 2

After publishing the failed v4 screen, a [separate preregistration](matcher-semantic-preregistration.md)
fixed three candidates on unchanged v3: control, semantic cosine summaries (v5), and summaries
plus symmetric title-vector interactions (v6). The nine CPU fits ran from clean commit
`59ce11f`. [The aggregate](matcher-semantic-screening.json) preserves every outcome;
[the recorded stop decision](matcher-semantic-selection.json) confirms no candidate qualified.

| Candidate | Mean weighted AP | Mean recall at 99% weighted empirical precision | Folds improved over control |
| --- | ---: | ---: | ---: |
| v3 control | 0.91120 | 22.78% | — |
| v5 semantic summaries | 0.91399 | 21.95% | 1 of 3 |
| v6 summaries + interactions | 0.89664 | 8.92% | 0 of 3 |

All three had supported envelopes in all folds. Neither semantic candidate met the fixed
requirements of gains in two folds and at least .02 absolute mean recall gain; v6 also
missed the AP guard. **The semantic additions did not improve the target high-precision
matching objective.** The control remains selected. No representation or model changes
followed these outcomes, and no official-validation experiment was run in phase 2.

Frozen offline MiniLM encoded 4,920 unique field/content records from 2,841 training offers.
All titles were present and within 256 tokens. Descriptions were missing for 702 offers and
truncated for 293 of the 2,139 available descriptions. Missing fields used explicit zero
vectors and availability flags. Precomputation took 72.99 seconds, including 49.12 seconds
of encoding and 6.11 seconds of encoder loading. This run used CPU, batch size 64 and 14
Torch compute/interoperation threads; timings describe this machine, not service latency.
The cache started cold, then served all fold/candidate reuse without additional encoding.

The aggregate records the pinned model revision, 11 actual local asset hashes, dependency
versions, cache/timing counts and per-fold raw/weighted counts, AP, matched entity support,
and 95/98/99% envelopes. The model card declares Apache-2.0; this does not license product
data. No weights, vectors, product text or images are committed, and no model download ran.

Phase 2 reused the already-exposed three training folds. These are development-selection
diagnostics with thresholds selected on their reported populations, not independent 99%
precision guarantees. Control settings, effective parameters, training checksum, v3 feature
source hashes and every fold's AP/envelopes reproduce phase 1 exactly. Whole-source manifests
differ because semantic routing, caching and evaluation were added; this is recorded explicitly.

Across the two registered phases, 21 screening fits were run (12 + 9). Phase 1 additionally
ran one selected/control full-train fit and one official-validation replay; phase 2 ran none.
The prior validation result remains 22 correct matches, zero false matches and 4.4% recall
on seen-entity development data. The held-out test and production/default behavior remain
unchanged. These experiments improve reproducibility and diagnostic capability, not the
measured high-precision matcher performance.

```sh
# Requires the previously cached pinned MiniLM assets; no downloads are permitted here.
uv run catalog-identity screen-semantic --config configs/wdc-products-large.json --output runs/semantic-reproduction --cache runs/semantic-cache
```

## Train-only entity screening of matcher v4

The [preregistered matrix](matcher-v4-preregistration.md) ran twelve fits from clean
commit `5dac477`, using only the official large training file. All four outcomes are in
[matcher-v4-screening.json](matcher-v4-screening.json). No validation or test file was
opened or hashed during screening. This is a separate train-only selection procedure;
it does not make earlier validation-guided feature design blind to the benchmark.

Three deterministic seed-42 entity folds contain 197/148/155 assessment entities and
5,558/3,336/3,531 assessment pairs, with 3,663/2,350/2,458 positives. Their corresponding
training subsets contain 8,858/11,866/11,536 pairs. Boundary exclusions are
5,419/4,633/4,768 pairs. Vocabularies and models fit outside each assessment entity set.
Negative assessment weights are 3 and positive weights 1; raw results are also preserved.
Training restrictions alter class balance, and the folds are not deployment samples.

| Features / leaves / trees | Mean weighted AP | Mean recall at 99% weighted empirical precision | Folds improved over control at 99% |
| --- | ---: | ---: | ---: |
| v3 / 15 / 200, control | 0.91120 | 22.78% | — |
| v4 / 15 / 200 | 0.91888 | 17.94% | 1 of 3 |
| v3 / 31 / 400 | 0.90224 | 20.10% | 0 of 3 |
| v4 / 31 / 400 | 0.91608 | 7.95% | 1 of 3 |

All candidates qualified for supported envelopes in three folds, but **the fixed rule
retained the v3 control**. The v4 additions improved mean ranking AP without improving
high-precision recall. Version 4 remains an experimental implementation, not a promoted
matcher. Five predictions only establishes diagnostic eligibility, not precision reliability.
Each envelope selects its threshold on that same fold; no pooled raw scores or independent
operating-policy claim is made. Per-fold raw/weighted counts, matched entity support, AP,
95/98/99% envelopes, timing, effective parameters and source/data hashes are recorded.

Selection was frozen in commit `e442ff0` before official validation. Since selected and
control are identical, one full-train fit and one full-validation replay served both roles.
[matcher-v4-validation.json](matcher-v4-validation.json) records **22 TP, 0 FP, 478 FN,
4,000 TN**, 4.4% recall, AP 0.62306 and 22 matched positive entities at threshold
0.8902757707668776. The policy subset recovered 13 positives with zero false positives.
This reproduces the existing control; it is not a new matching-quality gain.

The official train/validation audit reports zero shared listings and all 500 shared
entities. This is **seen-entity development**, and full validation reuses calibration and
threshold-selection data. Zero errors among 22 predictions does not establish 99%
precision. No v4 validation fishing, test evaluation, model downloads, additional candidates,
or default/production promotion followed this result. Raw pair evidence stays under ignored
`runs/`; committed records contain aggregates only.

```sh
uv run catalog-identity screen-matchers --config configs/wdc-products-large.json --output runs/screening-reproduction
uv run catalog-identity develop --config configs/wdc-products-large.json --feature-version 3 --learners lightgbm --num-leaves 15 --n-estimators 200 --output runs/selected-control-reproduction
uv run catalog-identity validation-errors --config configs/wdc-products-large.json --model runs/selected-control-reproduction/lightgbm --output runs/selected-control-errors
```

On Windows, replace `uv run catalog-identity` with `.venv\Scripts\python -m catalog_identity.cli`.

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

## Precision development and 100k lexical retrieval

[precision-development-v4.json](precision-development-v4.json) records the follow-up experiments,
including unsuccessful comparisons. **No test file was opened for these experiments.** All
matcher figures below are full-validation diagnostics, including prior calibration and
policy-selection data. They are not independent quality estimates.

The inspected high-scoring validation errors involved quantity differences, memory form
factors, specific model codes, and edition terms. Feature version 3 adds word/bigram similarity,
unshared-word weights and symmetric distinctions for these patterns. Missing information,
one-sided hints and explicit conflicts remain different features; no universal benchmark
veto or invented label was introduced. Existing version 2 artifacts remain loadable.

The initial small-split comparison was mixed:

| Matcher | Correct / false matches | Precision | Recall | AP |
| --- | ---: | ---: | ---: | ---: |
| Previous logistic | 35 / 4 | 89.7% | 7.0% | 0.63302 |
| Specification-feature logistic | 7 / 1 | 87.5% | 1.4% | 0.70510 |
| Previous LightGBM | 20 / 0 | 100.0% | 4.0% | 0.67236 |
| Specification-feature LightGBM | 27 / 1 | 96.4% | 5.4% | 0.73130 |

Stronger ranking AP alone did not produce a uniformly better precision operating point.
The next comparison used the official large train/validation files already present in the
local archive: 19,835 training pairs (8,471 positive) and 4,500 validation pairs (500 positive).
The large validation partition has 1,289 calibration pairs, 1,178 policy pairs and 2,033
excluded cross-partition pairs. The published files were not reshuffled or mixed across sizes.

| Large-split matcher | Correct / false matches | Precision | Recall | AP |
| --- | ---: | ---: | ---: | ---: |
| Previous LightGBM features | 25 / 4 | 86.2% | 5.0% | 0.52956 |
| Specification features | 22 / 0 | 100.0% | 4.4% | 0.62306 |

The version 3 LightGBM threshold is 0.8902757707668776, selected on the policy subset using
the same weighted 99% empirical target. It recovered 13 policy positives without false
positives; the full-validation replay recovered 22. This is a useful observed precision/recall
tradeoff, but zero errors among 22 predictions does not establish 99% production precision.
This model remains an inspection candidate. The previous frozen test result is unchanged.

```sh
uv run catalog-identity develop --config configs/wdc-products-large.json --feature-version 2 --output runs/large-previous
uv run catalog-identity develop --config configs/wdc-products-large.json --feature-version 3 --output runs/large-specifications
uv run catalog-identity validation-errors --config configs/wdc-products-large.json --model runs/large-specifications/lightgbm --output runs/large-errors
```

Use `configs/wdc-products.example.json` for the small-split comparison. Error exports contain
pair titles, feature evidence and calibration/policy/cross-partition membership, and stay in
ignored local run directories. The committed aggregate excludes product text and PR curves.

The lexical ranking implementation now selects a bounded text candidate pool in NumPy before
building Python evidence objects. Identifier priority and deterministic ties are preserved;
tests compare outputs with exhaustive ranking. The API caches indexes by database revision
and rejects stale results when another process writes during retrieval. Pair inspection loads
only the requested pair rather than the entire catalog.

| Synthetic listings | Build incl. vocabulary | Query p50 / p95 | Queries/sec | Sampled RSS |
| --- | ---: | ---: | ---: | ---: |
| 1,000 | 0.073 s | 1.4 / 1.5 ms | 725.6 | 142.2 MiB |
| 10,000 | 0.728 s | 8.6 / 18.9 ms | 101.1 | 199.6 MiB |
| 100,000 | 10.738 s | 99.2 / 132.0 ms | 9.7 | 742.1 MiB |

This run used clean commit `fd6a2ad`, seed 42, k=50 and 100 queries per size. Compared with
the earlier 10k run, median lexical query time fell from 132.9 to 8.6 ms on this machine.
Memory remains sampled process RSS, not an exact peak. This is a lexical retriever measurement
on synthetic data; it excludes network requests, persistence, clustering and embedding
inference. No real-corpus accuracy or full-service 100k claim follows from it.
