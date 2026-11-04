# What the matcher needs next

The training audit found a data annotation priority, not a new accuracy gain. Preserve
the existing matcher. No validation/test file was opened or hashed, no new candidate
was fitted, and no benchmark label was changed. Three fits exactly reproduced the
saved control's AP and confusion counts at its existing thresholds.

## Measured findings

[Aggregate report](training-input-audit.json), produced from clean commit `26f0994`,
covers 19,835 official training pairs: 8,471 positives and 11,364 negatives, drawn from
2,841 offers and 500 benchmark entities. Of the negatives, 8,523 carry the supplied
hard-negative flag. The data already contains difficult negatives; simply requesting
more unspecified negatives is not a diagnosis.

- No opposite-label collisions or repeated pair inputs were found in raw full text,
  normalized titles, normalized full text, untruncated tokens or symmetric 256-token
  inputs. All 2,841 normalized listing titles were distinct. This excludes one exact
  collision mechanism; it does **not** prove semantic distinguishability or clean labels.
- Brand is present in 1,011 offers (35.6%); description in 2,139 (75.3%). Structured
  model/MPN, GTIN, category and attributes are absent from every offer. The source
  itself contains only ID, title, brand, description, price and currency per offer;
  these missing fields are not sitting unused in this source adapter.
- 5,792 pairs (29.2%) lose some text under the registered token budget. No title is
  truncated; the affected fields are description and brand. No new mixed-label input
  collision appears. Distinguishing evidence could still be lost without creating
  identical inputs, so this does not prove truncation harmless.
- Each entity has 3–11 offers. Positive pair counts per entity range from 3 to 55
  (median 10). Pair counts substantially overstate independent product coverage.

The frozen control diagnostic covers all 8,471 positives and 3,954 negatives. Another
7,410 negatives cross fold boundaries and are not scored by that protocol. It yields
1,759 true matches, 5 false matches and 6,712 missed positives across the three fixed
policies. These are diagnostic case counts; the previously reported mean fold recall
of 22.78% is not the ratio of these aggregated counts. No scores were pooled or thresholds
selected again. These reused development results do not establish target-domain accuracy.

| Training stratum | Positive pairs | Missed by control | Distinct entities |
| --- | ---: | ---: | ---: |
| Both titles have code hints, with no shared hint | 904 | 903 | 148 |
| Title token Jaccard <= 0.2 | 1,373 | 1,214 | 223 |
| At least one description missing | 3,722 | 2,990 | 340 |
| Some text truncated by the neural serializer | 2,470 | 1,966 | 312 |

Strata overlap; do not add their counts. Code-disagreement positives are the clearest
targeted review priority. This is an association with errors, not proof that changing
the code feature will recover those matches safely. The truncation row describes a
subset of LightGBM errors; LightGBM does not use the neural token budget.

An analyst inspection of the five false positives and the 20 sampled code-disagreement
positives found the following review themes. These are unadjudicated observations of
the available text, not human ground truth or population frequencies:

- Code hints can refer to measurements, a series, compatibility, a partial code or a
  differently formatted identifier. Some positive pairs also show apparently different
  model names; distinguishing aliases, listing mistakes and label problems needs source
  evidence. Do not strip punctuation/suffixes universally or rewrite labels.
- The false-positive cases contain differences in product type, manufacturer, bundle or
  lighting configuration, or focal range. A shared series/code is insufficient evidence.
- Some text uses different languages or shortened descriptions. No language classifier,
  multilingual error rate or translation-based identity judgment was measured.

## Review the existing evidence first

The local packet at `runs/training-audit-v1/review-blind.jsonl` contains **145 cases across
140 entities**, at most two cases per entity. It includes all five control false positives,
20 control false negatives, 20 cases from each remaining populated diagnostic stratum,
and 20 reference cases. Selection is hash-deterministic, with explicit priority and
shortfalls in the report. The reference sample comes from remaining eligible pairs after
the entity cap. This targeted packet cannot estimate population error rates.

Give separate copies of the blind packet to two independent reviewers. Keep
`review-key.jsonl` away from both until their decisions are locked. It contains the
original labels, entity membership, reason tags and scores. These local files are ignored;
do not commit or redistribute listing text. The packet does not download images or pages.

For each case record:

1. `input_sufficiency`: sufficient / insufficient / conflicting / needs_source_check.
2. `benchmark_identity`: same / different / uncertain, under the documented benchmark
   semantics. Keep the original benchmark label immutable, even if reviewers disagree.
3. `family_relation` and `variant_relation`: same / different / uncertain, separately.
   Same variant requires same family; do not infer either hierarchy label from a benchmark
   label. Use uncertain when category-dependent variant evidence is absent.
4. `evidence_fields` and `reason`: cite the actual observation and its role: own-product
   model/MPN, series, compatible-device code, measurement, pack/bundle, variant attribute,
   manufacturer, or product type. Record absent evidence explicitly. Add source URL,
   observation date and provenance only when verified through permitted sources.

Record reviewer identifiers and annotation-guideline version outside the blind inputs.
Measure disagreement and insufficient-evidence rates by review stratum. Adjudicate
disagreements with a third review and documented evidence. Treat these as new annotations
with provenance, not corrected official benchmark labels. This audit's analyst inspection
is not an independent reviewer judgment. No human annotations have been collected yet.

## Data request, in priority order

| Need | Required evidence | Why collect it |
| --- | --- | --- |
| Identifier meaning and aliases | Token spans and role labels; verified brand-scoped MPN/GTIN or documented alias links; raw and canonical values | Separate genuine contradictions from measurements, compatibility and alternate code formats |
| Near-neighbor distinctions | Product type/category, actual manufacturer, bundle/pack, capacity, dimensions, generation and applicable variant axes | Learn when similar titles or shared series still identify different products |
| Low-overlap positives | Independently verified identity links across retailer wording/languages; complete distinguishing specifications | Recover same-entity pairs that lexical overlap misses |
| Source sufficiency | Licensed feed/page provenance, timestamps, missing/uncertain/conflicting markers | Determine whether a case is answerable from the inputs available at serving time |
| Separate hierarchy truth | Independently reviewed family and exact-variant identities, with category rules | Avoid treating WDC benchmark entity labels as family/variant ground truth |

Do not fill GTIN/MPN from gold cluster IDs, guessed title matches or model predictions.
Enrichment must be available independently of the label and at serving time. Authoritative
source evidence can show that the listing text is wrong rather than the benchmark label;
keep these alternatives separate. Document feed rights and attribution before collecting
or redistributing it. Product images are unnecessary for this next step.

As an initial **proposed annotation budget**, collect 300 additional candidate pairs from
at least 300 previously unexamined product entities, approximately 60 per observable theme:
possible code aliases, measurement/compatibility code confusion, shared-series product-type
differences, low-overlap wording, and close configurations. Quotas describe candidate cues,
not presumed labels. Retain uncertain results and record sampling probabilities/shortfalls.
Keep both verified matches and verified nonmatches in each applicable theme; never invent
labels to balance a quota. This pilot is for data development, not precision certification.

Use permitted catalogs with provenance and multiple independent offers per product. Prefer
more distinct products over more combinations of the same few offers. Form entity/alias
components before splitting. Reserve disjoint product components for training, calibration
and final confirmation; check identifiers and near-duplicate text across partitions. Any
overlap audit involving an existing benchmark test set must be a separately frozen,
read-only protocol, not an opportunity to inspect its errors. No such audit ran here.

## The next measurable experiment

After the diagnostic review and evidence collection, register one data-enrichment comparison
before viewing the new confirmation labels. Keep model capacity fixed: compare the existing
control inputs against verified role/attribute-enriched inputs using the same training-label
budget. Annotate/enrich inputs blind to pair labels. Fit vocabularies, extraction models and
calibration only on their assigned partitions. Separate manually supplied evidence from
automatically extractable evidence and report its coverage and annotation cost.

Select operating thresholds on calibration data, then freeze models, data, thresholds and
the evaluation command. On a fresh entity-disjoint confirmation population measure:

- Precision and raw false merges at the fixed policy; uncertainty with independent entity
  support. Also report abstentions, unsupported fields and review volume.
- Paired recall gain over the same frozen control. A proposed practical target is at least
  **5 percentage points absolute recall gain while retaining 99% precision**, with uncertainty
  on the gain. This is a future acceptance target, not an achieved result.
- Recall and false merges by the prespecified evidence themes, with counts, plus AP and
  calibration as secondary diagnostics. Challenge-set results and representative production
  results remain separate; oversampled hard cases do not estimate production precision.

Plan support before evaluation. For illustration, with independent representative accepted
decisions, zero errors among 299 decisions gives a one-sided 95% binomial precision lower
bound just above 99% (`0.05 ** (1 / 299)`). That calculation does not apply directly to
correlated pair combinations or a targeted challenge sample. Predeclare sample size; do
not keep collecting until an interval happens to pass. If support is insufficient, report
that limitation and keep the model experimental. The 145-case diagnostic packet and
300-case collection pilot do not satisfy this requirement by themselves.

No larger-model or longer-context run follows from this audit alone. The next gain must be
demonstrated from better evidence on new products, rather than inferred from missing fields.

## Reproduce

Install the documented `models` extra for control replay and `semantic` extra for the
optional cached-tokenizer audit. With the official training file already available:

```sh
uv run catalog-identity audit-training --config configs/wdc-products-large.json --output runs/training-audit-reproduction --tokenizer-directory "/path/to/cached/1110a243fdf4706b3f48f1d95db1a4f5529b4d41" --control-report experiments/matcher-v4-screening.json
```

The tokenizer is local-only; no neural weights are loaded or trained. Unbounded token
inspection can emit a tokenizer sequence-length warning; actual encoded pairs remain
256 tokens, and no overlength sequence is passed through a neural model. Omit both optional
arguments for a core-dependency text-only audit. An existing output directory is rejected.
Only the aggregate `audit.json` is suitable for committing; review files stay local.
