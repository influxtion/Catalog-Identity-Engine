# Baseline evaluation protocol

## Data boundaries

One run uses exactly three official pair files, explicitly assigned to train, validation,
and test. It never splits or samples these rows. Repeated offers within a split must agree;
duplicate pairs, inconsistent gold metadata, and shared offer IDs across splits are errors.
Entity IDs and labels are held outside listing features and retrieval inputs.

The split audit reports shared entities and exact normalized title/brand/model signatures.
Text overlap alone does not establish identity. Approximate duplicate detection and manual
label review are not implemented; official entity IDs are the basis of the unseen-entity
audit. This does not establish absence of every possible content leak or annotation error.

The declared WDC protocol is supplied by the operator and recorded; the runner does not
authenticate downloaded files against a publisher checksum registry. The file hashes make
the inputs identifiable. Never label arbitrary pair files as an official benchmark run.

## Retrieval

Training offers fit a character 3–5 gram TF-IDF vocabulary over title, brand and model.
Validation and test indexes only transform their records. Each split's retrieval population
is all unique offers on both sides of its pair file. There is no artificial left/right store
restriction: the benchmark adapter does not have trustworthy store IDs.

Validated GTIN lookup and known brand+model lookup contribute candidates ahead of text-only
hits. Cosine score and stable listing ID break ties. Channels are recorded and duplicates
removed; self hits and zero-similarity text-only hits are excluded. The total union is bounded
by k. A repeated identifier is retrieval evidence, not permission to merge.

Labeled-pair recall counts directed positive links from the supplied pair rows. Entity recall
uses all directed same-entity links implied by complete gold metadata. Both include misses
and queries with no candidates. Neither metric measures full-catalog precision.

Build time excludes vocabulary fitting, which is reported separately. Query p50/p95 measure
single-process query work, excluding output serialization. Matrix bytes are not peak RAM.

## Pair classification

A standardized logistic regression uses token and numeric overlap, title length, brand/model/
GTIN/category agreement/conflict/missingness, shared attributes, and comparable prices.
Training includes all official training pairs, including published hard negatives. Gold
cluster IDs, source IDs, pair IDs and hard-negative flags never become model features.

The model has benchmark-entity semantics. It cannot infer a family-versus-variant label from
WDC's binary labels. Conflicts are observations; no category-independent color or capacity
veto is added to this benchmark classifier.

Validation selects a match threshold at an empirical precision target (default 0.99), with
at least five predicted matches. Recall breaks the primary tradeoff; equal-recall choices
retain the higher threshold. Tied scores enter together. An unattainable target yields no
automatic matches, with precision reported as null rather than a misleading perfect score.
Small support does not imply a statistically reliable precision guarantee.

The rejection threshold is fixed in configuration, not learned or validated as a safety
guarantee. Review is a third operational action. For binary precision/recall/F1, anything
not automatically matched counts as negative, including review. No probability calibration
is performed; Brier score, average precision, PR curve, and reliability bins expose this.

Test probabilities are computed only after fitting and validation threshold selection.
Use validation errors for model development. Repeatedly optimizing against test results
would invalidate held-out evaluation even though the code preserves its split boundary.

## Inspect errors

`validation-pairs.jsonl` and `test-pairs.jsonl` include false positives and false negatives
at the selected match threshold, hard-negative metadata, feature values, observed conflicts,
and retrieval positions. Rank null means absent from the largest requested retrieval cutoff.
The report separately breaks down errors involving structured conflicts and hard negatives.

The retrieval-plus-matching measure evaluates only labeled pair rows. It counts a positive
decision only if either query direction retrieves the pair at `pipeline_k`. Unjudged
candidates are not assigned negative labels; this measure is not end-to-end catalog
precision or clustering quality.

`model.json` contains numeric parameters, avoiding executable model serialization. Apply
`(features - mean) / scale`, then the saved linear coefficients and sigmoid to reproduce
probabilities; the automated round-trip test verifies this. Full reconstruction also needs
the recorded feature and normalization implementation. Run manifests hash that source and
record environment versions, input hashes, settings, and Git state.
