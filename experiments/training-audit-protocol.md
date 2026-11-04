# Training input audit

This branch audits the official large training split only. It does not open or hash
validation/test files, fit a new candidate, change labels, or promote a matcher.

Measure field availability; repeated representations and cross-entity collisions;
opposite labels for identical unordered pair inputs; loss of distinguishing text
after the registered 256-token serialization; and title-overlap/code-evidence strata.
Compare raw title/brand/description, normalized title, normalized full text, and the
actual symmetric token inputs. Representation collisions are evidence limitations,
not proof of incorrect labels. Minimum errors on conflicting observed groups are
training-sample bounds for that representation, not population accuracy ceilings.

Optionally reproduce the three saved v3-small control fits, with identical training
data, sources, folds and settings. Use the previously saved thresholds; verify every
fold's AP and confusion counts before publishing error strata. These are reused,
optimistic development diagnostics. Boundary negatives are excluded by the original
fold protocol; report coverage and do not pool scores or select new thresholds.

Produce an aggregate public report and a separate local review packet. Review strata
are exact input collisions, high-overlap negatives (title token Jaccard >= .8),
low-overlap positives (<= .2), positives with disagreeing code hints, positives with
missing descriptions, truncated positives, and control false positives/negatives.
Include a hash-selected reference sample. Strata can overlap; sample deterministically
with seed 42, a fixed quota of 20 each, and at most two selected pairs per entity
across the packet. Record shortfalls and selection provenance. This is targeted review,
not a representative performance sample. Keep labels/reasons/scores in a separate key;
the reviewer packet has no gold labels, scores, entity IDs or reason tags.

Do not redistribute listing text. Commit aggregates, code and instructions only.
Do not infer language, correct labels, or family/variant identity from lexical hints.
Use findings to specify the evidence, independent annotation and evaluation required
for a future measured gain. Tests cover split isolation, collision bounds, truncation,
missing fields, deterministic sampling, reviewer blinding and control reproduction.
