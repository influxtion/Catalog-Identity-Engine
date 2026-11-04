# Phase 2: frozen semantic pair features

Registered separately after the published negative v4 screening. No semantic feature
extraction or new fits preceded this registration. Phase 1 outcomes and selection
remain unchanged. This phase reuses already-exposed training folds, so selection
results are development evidence, not a new blind assessment.

## Fixed representation and matrix

Use only the cached `sentence-transformers/all-MiniLM-L6-v2` at immutable revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, offline on CPU. Its cached model card
declares Apache-2.0, English language, 384 dimensions and truncation at 256 tokens.
Encode normalized titles and descriptions separately, with that fixed truncation.
Missing fields are zero vectors with explicit availability, never empty-string
embeddings. No model fine-tuning, embedding centering or corpus-fitted transforms.

Exactly three candidates, all LightGBM 15 leaves / 200 trees:

1. Existing v3 control, unchanged.
2. v5: v3 plus title-title and description-description cosine, min/max of the two
   cross-title/opposite-description cosines, and availability/missingness indicators.
3. v6: v5 plus 384 absolute differences and 384 elementwise products of normalized
   title vectors. These interactions are symmetric.

All other parameters remain learning rate .05, minimum child samples 20, L2 1,
seed 42, deterministic column-wise fitting and one CPU training thread. Use the
same three SHA256(seed:entity) modulo-3 folds, pair exclusions, assessment weights
(positive 1, negative 3), and minimum five predictions as phase 1: nine fits, no
hyperparameter search. Report all candidates, per-fold weighted/raw AP and
95/98/99% precision envelopes, counts and matched positive entity support.

Frozen embeddings may be precomputed on all training offers and shared between
folds because they fit no dataset statistics or labels. TF-IDF vocabularies and
LightGBM remain fitted on each training fold only. Screening must never load,
hash or encode official validation/test data. One encoder instance and a validated
content-addressed numeric cache avoid repeated encoding; cache remains ignored.
Record encoder revision, configuration/tokenizer/weight hashes, preprocessing,
dependency versions, dimensions, token limit, missing/truncation counts and timing.

## Fixed selection and stopping rule

A semantic candidate qualifies only if all of the following hold:

- A supported 99% weighted empirical precision envelope in all three folds.
- Greater 99% envelope recall than control in at least two folds.
- Mean fold recall exceeds control by at least .02 absolute.
- Mean weighted AP is at least .99 times control mean weighted AP.

Among qualifiers choose highest mean recall, then mean weighted AP, then the
simpler v5 summaries. Otherwise retain control. Thresholds are selected on the
reported fold, not independent operating policies or established 99% guarantees.
Do not pool separately fitted raw scores into one threshold. No changes to fields,
dimensions, model capacity, folds or selection after outcomes are inspected.

If a semantic candidate qualifies, freeze it before official validation, fit it
once on full training, run existing calibration/policy selection and one full
validation replay. Compare the already-recorded phase 1 control after checking
input hashes and unchanged v3 implementation/effective settings; disclose source
provenance differences. Do not redundantly refit the full-train control. Official
validation shares all 500 training entities and reuses calibration/policy data.

If neither semantic candidate qualifies, stop without official-validation runs.
No new downloads, held-out test access, default promotion, hierarchy changes or
additional unregistered candidates. Publish failure honestly if this also fails.
