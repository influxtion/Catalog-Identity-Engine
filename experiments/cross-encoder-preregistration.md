# Phase 3: one supervised joint-text architecture, gated by measured CPU cost

Registered after the negative alignment and frozen-semantic experiments, before
pilot timing or joint-model accuracy inspection. No new weights, test access,
external credentials or paid compute. No further candidate additions in this phase.

## Architecture and inputs

Use the cached six-layer MiniLM BERT backbone at immutable revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, offline, without sentence pooling.
Freeze embeddings and layers 0–3, keeping those modules in evaluation mode; train
layers 4–5. Concatenate final CLS with unchanged v3 numeric features standardized
on the fitting fold only. Head: linear to 128, GELU, dropout .1, linear to one logit.

Only normalized title, optional brand and description enter tokenization, with
explicit field labels. No listing/entity/pair IDs, labels or hard-negative flags.
The 256-token pair limit accounts for tokenizer pair-special tokens and field
labels. Give both listings balanced budgets, title up to 96 tokens, brand up to
8, then description; redistribute unused side budget without losing either
listing's title. Record truncation. Use tokenizer longest-first as a final guard.
Seed-derived training orientation reverses every epoch. Inference averages the
two orientation logits before sigmoid, ensuring symmetric final scores.

## Fixed training

Seed 42, three epochs, final checkpoint only. Reset original cached backbone for
each fold. Train all retained pairs with unweighted BCEWithLogits. AdamW uses
backbone learning rate 2e-5, head 1e-3, weight decay .01; 10% linear warmup followed
by linear decay. Clip gradient norm at 1. Float32 CPU, 14 Torch threads, microbatch
8 and accumulation 4 (effective batch 32). Only a measured memory failure permits
microbatch 4 / accumulation 8; no other capacity, length, epoch or sampling changes.
No early stopping, assessment-selected checkpoints, resampling or hyperparameter search.

## Timing-only resource gate

Before accuracy work, use a deterministic training-only text sample with synthetic
alternating labels solely to exercise computation. Run warmup then 16 measured
forward/backward microbatches, plus separately timed two-orientation inference.
Inspect no accuracy and save no pilot model. Record actual token lengths, throughput,
sampled process RSS, thread counts and tokenizer/weight/config hashes. Project the
actual three-fold training counts and conditional full-train counts, including
three epochs, symmetric inference, vocabulary/features and a 1.5 safety factor.
The gate requires screening <=90 minutes AND conditional full fitting <=45 minutes.
If either fails, stop without accuracy training. Do not shorten training to fit;
report a GPU or separately revised compute plan as a future requirement.

## Screening, selection and stopping

If the gate passes, fit one candidate across the same three seed-42 entity folds
as prior phases. Fit vocabularies/scalers only on each training fold; assess only
within held-fold pairs, exclude boundary pairs. Assessment positives weight 1 and
negatives 3. Compare saved v3 control after verifying training hash, fold counts,
settings and relevant v3 source hashes. Do not rerun the control. Report all three
complete folds, raw/weighted AP, 95/98/99% envelopes, counts and matched entities.
Partial results from a resource-aborted run are not a completed accuracy result.

Qualification requires supported 99% envelopes in all folds (minimum five matches),
greater recall than control in at least two folds, mean recall >=control+.02 absolute,
and mean weighted AP >=.99*control. These reused folds and within-fold threshold
envelopes are development diagnostics, not independent precision guarantees.

If qualified, freeze selection and fit once on full training, then use existing
calibration/policy partitions and exactly one official-validation diagnostic replay.
Official validation shares all 500 training entities and reuses policy/calibration
observations. No test data, default changes or hierarchical identity promotion.
If screening fails, publish the negative result and stop. Safe model artifacts use
safetensors and JSON metadata/vocabulary/scaler plus tokenizer files; no pickle.
