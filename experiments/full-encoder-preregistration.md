# Final conventional full fine-tuning comparison

Registered after publication of the unsuccessful partial-freeze joint model (`72238bf`),
before extracting new benchmark accuracy or running its timing pilot. Discovering the
local GPU removed the original CPU constraint. This is one conventional configuration,
not an ablation that identifies why the previous custom configuration underperformed.
No further architecture candidates follow this comparison if it fails.

## Fixed representation and training

- Cached `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, loaded offline with actual asset hashes.
- Train embeddings and all six BERT layers. Remove the unused BERT pooler.
- Final CLS vector -> dropout 0.1 -> one binary linear classifier, initialized with
  normal standard deviation 0.02 and zero bias. No numeric features, hidden head,
  mean pooling, ensemble or fitted vocabulary/scaler.
- Reuse the exact 256-token balanced pair serialization, title/brand limits and field
  labels. Seed-derived pair orientation reverses each epoch; inference averages the
  two orientation logits before sigmoid.
- Seed 42, exactly three epochs, final checkpoint only, every retained training pair.
  Unweighted BCEWithLogits; AdamW at uniform 2e-5, weight decay 0.01 except zero decay
  for biases and LayerNorm parameters. Linear 10% warmup then linear decay, gradient
  clipping at norm 1, microbatch 8 and accumulation 4 (effective batch 32).
- Same explicit deterministic CUDA float32 execution, MATH attention, no autocast/TF32,
  non-fused/non-foreach AdamW. Original CPU environment remains unchanged.

## Prospective resource gate

One timing-only pilot uses the existing deterministic training-text length sample and
synthetic alternating labels: four warmup and sixteen measured microbatches, actual
all-layer backward, synchronized transfers and two-order inference. Pilot weights are
discarded. Record GPU allocated/reserved memory, CPU RSS, source/asset/environment hashes,
token lengths and full projections including preparation, audits and serialization.
Use safety factor 1.5. Require projected three-fold screening <=30 minutes and conditional
full fit/calibration/policy/replay <=20 minutes. Stop on failed gate, OOM, nonfinite values
or deterministic execution errors. No smaller batch, shorter sequence, changed precision,
epochs, seed or training schedule may rescue a failed gate.

## Training-only optimization audit

For every fixed epoch record example-weighted training loss, actual optimizer steps,
learning-rate range, finite pre-clipping gradient norms, effective batch sizes, final
partial accumulation and counts of each orientation. Record class counts and nonzero
parameter changes in embeddings, an early layer, a late layer and classifier. Verify
all intended backbone parameters are trainable and receive gradients; exclude the unused
pooler. These diagnostics cannot select checkpoints or alter any setting. Assessment
occurs only after all three epochs. Training loss is not a benchmark accuracy claim.

## Single candidate and fixed selection

Three seed-42 entity folds, same retained/boundary pairs, negative assessment weight 3
and positive weight 1. Compare the saved v3-small control only after verifying training
hash, fold definition, source hashes and settings. There are three new accuracy fits,
one candidate, no repeated control fit, hyperparameter search or assessment-driven retry.

Qualify only if the 99% weighted empirical-precision envelope has at least five matches
in every fold, recall improves in at least two folds, mean recall improves by at least
0.02 absolute, and mean weighted AP is at least 0.99 times the control's. Per-fold raw
and weighted counts, matched entities and 95/98/99% envelopes remain visible. Thresholds
are selected on their reported population; these reused development folds and small
zero-error supports do not establish independent performance or precision guarantees.

If qualified, freeze selection before one full-training fit, the existing validation
calibration/policy partitions and one complete seen-entity validation diagnostic replay.
Otherwise stop, preserve the current matcher and publish the unmet accuracy goal. Test
data is never loaded or hashed by this phase; any later evaluation requires a separately
registered frozen-artifact protocol. No production/default/hierarchy promotion is automatic.

## Verification before timing

Fictional fixtures cover all-layer gradient flow, deliberate learnability, optimizer groups,
fixed epochs and audits, pair symmetry, safe self-contained artifact roundtrip, CPU default
loading, malformed metadata/tensors, lazy optional dependencies and untouched legacy models.
Train-only pilot/screening must tolerate nonexistent validation/test paths and reject changed
execution/source/assets. Run full core tests, relevant CUDA tests, Ruff lint/format, and a
real offline fictional CUDA smoke before the benchmark timing pilot. Source changes after
timing require a new disclosed pilot before accuracy.
