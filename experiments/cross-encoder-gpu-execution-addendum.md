# Execution-only GPU addendum to phase 3

The published CPU timing gate failed and remains unchanged. A physical NVIDIA
GeForce RTX 4060 Laptop GPU (8,188 MiB reported by nvidia-smi) was subsequently
identified; the original environment had a CPU-only Torch build. This addendum
changes execution hardware only, not the registered architecture or accuracy candidate.

Use a separate ignored `.venv-gpu` with Python 3.12.10, exact versions of the 69
existing non-Torch packages, and official `torch==2.15.0+cu132` for Windows/Python 3.12.
Keep `.venv`, global Python, drivers and existing model assets untouched. Record
dependency snapshots, install reports, official wheel URL/hash, CUDA/driver versions
and hardware. No new model weights or paid/external compute.

All architecture, text/numeric inputs, balanced 256-token budget, three epochs,
seed 42, layers/head, learning rates, regularization, optimizer schedule, gradient
clipping, microbatch 8 / accumulation 4, folds, weighting, thresholds, qualification
and stopping rules remain exactly as registered. No autocast, mixed precision,
TF32, batch-size optimization or alternate attention after timing.

Execution is explicitly selected (`cpu` remains default), with CPU initialization
then float32 CUDA transfer. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA
initialization; seed Torch and all CUDA devices with 42. Require deterministic
algorithms without warn-only fallback, deterministic cuDNN with benchmarking off,
IEEE float32 precision using Torch 2.15's new precision APIs only, and MATH SDPA.
Use AdamW foreach=False/fused=False consistently. Record effective flags and a
versioned execution/environment fingerprint. CPU/CUDA rounding and dropout streams
differ; no cross-device bitwise-reproducibility claim is made.

The GPU timing pilot uses the same training-only text sampling, synthetic labels,
four warmup plus 16 measured microbatches and separate two-orientation inference.
Synchronize CUDA around measured intervals, including batch/label transfer cost.
Reset GPU peaks after warmup; report allocated/reserved VRAM separately from sampled
CPU RSS, plus GPU name/capability/total/free memory. Keep the same 1.5 safety factor
and 90-minute screening / 45-minute conditional full-fit gates.

A passing timing artifact must match execution fingerprint/device, source hashes,
training data hash, architecture and training constants. A CPU gate cannot authorize
GPU execution. Code correctness fixes require a new pilot before accuracy work.
Stop on CUDA unavailability, nonfinite values, deterministic-operation failure,
OOM or a failed gate. No silent CPU fallback, smaller batch or new candidate.

Before the benchmark pilot, run fictional CUDA training and same-device artifact
roundtrip, plus default CPU artifact loading and symmetric finite predictions with
predeclared 1e-5 float32 tolerance. Artifacts store CPU safetensors and a strictly
validated optional execution manifest; legacy artifacts without recorded flags remain
loadable without inventing historical settings. Default artifact inference is CPU.

If the GPU gate passes, run only the one registered three-fold joint-model accuracy
comparison. If it qualifies, perform the already-registered conditional full fit and
one validation replay. No held-out test access, default promotion or hierarchy changes.

Official runtime sources: [CUDA wheel index](https://download.pytorch.org/whl/cu132/torch/),
[Torch 2.15 CUDA precision](https://docs.pytorch.org/docs/2.15/notes/cuda.html), and
[Torch 2.15 reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html).
