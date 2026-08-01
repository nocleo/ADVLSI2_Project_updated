# Model Improvement Roadmap

## Goal

Develop a reproducible Metal-1 `m1.2` violation detector that generalizes to
unseen layouts and produces calibrated tile-level detections suitable for the
project's localization and GDS-reporting flow.

The roadmap follows one rule: **change one experimental factor at a time and
compare it with the last accepted benchmark on the same frozen evaluation
split**. Every phase should be a separate PR with its configuration, metrics,
and conclusion recorded.

## Metrics and experiment rules

Dirty tiles are the positive class. Accuracy alone is insufficient because a
one-class predictor can appear competitive on an imbalanced split. Every
benchmark must record:

- dirty-class precision, recall, and F1;
- confusion matrix and predicted-class counts;
- training and validation loss by epoch;
- best checkpoint epoch and test metrics from that checkpoint;
- seed, dataset manifest/hash, split strategy, model configuration, and runtime
  environment;
- per-layout results once layout-aware splitting is available.

Primary model-selection metric: dirty-class F1 on validation data. Recall is a
hard safety metric; a candidate with zero dirty recall is rejected. The test
split is evaluated only after configuration selection, not used to tune the
model.

## Phase plan

### B0 — Reproducible functional baseline (complete)

**Purpose:** prove that the end-to-end classifier training path learns and can
be rerun from a clean clone.

**Scope:** original `NCSU_DRCNN`, fixed seed, deterministic validation/test,
train-only 90-degree rotations and reflections, best-validation-loss checkpoint,
and complete binary metrics.

**Acceptance gate:**

- dependency installation, smoke verification, and the five-epoch full-dataset
  run complete without errors;
- training loss falls by at least 5% from its first-epoch value;
- validation and test predictions contain clean and dirty samples;
- dirty recall is greater than zero;
- command, environment, and JSON metrics are attached to the PR.

If B0 still collapses, diagnose it inside B0 with controlled no-augmentation,
optimizer, and batch-level checks. Do not start architecture tuning first.

**Accepted result:** the five-epoch CPU run with seed 42 selected epoch 5 and
reached test accuracy 79.21%, dirty precision 0.779, dirty recall 0.795, and
dirty F1 0.786. Predictions included both classes. This result establishes
functionality only; B1 replaces the tile-random evaluation with a leakage-free,
layout-grouped benchmark before model optimization begins.

### B1 — Dataset integrity and leakage-free evaluation

**Hypothesis:** duplicated clean tiles and tile-level random splitting inflate or
destabilize evaluation and do not measure generalization to new layouts.

**Work:**

1. Build a manifest containing path, label, source layout, coordinates, shape,
   dtype, density, and content hash.
2. Detect exact duplicates and contradictory labels before splitting.
3. Deduplicate before split or keep duplicate groups in exactly one split.
4. Replace random tile splitting with a fixed group split by source layout.
5. Report class balance and per-layout metrics; add visual label-audit samples.

**Acceptance gate:** no content hash crosses splits, no unresolved contradictory
labels remain, and repeated runs with the same manifest and seed reproduce the
same split and metrics.

### B2 — Training and optimization baseline

**Hypothesis:** optimizer, learning rate, regularization, and class sampling can
improve learning stability once the evaluation protocol is trustworthy.

**Work:** compare one variable per run, beginning with optimizer/learning-rate
sweeps, then scheduler, weight decay, batch size, and balanced sampling or class
weights if B1 demonstrates imbalance. Add early stopping and machine-readable
experiment summaries.

**Acceptance gate:** the selected configuration improves validation dirty F1
across multiple fixed seeds without reducing dirty recall or degrading held-out
layout performance. Record mean and variation, not only the best run.

### B3 — Model architecture experiments

**Hypothesis:** a compact modern CNN can outperform the original paper-style
network while retaining inexpensive inference.

**Work:** establish a simple architecture candidate with normalization and
global pooling, then compare capacity changes or a small residual model. Keep
input tiles, B1 splits, and B2 training configuration fixed. Track parameter
count, checkpoint size, and CPU/ONNX latency alongside quality metrics.

**Acceptance gate:** statistically consistent F1/recall improvement on held-out
layouts with an acceptable inference-cost increase. Reject complexity that only
improves the training layouts.

### B4 — Error analysis and targeted data improvement

**Hypothesis:** performance is limited by identifiable layout patterns, boundary
labels, metal-density regimes, or synthetic-error coverage rather than model
capacity alone.

**Work:** review false positives/negatives by source layout, density, geometry,
and distance from tile boundaries. Correct label-generation defects and add
targeted examples or hard-negative mining only where the analysis supports it.
Version the resulting dataset and manifest.

**Acceptance gate:** the targeted dataset change improves its intended error
slice and overall held-out-layout metrics without introducing leakage or a
material regression elsewhere.

### B5 — Calibration and end-to-end layout evaluation

**Hypothesis:** a threshold selected on validation layouts plus spatial merging
can turn tile probabilities into useful layout-level detections.

**Work:** calibrate the probability threshold on validation data, measure
precision-recall trade-offs, and evaluate NMS/merging independently. Compare CNN
detections with KLayout ground truth on untouched layouts using spatial matching,
not only tile labels. Measure layout-level recall, false detections, and latency.

**Acceptance gate:** a frozen threshold meets an explicitly chosen recall target
on unseen layouts, with documented false-positive rate and end-to-end runtime.

### B6 — Reproducible release candidate

**Purpose:** make the selected model easy to reproduce and safe to evaluate.

**Work:** freeze dataset/model versions, export and numerically compare PyTorch
and ONNX outputs, add automated smoke/regression tests, document hardware and
commands, and package weights plus metrics with provenance.

**Acceptance gate:** a clean clone can reproduce inference; PyTorch and ONNX
predictions agree within tolerance; regression tests pass; limitations clearly
state that the CNN assists analysis and does not replace sign-off DRC.

## Immediate sequence after B0

1. Merge the completed B0 benchmark into `main`.
2. Create B1 from updated `main`.
3. Implement the manifest and duplicate/conflict report first.
4. Freeze a layout-grouped split before any hyperparameter or architecture work.
5. Rerun the unchanged B0 model on that split to establish the leakage-free
   comparison point.
6. Start B2 experiments one controlled change per PR.

## Experiment record template

Each benchmark PR should include:

```text
Benchmark/phase:
Commit:
Dataset manifest/hash:
Split definition:
Model/configuration:
Seed(s):
Environment/device:
Command:
Best epoch:
Validation precision / recall / F1:
Test precision / recall / F1:
Per-layout results:
Checkpoint and metrics artifact:
Conclusion and next decision:
```
