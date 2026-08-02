# Model Improvement Roadmap

## Goal

Improve on the paper's Metal-1 DRC classifier with a reproducible system that:

1. preserves competitive clean/dirty classification under a reproducible,
   leakage-aware reference protocol;
2. demonstrates generalization to unseen layout families under a stricter,
   leakage-free evaluation;
3. localizes violations as pixel-level masks and exact layout coordinates; and
4. proposes repairs that are accepted only after DRC and connectivity/LVS
   verification.

[*Design Rule Checking with a CNN Based Feature
Extractor*](https://arxiv.org/abs/2012.11510) reports accuracy of up to 92%.
That is an external reference, not the B0 acceptance threshold. Its artificial
dataset was derived from 50 SRAM designs, while B0 used the project's current
Sky130 tiles and a different split. Claims against the paper require a
documented protocol that reproduces the paper's evaluation conditions.

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

Primary classifier-selection metric: dirty-class F1 on validation data. Recall
is a hard safety metric; a candidate with zero dirty recall is rejected. The
test split is evaluated only after configuration selection, not used to tune
the model. Localization phases additionally report mask IoU/Dice and
coordinate-level detection precision/recall. Repair phases report verified fix
rate, new-violation rate, and connectivity/LVS preservation.

## Evaluation tracks

After B1 freezes the data, every accepted classifier is evaluated on both:

- **Leakage-aware tile-random reference:** preserve the paper's clean/dirty
  classification task while keeping exact and Manhattan-equivalent content in
  one split. It is the project's internal classification reference, not a
  direct reproduction of the paper's incompletely documented protocol.
- **Unseen-layout generalization:** train, validate, and test on disjoint layout
  families. Original, clean, error-injected, and other derived variants of one
  design belong to the same group.

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

### B1 — Layout collection, dataset integrity, and frozen evaluation (complete)

**Hypothesis:** the current small set of related layouts, duplicated clean tiles,
and tile-level random splitting do not support a trustworthy paper comparison
or a meaningful generalization claim.

**Work:**

1. Inventory the existing layouts and candidate additions by design family,
   circuit type, density, geometry distribution, provenance, and license.
2. Collect/generate enough additional, reproducible layouts to support disjoint
   train, validation, and test families. Prefer diversity over many variants of
   one design.
3. Build a manifest containing path, label, source layout, layout family,
   clean/error-derived relationship, coordinates, shape, dtype, density,
   content hash, source/license, and data-generation configuration.
4. Detect exact duplicates and contradictory labels before splitting.
5. Deduplicate before split or keep duplicate groups in exactly one split.
6. Freeze two versioned protocols: a leakage-aware tile-random classification
   reference and a fixed group split by source layout family for unseen-layout
   generalization.
7. Report class balance and per-layout metrics; add visual label-audit samples.

**Acceptance gate:** the dataset has enough independent layout families for all
three splits; no family or content hash crosses splits; no unresolved
contradictory labels remain; provenance/configuration is complete; and repeated
runs with the same manifest and seed reproduce both protocols and metrics.

**Accepted result:** seed 42 produced 14,348 samples from 14 independent layout
families (7,784 clean and 6,564 dirty). The source audit verified every admitted
layout hash and found no exact or Manhattan-equivalent label conflicts. The
unseen-layout protocol retains 6,442 train / 2,627 validation / 2,682 test
samples after removing 2,597 equivalent duplicates; its eight/three/three
families are disjoint. The leakage-aware tile-random reference contains 11,478 /
2,153 / 717 samples. Manifest `9deef1271a14...` passes the B1 gate, and a
deterministic clean/dirty visual audit overlays the KLayout violation geometry.
The admitted source set includes digital CPU, FPGA, RISC-V SoC, crypto, DSP,
VGA, SRAM, and multiple analog layout families.

### B2 — Dual classification baselines (complete)

**Purpose:** establish the credible starting numbers that B0 could not provide.

**Work:** rerun the unchanged `NCSU_DRCNN` separately on the frozen
leakage-aware tile-random and unseen-layout protocols. Use a sufficient epoch
budget, best-validation-loss checkpointing, multiple fixed seeds, and
per-layout reporting.

**Acceptance gate:** both baselines are reproducible, neither collapses to one
class, protocol differences from the paper are documented, and metrics plus
checkpoints are stored with dataset/split identifiers.

**Accepted result:** all six CUDA runs completed for seeds 42, 43, and 44 using
30 epochs, RMSprop at `0.001`, batch size 32, and train-only Manhattan
augmentation. The tile-random reference reached **92.47% +/- 0.61% accuracy**
and **91.89% +/- 0.75% dirty F1**. The layout-family-disjoint protocol reached
**90.38% +/- 0.84% accuracy** and **90.94% +/- 0.72% dirty F1**. No selected
checkpoint collapsed to one class, and the aggregate report records manifest
`9deef1271a14...`, configuration `d7fa939a1212...`, source hashes, runtime,
checkpoint hashes, pooled confusion matrices, and per-layout support.

The generalization gap is 2.09 accuracy points and 0.95 dirty-F1 points. The
three unseen test families reached 92.91% (`tt_um_Bingyao_FCOTA`), 89.76%
(`tt_um_2048_vga_game`), and 88.97% (`tt_um_c4m_spsram_direct`) mean accuracy.
The 92.47% internal reference is numerically close to the paper's “up to 92%”
result, but the datasets and protocols differ, so B2 does not claim to
outperform the paper. The accepted artifacts are stored under
`results/b2_baselines/`; checkpoints remain reproducible local artifacts whose
SHA-256 digests are recorded in the summary.

### B3 — Training and optimization improvements (in progress)

**Hypothesis:** optimizer, learning rate, regularization, and class sampling can
improve learning stability once the evaluation protocol is trustworthy.

**Work:** compare one variable per run, beginning with optimizer/learning-rate
sweeps, then scheduler, weight decay, batch size, and balanced sampling or class
weights if B1 demonstrates imbalance. Add early stopping and machine-readable
experiment summaries.

**Acceptance gate:** the selected configuration improves validation dirty F1
across multiple fixed seeds without reducing dirty recall or degrading
held-out-layout performance on either evaluation track. Record mean and
variation, not only the best run.

**Pre-registered first experiment:** keep the B1 manifest, `NCSU_DRCNN`,
seeds 42/43/44, 30 epochs, batch size 32, zero weight decay, and train-only
Manhattan augmentation fixed. First compare RMSprop and Adam at `0.001`; then
compare `0.0003`, `0.001`, and `0.003` for the validation-selected optimizer.
Selection uses only mean dirty F1 on the `unseen_layout_v1` validation split,
requires validation dirty recall at least equal to B2, and requires paired F1
improvement on at least two seeds. Candidate training explicitly skips test
evaluation. Only the selected configuration is then evaluated on both frozen
test protocols, where mean accuracy, dirty recall, and dirty F1 must each be no
worse than B2. If the search or confirmation gate fails, retain the B2
configuration and record the negative result before trying the next B3 factor.

### B4 — Model architecture experiments

**Hypothesis:** a compact modern CNN can outperform the original paper-style
network while retaining inexpensive inference.

**Work:** establish a simple architecture candidate with normalization and
global pooling, then compare capacity changes or a small residual model. Keep
input tiles, B1 splits, and the selected B3 training configuration fixed. Track
parameter count, checkpoint size, and CPU/ONNX latency alongside quality
metrics.

**Acceptance gate:** statistically consistent F1/recall improvement on held-out
layouts with an acceptable inference-cost increase. Reject complexity that only
improves the training layouts.

### B5 — Error analysis and targeted layout/data improvement

**Hypothesis:** performance is limited by identifiable layout patterns, boundary
labels, metal-density regimes, or synthetic-error coverage rather than model
capacity alone.

**Work:** review false positives/negatives by source layout, density, geometry,
and distance from tile boundaries. Correct label-generation defects and add
targeted layouts, examples, or hard-negative mining only where the analysis
supports it. Additions may expand training/validation data, but the frozen test
layouts must not be selected or modified in response to test results. Version
the resulting dataset and manifest.

**Acceptance gate:** the targeted dataset change improves its intended error
slice and overall held-out-layout metrics without introducing leakage or a
material regression elsewhere.

### B6 — Pixel-level violation localization

**Hypothesis:** a segmentation model trained from exact KLayout violation
geometry can localize `m1.2` violations precisely enough for coordinate-level
verification, unlike Grad-CAM or tile boxes.

**Work:** generate aligned image/mask pairs from the DRC report geometry, audit
mask registration, and train a U-Net-style segmentation baseline. Evaluate on
unseen layout families and separate classification quality from localization
quality.

**Acceptance gate:** the model improves over a declared localization baseline
using mask IoU/Dice and coordinate-level precision/recall; mask overlays pass a
visual registration audit; and no test-layout information enters training.

### B7 — Layout coordinates and end-to-end detection calibration

**Hypothesis:** a threshold selected on validation layouts plus spatial merging
can turn tile probabilities into useful layout-level detections.

**Work:** map predicted masks through tile origins and raster scale into exact
layout/GDS coordinates. Calibrate thresholds on validation layouts, merge
overlapping predictions, and compare them with KLayout ground truth using
spatial matching rather than only tile labels. Measure layout-level recall,
false detections, coordinate error, and latency.

**Acceptance gate:** coordinate transforms pass synthetic round-trip tests; a
frozen threshold meets an explicitly chosen recall target on unseen layouts;
and false-positive rate, localization tolerance, and runtime are documented.

### B8 — Verified automatic-fix prototype

**Hypothesis:** exact `m1.2` violation geometry can support a constrained repair
proposal without silently changing circuit intent.

**Work:** implement one conservative spacing-repair strategy, preserve an audit
trail of proposed geometry edits, rerun KLayout DRC, and perform a
connectivity/LVS comparison. Never accept a repair based only on the model's
confidence.

**Acceptance gate:** report attempted fixes, original violations removed, new
violations introduced, verified-fix rate, and connectivity/LVS result. Any
failed safety gate leaves the source layout unchanged and marks the proposal as
rejected.

### B9 — Reproducible release candidate

**Purpose:** make the selected model easy to reproduce and safe to evaluate.

**Work:** freeze dataset/model versions, export and numerically compare PyTorch
and ONNX outputs, add automated smoke/regression tests, document hardware and
commands, and package weights plus metrics with provenance.

**Acceptance gate:** a clean clone can reproduce inference; PyTorch and ONNX
predictions agree within tolerance; regression tests pass; limitations clearly
state that the CNN assists analysis and does not replace sign-off DRC.

## Immediate sequence after B2

1. Freeze the accepted B2 configuration and metrics as the comparison point for
   both evaluation protocols.
2. Start B3 optimizer and learning-rate experiments, changing one factor per
   run and selecting configurations only from validation dirty F1.
3. Require every B3 candidate to preserve dirty recall and avoid degrading the
   unseen-layout baseline across the same fixed seeds.
4. Collect additional layouts again only in B5 when error analysis identifies a
   missing geometry or circuit regime; never adapt the frozen test set.

## Experiment record template

Each benchmark PR should include:

```text
Benchmark/phase:
Commit:
Dataset manifest/hash:
Evaluation track:
Split definition:
Model/configuration:
Seed(s):
Environment/device:
Command:
Best epoch:
Validation precision / recall / F1:
Test precision / recall / F1:
Per-layout results:
Localization IoU / Dice / coordinate metrics (when applicable):
Verified fixes / new violations / connectivity-LVS result (when applicable):
Checkpoint and metrics artifact:
Conclusion and next decision:
```
