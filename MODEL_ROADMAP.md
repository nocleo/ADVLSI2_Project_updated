# Model Improvement Roadmap

## Goal

Improve on the paper's Metal-1 DRC classifier with a reproducible system that:

1. exceeds the paper's reported 92% accuracy on a clearly documented,
   paper-style clean/dirty benchmark without sacrificing dirty recall;
2. demonstrates generalization to layout families and violation geometries that
   were not used for model or threshold selection;
3. localizes violations as pixel-level masks and exact layout coordinates;
4. reports realistic full-layout false-alarm rate, violation recall, and
   end-to-end runtime; and
5. proposes repairs that are accepted only after DRC and connectivity/LVS
   verification.

[*Design Rule Checking with a CNN Based Feature
Extractor*](https://arxiv.org/abs/2012.11510) reports accuracy of up to 92% on
artificial data derived from 50 SRAM designs. The authors' exact dataset and
split are not available in this repository, so the project must distinguish two
claims:

- **Numerically better than the paper:** mean accuracy above 92% across the
  fixed seeds on a pre-registered paper-style SRAM benchmark, with dirty recall
  preserved.
- **A stronger DRC system:** unseen-layout generalization, exact localization,
  realistic full-layout metrics, and verified repair in addition to tile
  classification.

Do not call the first claim a direct reproduction unless the original data and
protocol are obtained. Report the protocol differences beside every comparison.

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

B1 froze two useful development tracks:

- **Leakage-aware tile-random reference:** preserves the paper's clean/dirty
  classification task while keeping exact and Manhattan-equivalent content in
  one split. It is an internal reference, not a direct reproduction.
- **Unseen-layout development benchmark:** uses disjoint layout families.
  Original, clean, error-injected, and other derived variants of one design
  belong to the same group.

B3, B4, and B5.1 all inspected the existing test results and those observations
now influence later hypotheses. This is adaptive test reuse, even though no
sample leaked into training. From B5.2 onward, treat both existing test splits
as development-confirmation benchmarks rather than untouched final evidence.

Before the final model is selected, freeze two additional tracks:

- **Paper-style SRAM benchmark:** obtain the original data/protocol if possible;
  otherwise generate and document a 50-SRAM-style benchmark with macro-level
  grouping and report it only as a numerical comparison.
- **Untouched final holdout:** at least three to five new layout families,
  including generator-disjoint and near-threshold spacing cases. No result from
  this holdout may influence data collection, architecture, loss, threshold, or
  post-processing. Open it once for the release candidate.

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

### B3 — Training and optimization improvements (complete; no replacement accepted)

**Hypothesis:** optimizer, learning rate, regularization, and class sampling can
improve learning stability once the evaluation protocol is trustworthy.

**Work:** compare one variable per run, beginning with optimizer/learning-rate
sweeps, then scheduler, weight decay, batch size, and balanced sampling or class
weights if B1 demonstrates imbalance. Add early stopping and machine-readable
experiment summaries.

**Acceptance gate:** preserve mean validation/test accuracy, dirty recall, and
dirty F1 within 0.5 percentage points of B2, then earn at least one measurable
benefit: improved dirty F1/recall, lower seed variance, or at least 25% fewer
training epochs. Record mean and variation, not only the best run.

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

**B3.1 result:** completed as a valid negative experiment. RMSprop at `0.001`
remained the best and most stable recipe; Adam was less stable, `0.0003` did not
improve mean validation dirty F1, and `0.003` was unstable with one collapsed
seed. Frozen tests remained locked.

**Pre-registered extension:** B3.2a compares the unchanged RMSprop recipe with
one `ReduceLROnPlateau` scheduler for all 30 epochs. B3.2b adds early stopping
only after validation selection. B3.3 selects one dirty-class threshold per
seed from validation predictions, subject to the B2 recall floor minus the
0.5-point tolerance. Frozen-test commands are generated only if the validation
gate earns one of the benefits above.

**B3.2/B3.3 result:** the validation gate passed without test inspection. The
plateau scheduler improved mean validation dirty F1 from 91.36% to 91.78% and
dirty recall from 88.60% to 89.81%. Early stopping preserved those validation
metrics while averaging 26 epochs, a 13.3% reduction that did not meet the 25%
efficiency target. Per-seed validation thresholds (approximately 0.359–0.373)
raised mean validation dirty F1 to 92.08% and recall to 93.00%.

Frozen-test confirmation then failed the predeclared quality tolerance. On the
unseen-layout protocol, calibrated dirty recall rose to **94.42% +/- 0.93%**,
but accuracy fell from 90.38% to **89.71% +/- 0.89%**, a 0.67-point regression;
dirty F1 was **90.72% +/- 0.67%**. The tile-random reference remained comparable
at 92.52% accuracy and 91.97% dirty F1. B3 therefore records a useful
precision/recall trade-off but accepts no new training configuration or
threshold. B2 remains frozen for B4. Official evidence is stored under
`results/b3_optimization/`.

### B4 — Model architecture experiments (complete; no replacement accepted)

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

**Pre-registered first experiment:** compare the accepted B2 `NCSU_DRCNN` with
one `CompactBNPool` candidate containing four convolution/batch-normalization
blocks and concatenated global average/max pooling. Keep the B1 manifest, both
protocols, seeds 42/43/44, 30 epochs, RMSprop at `0.001`, batch size 32,
train-only Manhattan augmentation, best-validation-loss checkpointing, and
threshold `0.5` fixed. Select using only `unseen_layout_v1` validation data:
mean dirty F1 must improve in at least two paired seeds, validation accuracy and
recall must remain within 0.5 points of B2, and no seed may collapse.

Only a validation-selected candidate may be evaluated on frozen tests. Final
acceptance requires higher mean unseen-layout dirty F1 and recall, improvement
in at least two paired seeds for each, accuracy within 0.5 points of B2, fewer
parameters, a smaller state dict, tile-random accuracy/recall/F1 within 0.5
points of B2, and paired batch-one PyTorch and ONNX CPU median latency no more
than 1.5 times the baseline. If rejected, retain B2 and record the result before
considering a capacity or residual candidate.

**Result:** `CompactBNPool` passed validation selection, raising mean dirty F1
from 91.36% to 93.61% and improving all three paired seeds. It also reduced the
model from 602,114 to 42,178 parameters (14.3x fewer), reduced the state dict
from 2.41 MB to 179 KB, and passed the paired CPU/ONNX cost gate.

Frozen confirmation rejected the candidate. On unseen layouts, accuracy fell
from 90.38% to 89.36% and dirty F1 fell from 90.94% to 90.36%; no paired seed
improved unseen-layout F1. On the tile reference, accuracy and F1 improved, but
dirty recall fell from 93.39% to 91.36%, beyond tolerance. B2 therefore remains
the accepted classifier. The compact model is retained only as a possible
post-localization compression candidate. Official evidence is stored under
`results/b4_architecture/`.

### B5 — Evidence-driven classifier closeout

**Status:** B5.1 is complete and rejected. B2 remains the accepted classifier.

**B5.1 result:** validation selected the 25% B2 / 75% B4 probability blend and
raised mean validation dirty F1 to 93.87%. Frozen unseen-layout accuracy/F1
improved from 90.38%/90.94% to 90.47%/91.25%, and tile-reference accuracy/F1
improved to 94.70%/94.08%. Tile-reference dirty recall nevertheless fell from
93.39% to 92.07%, a 1.32-point regression beyond the 0.5-point tolerance. The
ensemble is correctly rejected.

The result is still useful: B4 uniquely corrected more tile-reference samples
than B2 (110 versus 68), but that advantage was concentrated on clean samples;
B2 uniquely corrected more dirty samples (51 versus 31).

#### B5.2 — Validation/training failure-slice audit

**Purpose:** identify one reproducible failure mechanism before spending another
GPU run.

**Work:** export one aligned record per training/validation sample containing
the true label, B2/B4 probabilities and predictions, layout family, metal
density, violation count, edge orientation and length, measured spacing
deficit, number of nearby shapes, and distance from the violation to the
supervised-center and tile boundaries. Report false negatives, false positives,
model disagreements, calibration, and per-family support. Do not inspect the
new final holdout.

**Gate:** a slice may motivate an experiment only when it has adequate support,
repeats across at least two layout families, and shows a meaningful error-rate
difference with uncertainty intervals. Otherwise close classifier tuning and
retain B2.

**Result:** complete. The audit runner and Colab notebook are available in
`scripts/run_b5_failure_audit.py` and
`notebooks/B5_2_Failure_Slice_Audit.ipynb`. They export and inspect only train
and validation predictions from the authoritative B2/B4 checkpoints. The B1
manifest supports layout-family and metal-density slices; exact violation
count, edge orientation/length, spacing deficit, nearby-shape count, and
boundary-distance slices require a separate exact-geometry annotation JSONL.
Absent geometry is marked unavailable and must not be inferred from raster
pixels.

B4 exceeded B2 on mean validation accuracy/F1 and calibration on both protocols,
but this does not reverse its frozen-confirmation rejection. The B2/B4
class-conditional disagreement mechanism passed only the tile-random protocol,
not unseen layouts. After correcting the audit to enforce its declared
two-family and two-seed requirements, 10 seed-level density candidates remain
across four repeated signatures. Every repeated signature concerns clean
samples; the affected density bin is `0.15-0.30` for unseen-layout validation
but `0.03-0.15` for tile-random validation. Coarse density is therefore a useful
diagnostic but not a protocol-stable mechanism for another classifier-only
experiment. The official report is in `results/b5_failure_audit/`.

#### B5.3 — One evidence-selected classifier experiment

Pre-register exactly one intervention from the B5.2 audit:

- **Boundary/context failures:** rebuild training/validation tiles with the B6
  halo/stride geometry and targeted boundary positives.
- **Small-versus-dense geometry failures:** test one compact multi-scale
  residual classifier with 1x1 bottlenecks plus local and dilated 3x3 branches.
- **Calibration/model-disagreement failures:** select one recall-constrained
  ensemble threshold or uncertainty gate jointly on both validation tracks.
- **Class/severity imbalance:** use targeted hard-positive/hard-negative
  sampling or a severity-aware loss without changing the architecture.

Selection uses validation only and must preserve dirty recall within 0.5 points
on both validation tracks while improving mean dirty F1 in at least two of three
seeds. Existing test splits may be reported as development confirmation but
cannot restore their status as untouched evidence. If the single experiment is
rejected, stop classifier-only work and keep B2.

**Decision:** close B5.3 without a training run. Neither the ensemble mechanism
nor the available density evidence selects a sufficiently specific intervention
that should generalize across both protocols. Retain B2 and generate exact
edge-pair, boundary, orientation, and severity annotations as part of B6.1.

### B6 — Coverage-correct dataset and multi-task localization

**Hypothesis:** exact mask supervision and a shared localization/classification
encoder can improve violation reasoning more effectively than additional blind
classifier sweeps.

#### B6.1 — Fix tiling and mask generation

The current 1600 nm input, 100 nm margin, and 1500 nm stride leave a 100 nm band
that is never inside a supervised center. Replace it with:

- 1600 nm input window;
- 160 nm halo on every side, greater than the 140 nm `m1.2` rule;
- 1280 nm central output region;
- 1280 nm stride, so valid output regions tile without gaps;
- deterministic padding and coverage at layout boundaries.

Generate aligned raster inputs and masks from the exact KLayout report geometry.
Preserve the original vector edge-pair annotation beside every raster mask.
Slight mask dilation may be used for the training loss, but coordinate
evaluation must use the undilated vector ground truth.

**Gate:** unit tests prove complete central-region coverage, no duplicate
ownership of output pixels, correct nm/pixel round trips, and mask registration.
A visual audit covers clean, dense, sparse, horizontal, vertical, boundary, and
near-threshold examples.

**Result:** accepted. The 14-family registry build contains 6,924 exact KLayout
`m1.2` edge pairs, each with one unique central-output owner, plus 8,021 dirty
and 8,021 balanced clean image/mask tiles. Inputs are 200x200 and central masks
are 160x160 at 8 nm/pixel. Every layout supplies all seven visual-audit
categories. Exact vectors remain authoritative; this build required no
one-pixel surrogate targets and omitted no non-owner subpixel fragments. The
large generated artifact remains ignored, while its compact result and archive
hash are versioned under `results/b6_localization_dataset/`. Proceed to B6.2.

#### B6.2 — Multi-task U-Net baseline

Train a small U-Net-style fully convolutional model with:

- a segmentation head for the central 1280 nm violation mask;
- an auxiliary clean/dirty classification head from the shared encoder;
- Dice plus weighted BCE/focal loss for sparse masks;
- classification cross-entropy with a pre-registered loss weight;
- train-only Manhattan transformations applied identically to image and mask.

Start with the simple baseline. Add a multi-scale/residual block only if B5.2
or B6.2 errors justify it. Grad-CAM and tile boxes are diagnostic baselines, not
acceptable exact-localization outputs.

**Gate:** compare classification accuracy/dirty precision/recall/F1 with B2 and
localization with a declared box/Grad-CAM baseline. Report mask Dice/IoU,
object-level precision/recall/F1, centroid distance, edge-pair distance, spacing
severity, boundary distance, and per-layout-family metrics. No final-holdout
result is used for tuning.

**Result:** accepted. All three CUDA runs completed for seeds 42, 43, and 44
using 30 epochs, AdamW, batch size 16, train-only paired Manhattan transforms,
and the registered weighted BCE + Dice + 0.25 x classification loss. The model
and both thresholds were selected using the three validation families only;
the three previously inspected test families remained development
confirmation and the B9 final holdout stayed unopened.

On the development-confirmation families, the 482,963-parameter U-Net reached
**95.51% +/- 0.83% classification accuracy**, **98.86% +/- 0.27% dirty recall**,
and **95.66% +/- 0.75% dirty F1**. Localization reached **86.32% +/- 1.65%
dirty-mask Dice**, **84.23% +/- 1.77% raster-object F1**, and **87.19% +/-
0.64% exact-vector unique-owner recall**. The same-tile B2 comparison reached
94.02% accuracy, 96.19% dirty recall, and 94.15% dirty F1, while its declared
full-box localization baseline reached only 3.82% Dice, 0% raster-object F1,
and 3.76% vector-owner recall.

The development acceptance gate required mean dirty-mask Dice >= 0.75,
per-tile raster-object F1 >= 0.75, exact-vector unique-owner recall >= 0.85,
and classification dirty recall within two percentage points of B2 when both
are evaluated on the same B6 tiles. All four checks passed, so B6.2 advances to
B7. This does not open or redefine the untouched B9 holdout. Full-layout
stitched precision and exact edge recovery remain B7 metrics.

The declared coarse localization baseline applies each authoritative B2 dirty
probability to the entire 160x160 central output box and evaluates it with the
same mask, raster-object, and exact-vector-owner metrics as the U-Net. This
quantifies the spatial gain over tile classification without treating Grad-CAM
as exact geometry.

The B7 error-analysis priority is precision, not another blind architecture
sweep. Raster-object recall is 99.36%, but precision is 73.12%, which indicates
fragmented or duplicate predicted components. The SRAM development family
`tt_um_c4m_spsram_direct` is weakest at 78.95% mean Dice and 83.25%
exact-vector recall. B7 must freeze a validation-only stitching/deployment
policy, report this family separately, and quantify full-layout false alarms
before any further model change.

### B7 — Exact coordinates and realistic full-layout evaluation

**Hypothesis:** central-region stitching plus deterministic geometry recovery
can convert model masks into sign-off-checkable candidate violations.

**Work:**

1. stitch only the non-overlapping central outputs from B6;
2. map connected mask components through tile origins and raster scale into
   layout coordinates;
3. query KLayout geometry near each component to recover the exact pair of M1
   edges and measured spacing;
4. choose mask, object-matching, and classification thresholds on validation
   layouts only;
5. evaluate complete layouts at their natural, mostly-clean prevalence.

Do not keep the current unexplained split between a 0.5 benchmark threshold and
a 0.80 inference default. Freeze one deployment policy from validation and use
it everywhere.

**Gate:** coordinate transforms pass synthetic round-trip tests. Report unique
violations detected/total violations, false detections per mm2, false-positive
tiles per million scanned tiles, clean layouts incorrectly flagged, recall by
spacing deficit and boundary distance, detections before/after merging, peak
memory, and end-to-end layout runtime. Compare tiled and fully convolutional
execution where practical; prefer the path that avoids recomputing overlapping
features without changing model outputs.

**Pre-registered B7 protocol:** implementation ready; measured GPU result
pending. Average the probability outputs of the accepted seeds 42/43/44. Scan
all central outputs of the three validation families and their original source
variants; do not reuse B6's balanced clean sampling for prevalence metrics.
Select the segmentation threshold first, then select the tile-classification
gate, minimum merged-component area, and 0/1/2-pixel fragment gap using only
those complete validation layouts. The selected policy must retain at least
85% validation violation recall and is frozen before the three development-
confirmation families are scanned. B7 is accepted only if complete-grid and
coordinate-round-trip checks pass, development exact-violation recall is at
least 85%, and development candidate-component precision is at least 80%.

Sparse components are stitched in global raster coordinates without allocating
a full-layout image. Each proposal exports its physical centroid/bounding box,
mean/maximum confidence, source tiles, and the exact two M1 edges recovered by
a local KLayout `m1.2` query. The teammate notebook's four-panel visualization,
per-component records, confidence fields, and validation threshold plot are
adapted; its random tile split, 200x200 mask ownership, and checkpoint are not.
The official execution remains batched non-overlapping tiling because the
accepted auxiliary classification head uses global pooling and a fixed central
crop, so a single full-layout convolution would change model outputs.

### B8 — Constrained, verified repair proposals

**Hypothesis:** a localized `m1.2` edge pair can support a small set of safe,
ranked repair candidates.

**Work:** begin only with isolated synthetic spacing violations. For each exact
edge pair, enumerate minimal grid-snapped translations or trims that add the
spacing deficit plus a declared guard band. Rank proposals by displacement,
area change, and collateral geometry impact. Write candidates to copies and
retain a complete audit trail; never edit the source layout in place and never
accept a repair from model confidence alone.

For every candidate:

1. rerun the exact `m1.2` check;
2. rerun the complete available Sky130 DRC deck;
3. reject any proposal that creates a new violation;
4. compare polygon/net connectivity with the original;
5. run true LVS only when a reference netlist and reproducible LVS flow exist.

**Gate:** report attempted proposals, original violations removed, new
violations introduced, verified-fix rate, displacement/area cost, and
connectivity or LVS result. Without a reference netlist, call the safety check
a connectivity-equivalence check rather than LVS.

### B9 — Final claim and reproducible release

**Purpose:** freeze the end-to-end result and state only claims supported by the
evaluation design.

**Work:** select the release candidate without the untouched final holdout,
then open that holdout once. Evaluate the paper-style SRAM benchmark, final
unseen/generator-disjoint layouts, full-layout detection, localization, and
repair. Export PyTorch and ONNX models, verify numerical agreement, record
hardware and commands, and add automated smoke/regression tests. INT8
post-training quantization is optional and is accepted only if it preserves the
quality gates while improving measured latency or memory.

**Gate:** a clean clone reproduces inference; PyTorch and ONNX agree within
tolerance; the final holdout is still untouched at evaluation time; and the
report separates:

- numerical accuracy relative to the paper;
- unseen-layout generalization;
- localization quality;
- full-layout false-alarm/throughput results; and
- verified repair quality.

The release must state that the CNN proposes candidates and does not replace
sign-off DRC.

## Immediate sequence after B5.1

1. Keep B2 as the accepted classifier and record B5.1 as a useful rejection.
2. Close B5.2 with its validation-only report and B5.3 no-run decision.
3. Acquire and lock the paper-style SRAM benchmark and a new untouched final
   holdout before the release candidate is chosen.
4. B6.1 complete: gap-free tiling and vector-backed masks are available.
5. Train the B6.2 multi-task U-Net, then recover exact geometry and full-layout
   metrics in B7.
6. Implement one conservative `m1.2` repair family in B8 and accept proposals
   only after DRC plus connectivity/LVS verification.
7. Open the final holdout once in B9 and make the strongest claim the evidence
   supports.

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
