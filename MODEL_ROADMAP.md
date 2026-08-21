# Model Improvement Roadmap

## Goal

Produce a publishable result that solves a problem an exact DRC engine does not
already solve better. A learned system is useful only if it creates a measured
workflow benefit while KLayout remains the source of truth.

The original classifier/localizer work through B7.2 is retained as a completed
feasibility study and negative competitiveness result. The controlled B7.2
audit rejected the CNN as a DRC accelerator. On the validation layouts, the
complete learned pipeline took 4,693.87 seconds while KLayout's median was
18.35 seconds (KLayout was 255.85x faster), and CNN violation recall was only
95.33% versus the registered 99.5% requirement. On development confirmation,
the corresponding times were 1,392.87 and 8.59 seconds (KLayout was 162.21x
faster) with 95.51% recall and one clean-layout false alarm. Exact recovered-
pair precision was 100% only after KLayout checked the CNN proposals; it does
not recover the violations the CNN never proposed. Do not tune or present this
detector as an accelerator, replacement, or competitor to KLayout.

The revised research goal is therefore:

1. preserve the controlled B7.2 negative result and stop the
   detection-accelerator branch;
2. use exact downstream DRC as the outcome oracle for a harder task—predicting
   the consequences of available flow actions and choosing an action before
   an expensive place-and-route run—rather than relearning a deterministic
   spacing check; and
3. release a reproducible, family-disjoint benchmark and an honest Pareto
   comparison against deterministic baselines.

[*Design Rule Checking with a CNN Based Feature
Extractor*](https://arxiv.org/abs/2012.11510) reports accuracy of up to 92% on
artificial data derived from 50 SRAM designs. The authors' exact dataset and
split are not available in this repository, so the project must distinguish two
claims:

- **Numerically better than the paper:** mean accuracy above 92% across the
  fixed seeds on a pre-registered paper-style SRAM benchmark, with dirty recall
  preserved.
- **A stronger design-flow system:** unseen-family generalization and measured
  reduction in routing attempts or turnaround through action-conditioned DRC
  prevention, while exact DRC and PPA checks remain authoritative.

Do not call the first claim a direct reproduction unless the original data and
protocol are obtained. Report the protocol differences beside every comparison.

## Impact test

[KLayout's DRC engine](https://www.klayout.de/doc/manual/drc_basic.html) already
returns exact error geometry and supports local region checks, tiling,
multicore execution, and hierarchical processing. Commercial interactive DRC
also checks a visible area, changed area, or whole cell view with signoff-quality
decks, as documented for [Cadence iPegasus](https://www.cadence.com/en_US/home/tools/digital-design-and-signoff/silicon-signoff/ipegasus-for-virtuoso-studio.html).
Therefore, “a CNN can detect an error” is not a sufficient contribution.

| Candidate contribution | Benefit over exact DRC | Decision |
|---|---|---|
| Finished-layout `m1.2` detection and coordinates | None: B7.2 measured much lower recall and 162x–256x slower end-to-end execution | Closed after failed B7.2 hard gate |
| Incremental changed-region screening | KLayout already performs exact local checks, and the learned full-layout path failed before an incremental claim was justified | Closed with the detector branch |
| Verified repair ranking | KLayout can accept/reject edits but does not encode which legal edit best preserves design intent and cost | Possible downstream extension, but current repair work is crowded |
| Pre-route hotspot prediction only | Exact polygon DRC cannot run before final geometry exists | Useful baseline, but prediction alone is already crowded |
| Action-conditioned flow control | Amortizes action selection across unseen designs and seeks to reduce expensive full-flow trials while exact DRC remains authoritative | Recommended publication track, subject to B8.0 actionability and learnability gates |

The research landscape is also moving beyond binary detection: recent work
studies [AI-guided detailed routing](https://doi.org/10.1145/3769306),
[offline-RL routing schedules](https://arxiv.org/abs/2512.03594),
[uncertainty-aware routability maps](https://arxiv.org/abs/2607.16674), and
[verification-in-the-loop repair](https://arxiv.org/abs/2607.22761). A
publishable B8 result must therefore contribute counterfactual action
conditioning, system-level closed-loop evidence, calibrated fallback, and
generalization—not merely add a predictor or tune global parameters.

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

**Original B7 protocol:** complete; precision gate failed. Average the
probability outputs of the accepted seeds 42/43/44. Scan
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

**Original B7 result:** the validation-selected policy was segmentation
threshold `0.4`, classification threshold `0.8`, 16-pixel minimum area, and a
2-pixel merge gap. It achieved 97.66% validation violation recall and 85.37%
validation component precision. Development confirmation retained 97.88%
violation recall and 100% exact recovered-pair precision, but component
precision was 62.57%, below the 80% gate. `tt_um_c4m_spsram_direct` accounts
for 686 of 810 false components, including 301 detections on its clean source
layout; the other two development families combine to 87.41% precision. B7 is
therefore rejected as a deployment policy, while its stitching, coordinate
mapping, and exact KLayout recovery are retained.

**B7.1 controlled correction:** preserve the original B7 evidence and reuse
only its checkpoint/hash-bound complete scan caches. Extend the validation
classification thresholds densely from `0.80` through `0.99`; keep the model,
freeze segmentation at the original validation-selected `0.4`, and keep the
minimum-area candidates, merge-gap candidates, layouts,
and checkpoints unchanged. Select maximum validation component precision
subject to at least 95% validation violation recall, freeze the policy, and
then recompute development metrics once. B7.1 retains the original B7
acceptance gates of at least 85% development violation recall and at least 80%
development candidate-component precision. This is sequential development,
not final evidence, and does not open B9.

**B7.1 result — accepted:** validation-only selection chose classification
threshold `0.92` with the unchanged segmentation threshold `0.4`, minimum area
`16` pixels, merge gap `2` pixels, and recovery radius `140` nm. Development
confirmation achieved 95.51% violation recall, 81.44% candidate-component
precision, 87.92% component F1, and 100% exact recovered-pair precision. All
registered gates passed. The final holdout remained unopened. B7.2 then tested
whether this accepted internal policy created any workflow advantage over exact
KLayout.

### B7.2 — KLayout competitiveness audit (complete; accelerator rejected)

**Question:** is there any density, layout size, or interactive-change regime
where the learned detector reduces end-to-end latency enough to justify its
misses and false candidates?

**Benchmark contract:** run the current frozen B7.1 system and direct KLayout
on the same source and injected layouts, same host, same `m1.2` rule, and same
loaded-layout boundary. Measure at least five repetitions after one warm-up.
Separate process startup and layout parsing from rule execution, and also report
the user-visible cold-command total.

Use a staged audit to avoid optimizing a detector that already fails the
quality gate. Stage A benchmarks the exact KLayout Python
`Region.space_check(140 nm)` operation used by the label and local-recovery
path, including fresh-layout, loaded-layout, and changed-region cases. If the
CNN fails the registered quality gate or is already slower, record the negative
result and stop. Only a detector that survives Stage A may support a positive
latency claim; Stage B must then test the KLayout CLI in applicable flat/deep
or hierarchical modes and sweep tiling and thread count rather than using a
deliberately weak default.

Record:

- layout area, shapes, hierarchy depth, tile count, and true violation count;
- KLayout parse, rule-execution, report-write, peak-memory, and total time;
- CNN rasterization, inference, stitching, local exact recovery, report-write,
  peak-memory, and total time;
- violation recall, false candidates per mm², clean-layout false-alarm rate,
  exact recovered-pair precision, and p50/p95 latency;
- scaling with area, shape count, violation density, thread count, and changed
  region size.

Include two deployment scenarios:

1. **Batch signoff-like:** a fresh process reads a whole layout and emits an
   exact report.
2. **Interactive incremental:** the layout is already loaded and only a known
   changed bounding box plus a rule halo is checked.

**Hard gate:** the accelerator claim survives only if the complete learned
pipeline is at least 2× faster than the best correctly configured KLayout
baseline at at least 99.5% violation recall, with no miss on the registered
critical and near-threshold slices, no clean-layout false alarm in the final
holdout, and lower p95 latency in the claimed deployment scenario. This is an
engineering claim, not a signoff-equivalence claim. If any requirement fails,
archive the CNN detection branch as a negative result and do not spend another
phase tuning it.

**Measured result:** the Stage-A audit completed on the same six layouts and
both source/injected variants. KLayout used five measured repetitions after one
warm-up; the CNN timing is one complete synchronized GPU run, so a CNN p95 is
not available. This does not affect the decision because both quality and
median-speed requirements fail by large margins.

| Split | CNN total | KLayout median | KLayout conservative p95 | KLayout speed advantage | CNN recall | Quality gate |
|---|---:|---:|---:|---:|---:|---|
| Validation | 4,693.87 s | 18.35 s | 21.15 s | 255.85x | 95.33% | Fail |
| Development confirmation | 1,392.87 s | 8.59 s | 10.65 s | 162.21x | 95.51% | Fail |

Validation had no clean-layout false alarm but missed registered near-threshold
and severe-slice violations. Development confirmation flagged one clean layout
and missed near-threshold violations. Exact recovered-pair precision was 100%
for proposed candidates because local KLayout recovery is authoritative; it
does not change end-to-end recall. The audit's comparison boundary already
favored the CNN by excluding model/layout load and result serialization, while
KLayout included layout parse, M1 materialization, exact rule execution, and
RDB writing.

**Decision:** the hard gate failed. Stage B optimization is unnecessary because
the CNN did not survive Stage A. Freeze B7.2 as a negative result, do not rerun
or retune the detector, keep B9 unopened, and proceed to B8.0.

### B8 — Amortized, action-conditioned OpenROAD control

**Research question:** from a frozen snapshot after floorplan/PDN and before
global placement, can a model choose a complete OpenROAD configuration for an
unseen design using zero or one expensive trial, while matching or improving
the exact post-route DRC/PPA result reached by per-design search?

This does **not** change KLayout or OpenROAD's algorithms. It selects existing
OpenROAD controls before the flow runs; the normal tools execute the chosen
configuration, and exact post-route verification decides whether the result is
acceptable. The early decision checkpoint is deliberate: placement and routing
controls are both still legal, and the model cannot inspect outcomes from the
run it is trying to avoid.

[OpenROAD AutoTuner](https://openroad-flow-scripts.readthedocs.io/en/latest/user/InstructionsForAutoTuner.html)
already supports random/grid search, Bayesian and evolutionary methods, and
tuning of command-line flow variables. Therefore automated parameter tuning is
not the novelty. The testable contribution is **amortized cross-design action
selection**: learn counterfactual action rankings from previous design families,
recommend an action for a new family before repeated full flows, quantify
uncertainty, and fall back when confidence is insufficient. AutoTuner is a
required named baseline under the same number of expensive full-flow trials.

The paper is viable only if this controller reduces online trials or turnaround
versus default, fixed heuristics, and AutoTuner while preserving exact DRC and
PPA. It must also report the offline dataset cost and the number of unseen
designs required to amortize that cost. A hotspot map, a tuned global default,
or a result that wins only with a larger search budget is not sufficient.

The executable preregistration is in
[`B8_ACTIONABILITY_PROTOCOL.md`](B8_ACTIONABILITY_PROTOCOL.md).

#### B8.0 — Actionability pilot and kill gate

Use the pinned OpenROAD-flow-scripts container and `sky130hd` platform. First
run a nine-flow harness smoke test: three official design families, three
actions, and one seed. It validates resume/checkpoint behavior, metric parsing,
exact verification, and deterministic action serialization; it cannot support a
scientific claim.

If the harness passes, run the frozen actionability matrix on the seven official
`sky130hd` families (`aes`, `chameleon`, `gcd`, `ibex`, `jpeg`, `microwatt`,
and `riscv32i`): nine actions by two seeds, for 126 full flows. The initial
action grid is the Cartesian product of
`PLACE_DENSITY_LB_ADDON = {0.00, 0.05, 0.10}` and
FastRoute layer adjustment `{0.20, 0.35, 0.50}`. The latter is serialized in a
per-run `FASTROUTE_TCL` using `set_global_routing_layer_adjustment`; it is not a
standalone ORFS environment variable. Seeds are paired nuisance repetitions,
not actions. Do not add macro-specific knobs until this gate passes.

Record flow success, exact post-route DRC total and per rule, stage/total
runtime, wirelength, vias, WNS/TNS, power/area proxies, the resolved placement
density, source snapshot/features, tool/container hashes, action, and seed.

Continue to B8.1 only if all preregistered conditions hold:

1. **Action effect:** on at least four of seven families, an action changes an
   exact DRC/PPA-feasibility outcome or improves DRC by at least 20%; if exact
   DRC ties, a 20% runtime improvement without registered PPA regression counts.
2. **Winner diversity:** no one fixed action is the acceptable Pareto winner on
   five or more of the seven families.
3. **Oracle headroom:** the per-design oracle action materially outperforms the
   best fixed action across families; otherwise selection cannot add value.
4. **Seed stability:** paired action effects exceed the corresponding seed
   variation often enough that the target is an action signal, not router noise.
5. **Pre-action learnability:** leave-one-family-out transparent baselines from
   only checkpoint-available features beat the best fixed action in normalized
   regret and recover the oracle in their top two recommendations often enough
   to justify collecting more families.

The exact scalarization and PPA tolerances are frozen in the protocol before
the first matrix result is inspected. Failure of action effect, winner
diversity, or oracle headroom stops the controller project. Failure only of
learnability permits one registered fallback: move the checkpoint later and
restrict the action set to controls that remain legal, then repeat the pilot as
a new hypothesis rather than silently changing the task.

#### B8.1 — Action-conditioned trajectory dataset

Only after B8.0 passes, expand to at least 20 independent design families before
making a learned generalization claim. For each frozen pre-global-placement
snapshot and candidate action, store:

- spatial features: macro, cell, pin, routing-demand, congestion, blockage, and
  layer-utilization maps;
- structural features: cells, nets, pin connectivity, placement, and timing
  criticality when available;
- context: PDK/rule parameters and the candidate action vector;
- exact outcomes: post-route DRC map/count by rule, iterations, runtime,
  wirelength, vias, WNS/TNS, power/area proxies, and DRC-clean status.

Keep design families and generators disjoint across training, development, and
final holdout. Never split action runs or seeds from one design family across
partitions. Hash every source snapshot, action configuration, tool version,
container, and output. CircuitNet can be used for representation pretraining or
external comparison, but the action labels and end-to-end claims must come from
the reproducible flow generated here.

#### B8.2 — Counterfactual multi-modal model

Build a shared spatial/structural encoder with explicit action and rule
conditioning. For each candidate action, predict:

- post-route DRC heatmap and count by rule;
- probability of DRC-clean completion;
- routing runtime/iteration count;
- timing, wirelength, via, and power/area deltas; and
- calibrated predictive uncertainty.

Start with transparent baselines before a large model: best fixed action,
per-action mean, linear/gradient-boosted rankers, a spatial CNN, and a
design-level graph model.
Only add CNN+GNN/point fusion when simpler models establish that spatial and
connectivity information are complementary. The controller enumerates the
declared candidate actions, rejects high-uncertainty predictions, and selects
the lowest-risk action satisfying registered PPA limits.

#### B8.3 — Closed-loop comparison

Compare the selected action against:

- default OpenROAD configuration;
- documented manual heuristic or bisection;
- random and grid search at the same full-flow run budget;
- official OpenROAD AutoTuner, including its Bayesian/search methods, at the
  same full-flow run budget;
- a hotspot predictor without action conditioning; and
- the oracle best measured action as an upper bound.

Primary metrics are system outcomes, not pixel accuracy: number of expensive
full routing attempts, total design-turnaround time, exact final DRC count,
DRC-clean completion rate, routing iterations, PPA change, action-selection
regret, inference overhead, calibration, and out-of-distribution fallback rate.
Report both the online unseen-design cost and the lifecycle break-even point
after charging the learned method for offline trajectory generation and
training. The main comparison fixes the number of expensive full flows, not
just model inference time.

**Acceptance gate:** on family-disjoint development data, reduce full routing
attempts by at least 25% or total turnaround time by at least 1.3x versus the
best non-oracle baseline, without a worse exact DRC result, with less than 1%
wirelength degradation and no material registered WNS/TNS regression. Use
paired confidence intervals rather than one favorable seed.

#### B8.4 — Stretch: cross-PDK and spatial actions

If B8.3 passes, condition on normalized technology/rule features and test
transfer to a second open platform with an exact available verification flow.
Then evaluate region-specific padding/capacity actions instead of global-only
configuration selection. These are stretch contributions; do not claim them
without unseen-technology evidence and a safe uncertainty fallback.

Verified local repair remains a possible downstream application, but it is not
the primary post-B7.2 publication track.

### B9 — Final holdout and reproducible release

Select the release controller without opening the holdout, then evaluate it
once. Release the trajectory/action manifest, split hashes, action generator,
non-learned baselines, learned model, exact verification scripts, environment,
and per-run results. Export ONNX only if it changes a measured deployment
constraint.

The paper must separate four claims:

- B7.2 evidence about whether the detector accelerator survived or failed;
- full-flow attempts and turnaround improvement from action selection;
- exact DRC and PPA preservation versus search/tuning baselines; and
- calibrated generalization to unseen design families and, if claimed, PDKs.

The title and abstract must not imply replacement of signoff DRC. KLayout is
the exact oracle and final acceptance authority.

## Immediate sequence after B7.2

1. Merge the B7.2 audit, measured evidence, and frozen B8.0 preregistration;
   do not rerun or tune the detector.
2. Implement the B8.0 resume-safe OpenROAD harness and pass the nine-flow smoke
   test before launching the matrix.
3. Run the frozen 126-flow actionability matrix and publish all run manifests,
   failures, exact metrics, seed variation, and gate decisions.
4. Stop if actions do not create stable, design-dependent oracle headroom. Do
   not train a controller merely because the infrastructure works.
5. If every B8.0 gate passes, expand to at least 20 family-disjoint designs and
   establish fixed-action, heuristic, and AutoTuner baselines before a large
   learned model.
6. Train and evaluate the counterfactual controller under equal full-flow
   budgets, including uncertainty fallback and lifecycle cost.
7. Open the final holdout once in B9 and state only claims supported by it.

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
