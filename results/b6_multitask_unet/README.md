# B6.2 multi-task U-Net

Status: **complete and accepted; proceed to B7 exact-coordinate recovery and
full-layout evaluation**.

## Protocol

- Dataset: B6.1 vector-backed localization archive, SHA-256
  `ebc35cebe605a467ba87f5e615b8e74986d3e15fccba4f9cc3ca0d33e7f2f9e8`.
- Fixed `unseen_layout_v1` family split: eight training, three validation, and
  three development-confirmation layout families.
- Seeds 42, 43, and 44; 30 epochs; AdamW; batch size 16; 482,963-parameter
  `MultiTaskUNet`.
- Checkpoints and classification/segmentation thresholds selected using the
  validation families only.
- The previously inspected test families are development confirmation, not the
  untouched B9 final holdout.
- B2 is evaluated on the same B6 tiles. Its dirty probability is applied to the
  full 160x160 central output box as the declared coarse-localization baseline.

## Development-confirmation result

Mean and sample standard deviation across the three seeds:

| Metric | Multi-task U-Net | B2 on the same B6 tiles | Delta |
|---|---:|---:|---:|
| Classification accuracy | 95.51% +/- 0.83% | 94.02% +/- 1.07% | +1.50 points |
| Dirty precision | 92.68% +/- 1.63% | 92.21% +/- 1.73% | +0.47 points |
| Dirty recall | 98.86% +/- 0.27% | 96.19% +/- 0.24% | +2.67 points |
| Dirty F1 | 95.66% +/- 0.75% | 94.15% +/- 0.99% | +1.51 points |
| Dirty-mask Dice | 86.32% +/- 1.65% | 3.82% +/- 0.02% | +82.50 points |
| Dirty-mask IoU | 75.96% +/- 2.54% | 1.95% +/- 0.01% | +74.01 points |
| Raster-object precision | 73.12% +/- 2.67% | 0.00% +/- 0.00% | +73.12 points |
| Raster-object recall | 99.36% +/- 0.10% | 0.00% +/- 0.00% | +99.36 points |
| Raster-object F1 | 84.23% +/- 1.77% | 0.00% +/- 0.00% | +84.23 points |
| Exact-vector owner recall | 87.19% +/- 0.64% | 3.76% +/- 0.04% | +83.43 points |

The selected U-Net masks place matched exact violations within 28.2 nm mean
centroid error (8.3 nm median) and 5.44 nm mean edge-pair-bisector error across
the three run summaries. These are tile-level mask-to-vector matching results;
they are not yet recovered sign-off edges.

## Acceptance decision

All pre-registered development gates passed:

| Gate | Required | Result |
|---|---:|---:|
| Dirty-mask Dice | >= 75% | 86.32% |
| Raster-object F1 | >= 75% | 84.23% |
| Exact-vector owner recall | >= 85% | 87.19% |
| Dirty recall versus B2 | no more than 2 points lower | 2.67 points higher |

The improvement is stable across seeds and classification also improves on the
same tiles. B6.2 is therefore accepted as the localization model for B7.

## Limits carried into B7

- Object recall is very high (99.36%), but object precision is 73.12%. B7 must
  merge duplicate/fragmented components and measure false alarms at natural
  full-layout prevalence.
- `tt_um_c4m_spsram_direct` is the weakest development family: 78.95% mean
  Dice, 78.52% object F1, 83.25% exact-vector recall, and 89.41% classification
  accuracy. B7 must report this family separately rather than hiding it in the
  aggregate.
- Segmentation thresholds differ by seed (0.1, 0.1, and 0.4), so B7 must freeze
  a validation-only deployment policy before full-layout confirmation.
- Full-layout stitching, unique-prediction precision, exact M1 edge recovery,
  false alarms per area, runtime, and the untouched B9 holdout remain open.

## Versioned evidence

- `summary.json`: aggregate U-Net/B2 comparison and acceptance decision.
- `runs/seed_42.json`, `runs/seed_43.json`, `runs/seed_44.json`: configurations,
  learning curves, selected thresholds, checkpoint hashes, per-layout results,
  coordinate errors, and severity/boundary slices.
- Checkpoint binaries remain in Drive and are identified by SHA-256 in each run
  record.
