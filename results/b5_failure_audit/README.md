# B5.2 Classifier Failure-Slice Audit

Status: **complete — classifier-only tuning closed**

This report uses authoritative B2/B4 training and validation predictions only. It does not read either existing test split or the new final holdout.

## Mean validation results

| Protocol | Model | Accuracy | Dirty recall | Dirty F1 | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| `unseen_layout_v1` | `NCSU_DRCNN` (B2) | 90.52% | 88.60% | 91.36% | 0.0655 | 0.0270 |
| `unseen_layout_v1` | `CompactBNPool` (B4) | 92.87% | 92.37% | 93.61% | 0.0489 | 0.0156 |
| `tile_random_reference` | `NCSU_DRCNN` (B2) | 92.17% | 92.15% | 91.50% | 0.0520 | 0.0245 |
| `tile_random_reference` | `CompactBNPool` (B4) | 94.23% | 91.27% | 93.53% | 0.0395 | 0.0125 |

B4 is stronger on these validation metrics, but it remains rejected because the already-recorded B4 frozen confirmation lost unseen-layout accuracy/F1 and reduced tile-reference dirty recall beyond tolerance. B2 therefore remains the accepted classifier.

## Evidence gates

- `unseen_layout_v1` class-conditional disagreement: **fail**; neither B2's dirty advantage nor B4's clean advantage repeats by family across seeds.
- `tile_random_reference` class-conditional disagreement: **pass**; B2's dirty advantage repeats in 6 families and B4's clean advantage in 12.
- Joint disagreement gate: **fail**, so an ensemble/uncertainty B5.3 is not justified.
- Density gate after enforcing both declared requirements—at least two families and two seeds: **10 eligible seed-level candidates across 4 repeated signatures**.

All four repeated density signatures concern clean samples:

| Protocol | Models | Density bin | Seeds |
|---|---|---|---|
| `unseen_layout_v1` | B2 and B4 | `0.15-0.30` | 42, 43, 44 |
| `tile_random_reference` | B2 and B4 | `0.03-0.15` | 43, 44 |

The affected density direction differs by protocol. Density is therefore a useful diagnostic covariate, but it is not a sufficiently specific, protocol-stable failure mechanism for another classifier-only experiment.

## Feature availability

- Available for all 68,100 aligned records: layout family and metal density.
- Unavailable: exact violation count, edge orientation/length, spacing deficit, nearby-shape count, and distance to tile/supervised boundaries.

No boundary, orientation, scale, or severity claim is made from raster pixels. Those fields will be generated from exact edge-pair geometry in B6.1.

## Decision

- Close B5.3 without another classifier experiment.
- Retain `NCSU_DRCNN` B2 as the accepted classifier.
- Proceed to B6.1: gap-free tiling and vector-backed violation masks, then B6.2 multi-task localization/classification.

The complete machine-readable audit is in `summary.json`. The 33 MB path-level `records.jsonl` remains an external reproducibility artifact and is intentionally not committed.
