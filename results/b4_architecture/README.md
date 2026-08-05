# B4 Compact Architecture Experiment

Status: **completed; candidate rejected**

Frozen tests unlocked: **yes**
Accepted production baseline: **B2 `NCSU_DRCNN`**

## Controlled change

B4 changed only the classifier architecture. `CompactBNPool` used four
convolution/batch-normalization blocks and concatenated global average/maximum
pooling. The B1 manifest and protocols, seeds 42/43/44, 30-epoch budget,
RMSprop at `0.001`, batch size 32, train-only Manhattan augmentation,
best-validation-loss selection, and threshold `0.5` remained fixed.

## Validation-only selection

| Model | Accuracy | Dirty recall | Dirty F1 |
|---|---:|---:|---:|
| B2 `NCSU_DRCNN` | 90.52% | 88.60% | 91.36% |
| `CompactBNPool` | **92.87%** | **92.37%** | **93.61%** |

The candidate passed the predeclared validation gate and improved dirty F1 in
all three paired seeds, so frozen-test evaluation was unlocked.

## Frozen-test confirmation

| Protocol / model | Accuracy | Dirty recall | Dirty F1 |
|---|---:|---:|---:|
| Unseen layouts — B2 | **90.38%** | 90.73% | **90.94%** |
| Unseen layouts — compact | 89.36% | **93.72%** | 90.36% |
| Tile reference — B2 | 92.47% | **93.39%** | 91.89% |
| Tile reference — compact | **94.42%** | 91.36% | **93.74%** |

The candidate failed the final gate. On unseen layouts, accuracy regressed by
1.02 points and dirty F1 regressed by 0.58 points, with zero paired-seed F1
wins. On the tile reference, dirty recall regressed by 2.03 points, beyond the
0.5-point tolerance.

## Deployment cost

| Model | Parameters | State dict | PyTorch CPU median | ONNX CPU median |
|---|---:|---:|---:|---:|
| B2 `NCSU_DRCNN` | 602,114 | 2,413,493 B | 14.06 ms | **1.92 ms** |
| `CompactBNPool` | **42,178** | **179,029 B** | **8.63 ms** | 2.19 ms |

`CompactBNPool` is approximately 14.3x smaller, reduces PyTorch CPU latency by
38.6%, and remains within the ONNX latency limit. This is useful compression
evidence, but it does not override the frozen unseen-layout quality gate.

## Decision

B2 remains the accepted single-model classifier. B4's higher unseen-layout
recall and B2's higher precision/F1 motivate a validation-only probability
ensemble or agreement-gate experiment in B5 before localization. The compact
architecture is also retained as a documented deployment candidate for
possible post-localization compression, not as the single-model baseline.

Authoritative evidence is in `summary.json`; it includes all aggregate,
per-seed, per-layout, gate, cost, runtime, source-hash, and checkpoint-hash
records. `architecture_benchmark.json` is the paired cost subset.
