# B5 Validation-Selected B2+B4 Ensemble

Status: **rejected**
Frozen tests unlocked: **True**

## Validation-only search

| Candidate | Accuracy | Dirty recall | Dirty F1 |
|---|---:|---:|---:|
| `NCSU_DRCNN` | 90.52% | 88.60% | 91.36% |
| `CompactBNPool` | 92.87% | 92.37% | 93.61% |
| B2 0.25 / B4 0.75 (selected) | 93.20% | 92.21% | 93.87% |
| B2 0.50 / B4 0.50 | 92.84% | 91.36% | 93.52% |
| B2 0.75 / B4 0.25 | 91.70% | 89.85% | 92.45% |

## Decision

- `unseen_layout_v1` ensemble: accuracy 90.47%, recall 93.39%, F1 91.25%. Gate passed: **True**.
- `tile_random_reference` ensemble: accuracy 94.70%, recall 92.07%, F1 94.08%. Gate passed: **False**.

The ensemble is accepted only if both frozen protocols pass; otherwise B2 remains the classifier baseline.
