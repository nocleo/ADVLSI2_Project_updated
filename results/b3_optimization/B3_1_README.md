# B3 Training Optimization

Status: **FAILED_SEARCH**  
Manifest: `9deef1271a145198a60abccf291ec499d53226075b4ccfa2061a215afbd01472`  
Selection: unseen-layout validation dirty F1 only; frozen test data was not used.

## Optimizer stage (learning rate 0.001)

| Candidate | Validation dirty F1 | Validation dirty recall |
|---|---:|---:|
| `rmsprop_lr_0p001` | 91.36% +/- 0.46% | 88.60% +/- 1.19% |
| `adam_lr_0p001` | 89.84% +/- 3.12% | 89.38% +/- 2.29% |

## Learning-rate stage

| Candidate | Validation dirty F1 | Validation dirty recall |
|---|---:|---:|
| `rmsprop_lr_0p0003` | 91.08% +/- 0.86% | 87.65% +/- 1.71% |
| `rmsprop_lr_0p001` | 91.36% +/- 0.46% | 88.60% +/- 1.19% |
| `rmsprop_lr_0p003` | 83.75% +/- 10.03% | 91.47% +/- 7.39% |

Selected: `rmsprop_lr_0p001`; paired validation-F1 wins: 0/3.

## Search acceptance issues

- selected mean validation dirty F1 did not improve over B2
- selected validation dirty F1 improved on only 0/3 paired seeds
