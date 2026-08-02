# B2 Dual Classification Baselines

Status: **PASSED**  
Configuration: `d7fa939a1212`  
Manifest: `9deef1271a145198a60abccf291ec499d53226075b4ccfa2061a215afbd01472`

| Protocol | Accuracy | Dirty precision | Dirty recall | Dirty F1 | Seeds |
|---|---:|---:|---:|---:|---|
| `tile_random_reference` | 92.47% ± 0.61% | 90.46% ± 0.71% | 93.39% ± 1.98% | 91.89% ± 0.75% | 42, 43, 44 |
| `unseen_layout_v1` | 90.38% ± 0.84% | 91.22% ± 2.26% | 90.73% ± 1.95% | 90.94% ± 0.72% | 42, 43, 44 |

## tile_random_reference: per-layout test metrics

| Layout | Samples | Accuracy | Dirty recall | Dirty F1 |
|---|---:|---:|---:|---:|
| `tt_um_2048_vga_game` | 54 | 87.65% ± 1.07% | 90.74% ± 3.21% | 83.03% ± 1.70% |
| `tt_um_8_bit_cpu` | 46 | 88.41% ± 1.26% | 88.46% ± 0.00% | 89.62% ± 1.00% |
| `tt_um_Bingyao_FCOTA` | 65 | 95.38% ± 1.54% | 90.48% ± 4.12% | 94.38% ± 2.00% |
| `tt_um_TSARKA_TinyQV` | 66 | 97.47% ± 2.31% | 97.62% ± 2.06% | 97.06% ± 2.69% |
| `tt_um_aes_sbox` | 55 | 92.73% ± 1.82% | 96.00% ± 4.00% | 92.28% ± 2.07% |
| `tt_um_analog_atenfyr1` | 44 | 95.45% ± 2.27% | 92.06% ± 2.75% | 95.08% ± 2.44% |
| `tt_um_c4m_spsram_direct` | 62 | 97.85% ± 2.46% | 98.25% ± 3.04% | 96.58% ± 3.92% |
| `tt_um_cmos_inverter` | 56 | 94.05% ± 1.03% | 94.20% ± 2.51% | 92.87% ± 1.05% |
| `tt_um_essen` | 32 | 92.71% ± 1.80% | 90.67% ± 2.31% | 95.09% ± 1.28% |
| `tt_um_fabulous_sky_26a` | 71 | 89.67% ± 0.81% | 92.71% ± 1.80% | 89.01% ± 0.69% |
| `tt_um_fft_adityaamehra` | 55 | 89.09% ± 3.15% | 89.39% ± 5.25% | 86.73% ± 3.96% |
| `tt_um_irfantekin_analog` | 41 | 89.43% ± 5.08% | 95.24% ± 4.76% | 90.24% ± 4.69% |
| `tt_um_jyblue1001_pll` | 33 | 87.88% ± 5.25% | 100.00% ± 0.00% | 87.67% ± 4.60% |
| `tt_um_yen` | 37 | 92.79% ± 4.13% | 94.87% ± 4.44% | 94.85% ± 2.97% |

Pooled test confusion matrix across seeds:

| Actual / predicted | Clean | Dirty |
|---|---:|---:|
| Clean | 1070 | 97 |
| Dirty | 65 | 919 |

## unseen_layout_v1: per-layout test metrics

| Layout | Samples | Accuracy | Dirty recall | Dirty F1 |
|---|---:|---:|---:|---:|
| `tt_um_2048_vga_game` | 1136 | 89.76% ± 0.79% | 85.30% ± 2.07% | 88.67% ± 0.54% |
| `tt_um_Bingyao_FCOTA` | 733 | 92.91% ± 0.59% | 93.46% ± 2.13% | 93.96% ± 0.61% |
| `tt_um_c4m_spsram_direct` | 813 | 88.97% ± 2.29% | 94.43% ± 1.97% | 90.68% ± 1.83% |

Pooled test confusion matrix across seeds:

| Actual / predicted | Clean | Dirty |
|---|---:|---:|
| Clean | 3388 | 377 |
| Dirty | 397 | 3884 |

