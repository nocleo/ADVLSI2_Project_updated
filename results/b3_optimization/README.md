# B3.2/B3.3 Extension

Status: **FAILED_CONFIRMATION**  
Selection used validation data only; frozen tests were locked until the validation gate passed.

| Stage | Validation dirty F1 | Validation dirty recall | Mean epochs |
|---|---:|---:|---:|
| B2 baseline | 91.36% +/- 0.46% | 88.60% +/- 1.19% | 30.0 |
| Plateau scheduler | 91.78% +/- 0.89% | 89.81% +/- 1.00% | 30.0 |
| Early stopping | 91.78% +/- 0.89% | 89.81% +/- 1.00% | 26.0 |

Selected recipe: `plateau_full`.
Calibrated validation dirty F1: 92.08% +/- 0.94%.

Benefits:
- quality gain: **True**
- recall gain: **True**
- lower seed variance: **False**
- at least 25 percent fewer epochs: **False**

## Frozen-test confirmation

| Protocol | Accuracy | Dirty recall | Dirty F1 |
|---|---:|---:|---:|
| `unseen_layout_v1` | 89.71% +/- 0.89% | 94.42% +/- 0.93% | 90.72% +/- 0.67% |
| `tile_random_reference` | 92.52% +/- 0.98% | 93.70% +/- 0.63% | 91.97% +/- 1.02% |
