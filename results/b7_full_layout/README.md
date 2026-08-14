# B7 full-layout stitching and exact-coordinate recovery

Status: **original B7 rejected; B7.1 cache-based correction pending**.

The runner averages the accepted B6.2 seeds 42/43/44, scans every central
output of each complete layout at natural prevalence, and includes the original
clean source beside every injected layout. The B9 final holdout is not read.

## Original B7 result

The original validation-only F1 selector froze this policy:

- segmentation threshold: `0.4`;
- classification threshold: `0.8`;
- minimum merged component area: `16` pixels;
- fragment merge gap: `2` pixels;
- exact-recovery radius: `140` nm.

| Metric | Validation | Development confirmation | Gate |
|---|---:|---:|---:|
| Unique violation recall | 97.66% | 97.88% | >=85% |
| Candidate-component precision | 85.37% | **62.57%** | >=80% |
| Component F1 | 91.11% | 76.34% | report |
| Exact recovered-pair precision | 100.00% | 100.00% | report |
| False detections/mm2 | 400.05 | 3019.51 | report |
| False-positive tiles/million | 117.3 | 2462.1 | report |
| Clean layouts incorrectly flagged | 1 | 1 | report |

B7 fails only the development component-precision gate. All 33 missed
development violations are in the near-threshold severity slice; medium and
severe violations reached 100% recall. The failure is concentrated in
`tt_um_c4m_spsram_direct`: its injected/source pair contributes 686 of 810
false development components, including 301 false components on the clean
source. The other two development families combine to 87.41% component
precision. Stitching and exact-coordinate recovery therefore work, while the
tile classification gate generalizes poorly to repeated SRAM geometry.

## B7.1 protocol

B7.1 is a controlled validation-policy correction, not a model retrain:

1. preserve the original B7 headline and validation selection in Drive history;
2. reuse only complete scan caches whose layout, report, and three checkpoint
   hashes match;
3. freeze segmentation at the original validation-selected `0.4` and keep the
   original area and merge-gap candidate sets;
4. extend classification thresholds densely from `0.80` through `0.99`;
5. select maximum validation component precision subject to at least 95%
   validation violation recall;
6. freeze that policy and recompute development confirmation once;
7. retain the original B7 acceptance gates: at least 85% development recall
   and at least 80% development component precision.

The development layouts have already informed this correction and are not a
final holdout. B9 remains sealed and cannot influence B7.1.

Run [`../../notebooks/B7_Full_Layout_Stitching.ipynb`](../../notebooks/B7_Full_Layout_Stitching.ipynb).
The notebook uses `--reuse-scans-only`, so a missing or stale cache causes an
immediate failure instead of another multi-hour inference scan. Upload
`ADVLSI2_B7_1_results.zip` after completion. The compact measured evidence and
presentation update must be added before PR #11 is eligible to merge.
