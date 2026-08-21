# B7 full-layout stitching and exact-coordinate recovery

Status: **original B7 rejected; B7.1 internal policy accepted; B7.2
competitiveness gate failed and the detector branch is closed**.

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

## B7.1 measured result

Validation-only selection froze this policy:

- segmentation threshold: `0.4`;
- classification threshold: `0.92`;
- minimum merged component area: `16` pixels;
- fragment merge gap: `2` pixels;
- exact-recovery radius: `140` nm.

| Metric | Validation | Development confirmation | Gate |
|---|---:|---:|---:|
| Unique violation recall | 95.33% | **95.51%** | >=85% |
| Candidate-component precision | 86.85% | **81.44%** | >=80% |
| Component F1 | 90.90% | 87.92% | report |
| Exact recovered-pair precision | 100.00% | 100.00% | report |
| False detections/mm2 | 345.75 | 1122.07 | report |
| False-positive tiles/million | 41.0 | 199.1 | report |
| Clean layouts incorrectly flagged | 0 | 1 | report |

All registered B7.1 gates passed. Validation selected the deployment policy;
the three development families were recomputed once and are sequential
confirmation rather than a final holdout. The untouched B9 holdout was not
opened. Full machine-readable evidence is in [`b7_1_summary.json`](b7_1_summary.json),
while [`b7_original_failure_summary.json`](b7_original_failure_summary.json)
preserves the rejected original result.

Run [`../../notebooks/B7_Full_Layout_Stitching.ipynb`](../../notebooks/B7_Full_Layout_Stitching.ipynb).
The notebook uses `--reuse-scans-only`, so a missing or stale cache causes an
immediate failure instead of another multi-hour inference scan.

## B7.2 measured competitiveness result

B7.2 reran the frozen B7.1 policy with synchronized CUDA timing and compared it
with exact KLayout on the same six layouts and both source/injected variants.
KLayout used one warm-up and five measured repetitions. The CNN has one
complete timing repetition, so its p95 is unavailable; this does not change the
decision because recall and median speed both fail by large margins.

| Split | CNN total | KLayout median | KLayout conservative p95 | KLayout faster by | CNN recall |
|---|---:|---:|---:|---:|---:|
| Validation | 4,693.87 s | 18.35 s | 21.15 s | 255.85x | 95.33% |
| Development confirmation | 1,392.87 s | 8.59 s | 10.65 s | 162.21x | 95.51% |

The hard gate required at least 2x CNN speedup, at least 99.5% violation recall,
no registered critical/near-threshold miss, no clean-layout false alarm, exact
recovered-pair precision of one, and lower p95 latency. Validation missed
near-threshold and severe-slice violations. Development confirmation missed
near-threshold violations and flagged one clean layout. The comparison boundary
favored the CNN by excluding model/layout load and result serialization, while
KLayout included parsing, M1 materialization, exact checking, and RDB writing.

Exact recovered-pair precision remained 100% only for CNN-proposed candidates
because KLayout performs local exact recovery. It does not recover the 4.5–4.7%
of true violations that the CNN never proposed. The B7.2 hard gate therefore
failed; no optimized Stage-B benchmark or further detector tuning is warranted.
B9 remains unopened, and the next experiment is the preregistered B8.0
OpenROAD actionability pilot.

Machine-readable evidence:

- [`b7_2_cnn_gpu_summary.json`](b7_2_cnn_gpu_summary.json)
- [`b7_2_klayout_competitiveness_summary.json`](b7_2_klayout_competitiveness_summary.json)
