# B7 full-layout stitching and exact-coordinate recovery

Status: **implementation and protocol ready; measured GPU result pending**.

The B7 runner averages the accepted B6.2 seeds 42/43/44, scans every central
output of each complete layout at natural prevalence, and selects one deployment
policy on the three validation families plus their original source variants.
The selected policy is frozen before development confirmation. The B9 final
holdout is not read.

Pre-registered acceptance gates:

- complete-grid scans and passing synthetic coordinate round trips;
- at least 85% validation and development exact-violation recall;
- at least 80% development candidate-component precision;
- explicit false detections/mm2, false-positive tiles/million, clean-layout
  flags, severity/boundary recall, merge counts, memory, and runtime.

The useful reporting ideas adapted from the teammate Colab are four-panel
input/ground-truth/probability/overlay images, validation threshold trade-off
plots, and one numerical record per component with centroid, bounding box, mean
confidence, and maximum confidence. The teammate checkpoint and random tile
evaluation are not used.

Run [`../../notebooks/B7_Full_Layout_Stitching.ipynb`](../../notebooks/B7_Full_Layout_Stitching.ipynb)
with a GPU runtime. Upload `ADVLSI2_B7_results.zip` after completion. The compact
measured JSON/Markdown evidence and presentation update will be added before the
B7 PR is eligible to merge.

Drive keeps hash-verified injected-layout and sparse per-layout scan caches so
an interrupted run resumes without repeating completed inference. Those caches
are excluded from the downloaded result archive and repository.
