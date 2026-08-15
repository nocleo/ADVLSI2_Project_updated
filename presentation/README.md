# Project presentation

`ADVLSI2_DRC_CNN_Project_Progress.pptx` is the versioned project presentation used for professor reviews.

Convenience Google Slides copy: [Google Slides](https://docs.google.com/presentation/d/1DpISDh2x72WqxdPcGVeVSKl17FaQtV45CW2P3KQnXDY/edit)

The versioned PPTX in this directory is the authoritative phase snapshot.

At the end of every project phase:

1. Update the phase dashboard and the delta from the previous phase.
2. Add the experiment setup, primary metrics, and relevant visual evidence.
3. Update this PPTX path and refresh the convenience Slides copy when needed.
4. Commit the presentation together with the phase results.

The current 29-slide deck includes the completed B2 three-seed baselines:
92.47% +/- 0.61% tile-random reference accuracy and 90.38% +/- 0.84%
accuracy on unseen layout families. The paper's reported result remains an
external reference because the datasets and protocols differ. It also records
B3's validation gains and failed frozen-test confirmation: calibrated dirty
recall reached 94.42%, but unseen-layout accuracy regressed to 89.71%. B2
therefore remained the accepted baseline. It also records B4's completed
compact-architecture experiment: validation dirty F1 reached 93.61% and the
model used 14.3x fewer parameters, but unseen-layout accuracy/F1 regressed to
89.36%/90.36% and tile-reference recall fell to 91.36%. B2 remains accepted.
It also records B5.1's completed ensemble experiment: the 25% B2 / 75% B4
blend improved unseen-layout accuracy/F1 to 90.47%/91.25%, but tile-reference
recall fell to 92.07%, beyond tolerance. The ensemble was rejected. B5.2 then
closed classifier-only tuning after the B2/B4 disagreement mechanism failed to
repeat across both validation protocols; no B5.3 training run is planned.

B6.1 is complete. The deck records the gap-free localization dataset across
14 layout families: 6,924 exact KLayout violations with one unique owner each,
8,021 dirty and 8,021 balanced clean tiles, 1600 nm contextual inputs with
160 nm halos, and 1280 nm central outputs. It now also records B6.2's accepted
three-seed multi-task U-Net result: 95.51% classification accuracy, 98.86%
dirty recall, 86.32% mask Dice, 84.23% raster-object F1, and 87.19%
exact-vector owner recall on unseen development-confirmation layouts. The deck
does not overstate this as sign-off localization: object precision is 73.12%,
and the SRAM family remains weakest. It now also records the accepted B7.1
cache-only policy correction: classification threshold 0.92, segmentation
threshold 0.4, 95.51% development violation recall, 81.44% candidate precision,
87.92% component F1, and 100% exact-pair precision. DRC/connectivity-verified
repair follows in B8, and the untouched final holdout remains B9.
