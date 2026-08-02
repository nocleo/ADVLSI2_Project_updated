# Project presentation

`ADVLSI2_DRC_CNN_Project_Progress.pptx` is the versioned project presentation used for professor reviews.

Convenience Google Slides copy: [Google Slides](https://docs.google.com/presentation/d/1DpISDh2x72WqxdPcGVeVSKl17FaQtV45CW2P3KQnXDY/edit)

The versioned PPTX in this directory is the authoritative phase snapshot.

At the end of every project phase:

1. Update the phase dashboard and the delta from the previous phase.
2. Add the experiment setup, primary metrics, and relevant visual evidence.
3. Update this PPTX path and refresh the convenience Slides copy when needed.
4. Commit the presentation together with the phase results.

The current 22-slide deck includes the completed B2 three-seed baselines:
92.47% +/- 0.61% tile-random reference accuracy and 90.38% +/- 0.84%
accuracy on unseen layout families. The paper's reported result remains an
external reference because the datasets and protocols differ. It also records
B3's validation gains and failed frozen-test confirmation: calibrated dirty
recall reached 94.42%, but unseen-layout accuracy regressed to 89.71%. B2
therefore remains the accepted baseline and B4 architecture experiments are
next.
