# B6.2 multi-task U-Net

Status: **implementation complete; official three-seed GPU result pending**.

The executable protocol is in `training/train_multitask_unet.py`, with the
repository launcher at `scripts/run_b6_multitask_unet.py` and the Colab/Drive
runner at `notebooks/B6_2_Multitask_UNet.ipynb`.

The official run must use seeds 42, 43, and 44, the fixed family-disjoint
`unseen_layout_v1` split, validation-only checkpoint and threshold selection,
and the authoritative B2 checkpoints from `ADVLSI2_B2/b2_baselines/checkpoints`
for a same-B6-tile classification comparison.

This directory will receive the compact run JSON, aggregate summary, and final
decision after the Colab result is uploaded. Checkpoints remain in Drive and
are identified by SHA-256 rather than committed.
