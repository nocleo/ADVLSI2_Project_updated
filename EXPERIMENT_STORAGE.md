# Experiment storage and rerun map

All Colab notebooks and persistent artifacts use one Google Drive root:

```text
/content/drive/MyDrive/ADVLSI2 2026 Project
```

Drive folder: [ADVLSI2 2026 Project](https://drive.google.com/drive/folders/1WVKzwWkh-onnEnmSKJpF4Wp-NSkbjuBw)

## Canonical structure

```text
ADVLSI2 2026 Project/
├── notebooks/
│   ├── B2_Dual_Baselines.ipynb
│   ├── B3_Training_Optimization.ipynb
│   ├── B4_Compact_Architecture.ipynb
│   ├── B5_Ensemble.ipynb
│   ├── B5_2_Failure_Slice_Audit.ipynb
│   ├── B6_1_Localization_Dataset.ipynb
│   ├── B6_2_Multitask_UNet.ipynb
│   ├── B7_Full_Layout_Stitching.ipynb
│   ├── B7_2_KLayout_Competitiveness.ipynb
│   └── legacy_notebooks/
├── datasets/
│   └── b6_localization_dataset.zip
├── experiments/
│   ├── B2_baselines/b2_baselines/
│   ├── B3_training_optimization/b3_optimization/
│   ├── B4_compact_architecture/b4_architecture/
│   ├── B5_ensemble/b5_ensemble/
│   ├── B5_2_failure_audit/
│   ├── B6_localization/
│   ├── B7_full_layout/
│   │   ├── b7_2_cnn_gpu/<UTC run tag>/
│   │   └── b7_2_klayout_benchmark/<UTC run tag>/
│   └── B8_action_control/
│       └── b8_0_actionability/<protocol hash>/
└── reports/
    ├── ADVLSI2 DRC CNN — Project Progress
    └── proposal presentation
```

The nested directories under B2–B5 are retained so existing checkpoints and
results keep their identities. New experiments must be created below
`experiments/`; notebooks, datasets, and reports must not write new top-level
folders in `My Drive`.

Pre-existing folders such as `Research Papers`, `Study Resources`,
`Open-Source EDA Tools`, and `TinyTapeout Layouts` remain under the project root
as reference inputs. They are not experiment-output locations.

## Canonical notebooks

| Phase | Notebook |
|---|---|
| B2 | [Dual baselines](https://colab.research.google.com/drive/1chfOdoAjGRbmAHXOUIwyWQkFCtBG8xXE) |
| B3 | [Training optimization](https://colab.research.google.com/drive/1hCS76kcLak7SuC54aKrGqpV7XO1EhhgT) |
| B4 | [Compact architecture](https://colab.research.google.com/drive/135X-NvQrBsjOedE497bTGXpiUaFFedR9) |
| B5 | [Ensemble](https://colab.research.google.com/drive/1JdqLB5nBGRLduNsgF58hHSH3mgmy1QG8) |
| B5.2 | [Failure-slice audit](https://colab.research.google.com/drive/1E6jGk5DzY80IeM2hYdjeiL6XpZeEuB-O) |
| B6.1 | [Localization dataset](https://colab.research.google.com/drive/1XvRYnfTmZieAvu5EymiXVk524QLXyIXV) |
| B6.2 | [Multi-task U-Net](https://colab.research.google.com/drive/16WKJV8nePYJrY_KVPQDqVM-CUAHPgWOF) |
| B7 | [Full-layout stitching](https://colab.research.google.com/drive/1nZlAcR6YDWxXfIX-P5OlxZPaY4DTNRqA) |
| B7.2 | [KLayout competitiveness audit](https://colab.research.google.com/drive/1U6EBCcYkDBqJBDk6xDoGK9MRhkeLuaB2) |

## Path contract

Every canonical notebook defines the project root after mounting Drive and
derives all other persistent paths from it:

```python
from pathlib import Path

PROJECT_DRIVE = Path('/content/drive/MyDrive/ADVLSI2 2026 Project')
assert PROJECT_DRIVE.is_dir(), (
    f'Missing canonical project root: {PROJECT_DRIVE}. '
    'Do not create an empty replacement with the same name.'
)
NOTEBOOKS_ROOT = PROJECT_DRIVE / 'notebooks'
EXPERIMENTS_ROOT = PROJECT_DRIVE / 'experiments'
DATASETS_ROOT = PROJECT_DRIVE / 'datasets'
REPORTS_ROOT = PROJECT_DRIVE / 'reports'
```

Temporary archives may still be created under `/content` for download, but the
authoritative inputs, checkpoints, caches, metrics, and reports live under the
Drive root above.

The project folder must already exist directly in **My Drive** before a
notebook creates phase output directories. This guard prevents Colab from
silently producing a second, empty `ADVLSI2 2026 Project` folder when the
canonical folder is only visible under **Shared with me**.

## Rerun order

The historical pipeline can be reproduced in phase order B2 → B3 → B4 → B5 →
B5.2 → B6.1 → B6.2 → B7. A phase may be skipped when its manifest- and
hash-verified artifacts already exist. B7 depends on the accepted B6.2
checkpoints and B6 localization dataset. B3 and B5 depend on the accepted B2
checkpoints; B5 also uses B4 checkpoints.

B7.2 is complete and failed its accelerator hard gate. Its synchronized GPU
result is below
`experiments/B7_full_layout/b7_2_cnn_gpu/20260817T195310Z/`; the exact KLayout
comparison is in the canonical run referenced by that phase's
`LATEST_RUN.txt`. The resumable launcher reuses only hash-matched completed
validation scans and incrementally materialized development components; it
does not silently accept stale checkpoints or layout/report caches. No B7.2
rerun or CNN retraining is planned.

The next execution phase is the B8.0 OpenROAD actionability pilot defined in
[`B8_ACTIONABILITY_PROTOCOL.md`](B8_ACTIONABILITY_PROTOCOL.md). Its
implementation must write one hash-bound record per full flow below
`experiments/B8_action_control/b8_0_actionability/<protocol hash>/`, persist
after every run, and resume without keeping the 126-run matrix in notebook
memory. Run the nine-flow harness smoke test before the full matrix.
