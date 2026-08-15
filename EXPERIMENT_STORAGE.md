# Experiment storage and rerun map

All Colab notebooks and persistent artifacts use one Google Drive root:

```text
/content/drive/MyDrive/ADVLSI2 2026 Project
```

Drive folder: [ADVLSI2 2026 Project](https://drive.google.com/drive/folders/1gGH6ETL-xT1HUYrNv_T5ScPzio9bQXLf)

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
│   └── B7_full_layout/
│       ├── b7_2_cnn_gpu/<UTC run tag>/
│       └── b7_2_klayout_benchmark/<UTC run tag>/
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
| B7.2 | [`B7_2_KLayout_Competitiveness.ipynb`](notebooks/B7_2_KLayout_Competitiveness.ipynb) (repository launcher; copy it to the Drive notebook folder before execution) |

## Path contract

Every canonical notebook defines the project root after mounting Drive and
derives all other persistent paths from it:

```python
from pathlib import Path

PROJECT_DRIVE = Path('/content/drive/MyDrive/ADVLSI2 2026 Project')
NOTEBOOKS_ROOT = PROJECT_DRIVE / 'notebooks'
EXPERIMENTS_ROOT = PROJECT_DRIVE / 'experiments'
DATASETS_ROOT = PROJECT_DRIVE / 'datasets'
REPORTS_ROOT = PROJECT_DRIVE / 'reports'
```

Temporary archives may still be created under `/content` for download, but the
authoritative inputs, checkpoints, caches, metrics, and reports live under the
Drive root above.

## Rerun order

The historical pipeline can be reproduced in phase order B2 → B3 → B4 → B5 →
B5.2 → B6.1 → B6.2 → B7. A phase may be skipped when its manifest- and
hash-verified artifacts already exist. B7 depends on the accepted B6.2
checkpoints and B6 localization dataset. B3 and B5 depend on the accepted B2
checkpoints; B5 also uses B4 checkpoints.

The next research phase is not another training run. Run the B7.2 KLayout
competitiveness audit defined in [MODEL_ROADMAP.md](MODEL_ROADMAP.md) first. It
writes each fresh synchronized GPU result below
`experiments/B7_full_layout/b7_2_cnn_gpu/<UTC run tag>/` and exact comparison
evidence below
`experiments/B7_full_layout/b7_2_klayout_benchmark/<UTC run tag>/`. Each root
contains `LATEST_RUN.txt`; reruns never silently reuse a previous scan cache.
