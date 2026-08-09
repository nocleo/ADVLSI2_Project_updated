# B6.1 coverage-correct localization dataset

Status: **accepted; proceed to B6.2 multi-task U-Net training**.

B6.1 removes the old tiling blind band by using a 1600 nm input with a 160 nm
halo around a 1280 nm central output and a matching 1280 nm stride. The input
and output rasters use the same 8 nm/pixel scale (200x200 input, 160x160 mask).

## Official registry result

| Measure | Result |
|---|---:|
| Layout families | 14 |
| Exact KLayout `m1.2` edge pairs | 6,924 |
| Unique violation owners | 6,924 |
| Dirty localization tiles | 8,021 |
| Balanced clean tiles | 8,021 |
| Total tiles | 16,042 |
| Owned one-pixel raster surrogates | 0 |
| Omitted non-owner subpixel fragments | 0 |

Every layout passed the clean, dense, sparse, horizontal, vertical, boundary,
and near-threshold visual-audit categories. Raster masks are training targets;
the stored edge pairs, measured spacing, and rule deficits are authoritative
for coordinate and physical-error evaluation.

`summary.json` is the compact machine-readable record. The generated dataset
and archive are ignored because they are reproducible. The official local
archive is identified by the SHA-256 recorded in that summary.

## Reproduction

```bash
python -m pip install -r requirements.txt
python scripts/build_b6_localization_dataset.py
```

The registry runner is resume-safe. It uses existing injected layouts and RDB
reports when present, creates deterministic missing intermediates, and stops if
exact ownership, balanced sampling, or visual-audit gates fail.
