# Third-party layout sources

B1 layout sources are pinned in `layout_registry.json` to commits in
[`TinyTapeout/tinytapeout-sky-26a`](https://github.com/TinyTapeout/tinytapeout-sky-26a).
Each selected project contains its own Apache-2.0 `LICENSE` file. The
acquisition script verifies both the pinned OAS SHA-256 digest and the expected
license text before writing a layout into `real_layouts_tt/`.

The existing `tt_um_yen_1err` file is a local error-derived variant of
`tt_um_yen`; it is grouped with that parent family and excluded from the B1
dataset rebuild because the generation flow injects deterministic errors into
the canonical source layout.

Run:

```bash
python scripts/acquire_b1_layouts.py
```

The command acquires all nine reviewed candidates so their pinned sources stay
available. All nine are admitted to the accepted B1 rebuild: AES S-box, SRAM,
analog oscillator, OTA, PLL, 2048 VGA, FPGA fabric, RISC-V SoC, and FFT. The
targeted tiler enumerates dirty windows and samples clean windows
deterministically, which makes the three larger layouts practical without
rasterizing every empty grid window.

Existing files are not overwritten. Use `--include-existing` to verify the
canonical source set and `--replace-mismatched` only after reviewing a reported
mismatch.
