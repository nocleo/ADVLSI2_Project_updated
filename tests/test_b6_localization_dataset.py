import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generate_training_dataset_scripts"))

try:
    import klayout.db as db

    from generate_localization_dataset import (
        CoverageGrid,
        LocalizationConfig,
        _apply_owned_subpixel_surrogate,
        build_layout_dataset,
        extract_exact_violations,
        nm_to_pixel,
        pixel_to_nm,
        rasterize_registered_tile,
    )
except ModuleNotFoundError:
    db = None


@unittest.skipIf(db is None, "KLayout is not installed in the lightweight test runtime")
class B6LocalizationDatasetTest(unittest.TestCase):
    @staticmethod
    def spacing_region():
        region = db.Region()
        region.insert(db.Box(200, 100, 400, 700))
        region.insert(db.Box(500, 100, 700, 700))
        region.merge()
        return region

    def test_localization_config_is_gap_free_and_registered(self):
        config = LocalizationConfig()
        config.validate()
        self.assertEqual(config.input_nm, 1600)
        self.assertEqual(config.output_nm, config.stride_nm)
        self.assertEqual(config.output_nm, 1280)
        self.assertEqual(config.halo_nm, 160)
        self.assertEqual(config.nm_per_pixel, 8)
        self.assertEqual(config.output_px, 160)

    def test_coverage_grid_has_no_gap_or_overlap(self):
        grid = CoverageGrid.from_bbox((125, 77, 3900, 2700))
        self.assertLessEqual(grid.output_box(0, 0)[0], 125)
        for ix in range(grid.nx - 1):
            self.assertEqual(grid.output_box(ix, 0)[2], grid.output_box(ix + 1, 0)[0])
        for iy in range(grid.ny - 1):
            self.assertEqual(grid.output_box(0, iy)[3], grid.output_box(0, iy + 1)[1])
        self.assertGreaterEqual(grid.output_box(grid.nx - 1, grid.ny - 1)[2], 3900)
        self.assertGreaterEqual(grid.output_box(grid.nx - 1, grid.ny - 1)[3], 2700)

    def test_half_open_boundary_has_one_owner(self):
        grid = CoverageGrid(0, 0, 3, 2)
        self.assertEqual(grid.owner(1279.999, 100), (0, 0))
        self.assertEqual(grid.owner(1280, 100), (1, 0))
        self.assertEqual(grid.owner(2560, 1280), (2, 1))

    def test_nm_pixel_round_trip(self):
        for value_nm in (-160, 0, 640, 1279.5, 1440):
            pixel = nm_to_pixel(value_nm, -160, 8)
            self.assertAlmostEqual(pixel_to_nm(pixel, -160, 8), value_nm)

    def test_exact_edge_pair_geometry_and_owner(self):
        region = self.spacing_region()
        grid = CoverageGrid.from_bbox((0, 0, 2560, 1280))
        records, polygons = extract_exact_violations(region, 1.0, "synthetic", grid)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["spacing_nm"], 100)
        self.assertEqual(record["deficit_nm"], 40)
        self.assertEqual(record["orientation"], "vertical")
        self.assertEqual(record["owner_index"], [0, 0])
        self.assertEqual(polygons[record["violation_id"]].area(), 60000)

    def test_image_and_mask_are_physically_registered(self):
        config = LocalizationConfig()
        metal = self.spacing_region()
        grid = CoverageGrid.from_bbox((0, 0, 2560, 1280))
        records, polygons = extract_exact_violations(metal, 1.0, "synthetic", grid)
        markers = db.Region(polygons[records[0]["violation_id"]])
        image, mask = rasterize_registered_tile(
            metal,
            markers,
            grid.input_box(0, 0, config.halo_nm),
            grid.output_box(0, 0),
            config,
            1.0,
        )
        self.assertEqual(image.shape, (200, 200))
        self.assertEqual(mask.shape, (160, 160))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertGreater(image.sum(), 0)
        self.assertGreater(mask.sum(), 0)

    def test_owned_subpixel_surrogate_is_explicit(self):
        config = LocalizationConfig()
        mask = np.zeros((config.output_px, config.output_px), dtype=np.uint8)
        violation_id = "synthetic:m1.2:000000"
        records = {
            violation_id: {
                "owner_index": [0, 0],
                "midpoint_nm": [1279, 640],
                "owned_subpixel_surrogate": False,
            }
        }
        polygons = {violation_id: db.Polygon(db.Box(1280, 600, 1281, 680))}
        surrogates, omitted = _apply_owned_subpixel_surrogate(
            mask,
            records,
            [violation_id],
            0,
            0,
            (0, 0, 1280, 1280),
            config,
            polygons,
            1.0,
        )
        self.assertEqual(surrogates, [violation_id])
        self.assertEqual(omitted, [])
        self.assertEqual(mask.sum(), 1)
        self.assertTrue(records[violation_id]["owned_subpixel_surrogate"])

    def test_synthetic_layout_build_writes_vector_and_raster_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            layout = db.Layout()
            layout.dbu = 0.001
            top = layout.create_cell("TOP")
            layer = layout.layer(68, 20)
            top.shapes(layer).insert(db.Box(1080, 100, 1230, 700))
            top.shapes(layer).insert(db.Box(1330, 100, 1480, 700))
            top.shapes(layer).insert(db.Box(1800, 100, 2200, 700))
            top.shapes(layer).insert(db.Box(200, 1500, 800, 1700))
            top.shapes(layer).insert(db.Box(1800, 1500, 2400, 1700))
            top.shapes(layer).insert(db.Box(1800, 1800, 2400, 2000))
            top.shapes(layer).insert(db.Box(3000, 100, 3400, 700))
            top.shapes(layer).insert(db.Box(3000, 1500, 3400, 1700))
            source = tmp_path / "synthetic.gds"
            layout.write(str(source))

            output = tmp_path / "dataset"
            summary = build_layout_dataset("synthetic", source, output)
            self.assertEqual(summary["exact_violation_count"], 2)
            self.assertEqual(summary["unique_owner_count"], 2)
            self.assertEqual(summary["dirty_tile_count"], 3)
            self.assertEqual(summary["clean_tile_count"], 3)
            self.assertEqual(
                set(summary["visual_audit"]),
                {"clean", "dense", "sparse", "horizontal", "vertical", "boundary", "near_threshold"},
            )
            vectors = [json.loads(line) for line in (output / "violations.jsonl").read_text().splitlines()]
            tiles = [json.loads(line) for line in (output / "tiles.jsonl").read_text().splitlines()]
            self.assertEqual(len(vectors), 2)
            self.assertEqual(len(tiles), 6)
            dirty = next(tile for tile in tiles if tile["label"] == "dirty")
            arrays = np.load(output / dirty["image_path"])
            self.assertEqual(arrays["image"].shape, (200, 200))
            self.assertEqual(arrays["mask"].shape, (160, 160))
            resumed = build_layout_dataset("synthetic", source, output)
            self.assertEqual(resumed, summary)


if __name__ == "__main__":
    unittest.main()
