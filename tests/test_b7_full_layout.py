from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.full_layout_stitching import (
    DeploymentPolicy,
    add_layout_coordinates,
    proxy_component_vector_matches,
    stitch_components,
    tile_components,
)

try:
    import klayout.db as db
    import torch

    from generate_training_dataset_scripts.generate_localization_dataset import (
        CoverageGrid,
        LocalizationConfig,
    )
    from training.full_layout_evaluation import (
        canonical_pair_key,
        evaluate_exact_layout,
        load_layout_variant,
        scan_full_layout,
        select_validation_policy,
    )
    from training.multitask_unet import MultiTaskUNet
except ModuleNotFoundError:
    db = None
    torch = None


class B7SparseStitchingTest(unittest.TestCase):
    def test_raster_row_is_inverted_into_upward_layout_y(self) -> None:
        probability = np.zeros((160, 160), dtype=np.float32)
        probability[0, 0] = 0.9
        probability[159, 159] = 0.9
        components = tile_components(probability, 0.5, [2, 3], "tile", 0.8)
        pixels = sorted(tuple(item["_pixels_yx"][0]) for item in components)
        self.assertEqual(pixels, [(480, 479), (639, 320)])

    def test_components_join_across_nonoverlapping_tile_boundary(self) -> None:
        left = np.zeros((160, 160), dtype=np.float32)
        right = np.zeros((160, 160), dtype=np.float32)
        left[80, 159] = 0.9
        right[80, 0] = 0.9
        raw = tile_components(left, 0.5, [0, 0], "left", 0.9)
        raw += tile_components(right, 0.5, [1, 0], "right", 0.9)
        stitched = stitch_components(raw, DeploymentPolicy(0.5, 0.5, 1, 0))
        self.assertEqual(len(raw), 2)
        self.assertEqual(len(stitched), 1)
        self.assertEqual(stitched[0]["source_tile_ids"], ["left", "right"])
        self.assertEqual(stitched[0]["area_pixels"], 2)

    def test_declared_gap_area_and_classification_gates_are_applied(self) -> None:
        probability = np.zeros((160, 160), dtype=np.float32)
        probability[80, 50] = 0.9
        probability[80, 53] = 0.8
        raw = tile_components(probability, 0.5, [0, 0], "tile", 0.4)
        self.assertEqual(
            stitch_components(raw, DeploymentPolicy(0.5, 0.5, 1, 2)), []
        )
        raw = tile_components(probability, 0.5, [0, 0], "tile", 0.9)
        self.assertEqual(
            len(stitch_components(raw, DeploymentPolicy(0.5, 0.5, 2, 0))), 0
        )
        merged = stitch_components(raw, DeploymentPolicy(0.5, 0.5, 2, 2))
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["area_pixels"], 2)

    def test_global_pixels_round_trip_to_physical_coordinates(self) -> None:
        probability = np.zeros((160, 160), dtype=np.float32)
        probability[159, 0] = 0.9
        raw = tile_components(probability, 0.5, [1, 2], "tile", 0.9)
        stitched = stitch_components(raw, DeploymentPolicy(0.5, 0.5, 1, 0))
        physical = add_layout_coordinates(stitched, [-1280, 2560], 8)[0]
        self.assertEqual(physical["bbox_nm"], [0.0, 5120.0, 8.0, 5128.0])
        self.assertEqual(physical["centroid_nm"], [4.0, 5124.0])

    def test_proxy_counts_unique_vectors_and_false_components(self) -> None:
        components = [
            {
                "component_id": "a",
                "bbox_nm": [0, 0, 100, 100],
            },
            {
                "component_id": "b",
                "bbox_nm": [1000, 1000, 1100, 1100],
            },
        ]
        vectors = [
            {"violation_id": "v1", "midpoint_nm": [40, 40]},
            {"violation_id": "v2", "midpoint_nm": [90, 90]},
        ]
        metrics = proxy_component_vector_matches(components, vectors, tolerance_nm=0)
        self.assertEqual(metrics["detected_violation_count"], 2)
        self.assertEqual(metrics["true_component_count"], 1)
        self.assertEqual(metrics["false_component_count"], 1)
        self.assertEqual(metrics["component_precision"], 0.5)
        self.assertEqual(metrics["violation_recall"], 1.0)


@unittest.skipIf(torch is None or db is None, "PyTorch and KLayout are optional")
class B7DependencyBackedTest(unittest.TestCase):
    def test_canonical_pair_is_endpoint_and_edge_order_invariant(self) -> None:
        first = canonical_pair_key([0, 0, 0, 10], [20, 0, 20, 10])
        second = canonical_pair_key([20, 10, 20, 0], [0, 10, 0, 0])
        self.assertEqual(first, second)

    @staticmethod
    def _write_layout(path: Path, with_violation: bool) -> None:
        layout = db.Layout()
        layout.dbu = 0.001
        cell = layout.create_cell("TOP")
        layer = layout.layer(68, 20)
        cell.shapes(layer).insert(db.Box(0, 0, 400, 400))
        if with_violation:
            cell.shapes(layer).insert(db.Box(500, 0, 900, 400))
        layout.write(str(path))

    def test_synthetic_full_layout_scan_stitch_and_exact_recovery(self) -> None:
        class DeterministicModel(torch.nn.Module):
            def forward(self, inputs):
                count = inputs.shape[0]
                segmentation = torch.full(
                    (count, 1, 160, 160), -20.0, device=inputs.device
                )
                segmentation[:, :, 70:90, 55:105] = 20.0
                classification = torch.tensor(
                    [[0.0, 10.0]], device=inputs.device
                ).repeat(count, 1)
                return segmentation, classification

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "layout.gds"
            self._write_layout(path, with_violation=True)
            variant = load_layout_variant(
                "synthetic",
                path,
                None,
                LocalizationConfig(),
                "injected",
            )
            scan = scan_full_layout(
                variant,
                [DeterministicModel()],
                torch.device("cpu"),
                thresholds=[0.5],
                batch_size=2,
            )
            result = evaluate_exact_layout(
                scan, DeploymentPolicy(0.5, 0.5, 1, 0, recovery_radius_nm=400)
            )
            self.assertTrue(result["complete_scan"])
            self.assertEqual(result["ground_truth_violation_count"], 1)
            self.assertGreaterEqual(result["exact_candidate_count"], 1)
            self.assertEqual(result["detected_unique_violation_count"], 1)

    def test_policy_selection_uses_only_supplied_validation_scans(self) -> None:
        probability = np.zeros((160, 160), dtype=np.float32)
        probability[80, 80] = 0.8
        raw = tile_components(probability, 0.5, [0, 0], "v", 0.9)
        scan = {
            "layout": "validation",
            "variant": "injected",
            "grid": {"x0_nm": 0, "y0_nm": 0},
            "vectors": [
                {"violation_id": "v1", "midpoint_nm": [644, 636]}
            ],
            "components_by_threshold": {f"{value:.3f}": raw for value in (0.5,)},
        }
        selection = select_validation_policy(
            [scan],
            segmentation_thresholds=[0.5],
            classification_thresholds=[0.5],
            minimum_areas=[1],
            merge_gaps=[0],
        )
        self.assertEqual(selection["selection_split"], "validation_layout_families_only")
        self.assertEqual(selection["selected_policy"]["segmentation_threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
