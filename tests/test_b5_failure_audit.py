from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.run_b5_failure_audit import (
    B2_MODEL,
    B4_MODEL,
    density_bin,
    disagreement_gate,
    join_records,
    model_metrics,
    run,
    wilson_interval,
)


def prediction_artifact(model: str, protocol: str, split: str, seed: int, records):
    return {
        "phase": "B5",
        "manifest_id": "manifest-test",
        "protocol": protocol,
        "split": split,
        "seed": seed,
        "model": model,
        "samples": len(records),
        "records": records,
    }


class B5FailureAuditTest(unittest.TestCase):
    def test_wilson_interval_contains_observed_rate(self) -> None:
        lower, upper = wilson_interval(8, 10)
        self.assertLess(lower, 0.8)
        self.assertGreater(upper, 0.8)

    def test_density_bins_have_stable_edges(self) -> None:
        self.assertEqual(density_bin(0.03), "0.03-0.15")
        self.assertEqual(density_bin(0.15), "0.15-0.30")
        self.assertEqual(density_bin(0.85), "0.60-0.85")

    def test_join_rejects_path_misalignment(self) -> None:
        left = {"records": [{"path": "clean/a_tile_0_0.npy", "label": 0, "dirty_probability": 0.1}]}
        right = {"records": [{"path": "clean/b_tile_0_0.npy", "label": 0, "dirty_probability": 0.1}]}
        manifest = {
            "clean/a_tile_0_0.npy": {
                "metal_density": 0.2,
                "source_layout": "a",
                "layout_family": "a",
            }
        }
        with self.assertRaisesRegex(ValueError, "not path-aligned"):
            join_records(left, right, manifest, {}, protocol="p", split="validation", seed=42)

    def test_metrics_include_calibration(self) -> None:
        records = [
            {"label": 0, "b2_probability": 0.1, "b2_prediction": 0},
            {"label": 0, "b2_probability": 0.6, "b2_prediction": 1},
            {"label": 1, "b2_probability": 0.8, "b2_prediction": 1},
            {"label": 1, "b2_probability": 0.4, "b2_prediction": 0},
        ]
        metrics = model_metrics(records, B2_MODEL)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertGreater(metrics["brier_score"], 0.0)
        self.assertGreaterEqual(metrics["expected_calibration_error_10bin"], 0.0)

    def test_disagreement_gate_requires_repeated_class_complementarity(self) -> None:
        runs = []
        for seed in (42, 43):
            runs.append(
                {
                    "seed": seed,
                    "disagreement": {
                        "by_family": {
                            family: {
                                "dirty": {"b2_only_correct": 5, "b4_only_correct": 1},
                                "clean": {"b2_only_correct": 1, "b4_only_correct": 5},
                            }
                            for family in ("a", "b")
                        }
                    },
                }
            )
        self.assertTrue(disagreement_gate(runs)["passed"])

    def test_end_to_end_writes_report_without_geometry_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [
                "clean/a_tile_0_0.npy",
                "dirty/a_tile_1_0.npy",
                "clean/b_tile_0_0.npy",
                "dirty/b_tile_1_0.npy",
            ]
            manifest = {
                "manifest_id": "manifest-test",
                "records": [
                    {
                        "path": path,
                        "source_layout": path.split("/")[1].split("_tile_")[0],
                        "layout_family": path.split("/")[1].split("_tile_")[0],
                        "metal_density": 0.2 + index * 0.1,
                    }
                    for index, path in enumerate(paths)
                ],
                "protocols": {
                    protocol: {"splits": {split: paths for split in ("train", "validation")}}
                    for protocol in ("unseen_layout_v1", "tile_random_reference")
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            predictions = root / "predictions"
            for protocol in manifest["protocols"]:
                for split in ("train", "validation"):
                    for seed in (42, 43, 44):
                        for model, probabilities in (
                            (B2_MODEL, (0.1, 0.8, 0.2, 0.9)),
                            (B4_MODEL, (0.2, 0.7, 0.3, 0.8)),
                        ):
                            records = [
                                {"path": path, "label": int(path.startswith("dirty/")), "dirty_probability": probability}
                                for path, probability in zip(paths, probabilities)
                            ]
                            artifact = prediction_artifact(model, protocol, split, seed, records)
                            output = predictions / split / f"{protocol}__{model}__seed_{seed}.json"
                            output.parent.mkdir(parents=True, exist_ok=True)
                            output.write_text(json.dumps(artifact), encoding="utf-8")
            output_dir = root / "audit"
            args = Namespace(
                manifest=manifest_path,
                predictions=predictions,
                output_dir=output_dir,
                protocols=["unseen_layout_v1", "tile_random_reference"],
                splits=["train", "validation"],
                seeds=[42, 43, 44],
                geometry_annotations=None,
                dataset=None,
                export_missing=False,
                force=False,
                verify_checkpoint_hashes=False,
                b2_checkpoints=None,
                b4_checkpoints=None,
                b2_results=Path("unused"),
                b4_results=Path("unused"),
                exporter=Path("unused"),
                python=Path("python"),
                batch_size=32,
                device="cpu",
                cpu=True,
            )
            summary = run(args)
            self.assertEqual(summary["status"], "complete_measured_features_geometry_unavailable")
            self.assertEqual(
                summary["feature_availability"]["edge_orientation"]["status"],
                "unavailable",
            )
            self.assertEqual(len(summary["train_minus_validation_gaps"]), 6)
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "records.jsonl").is_file())
            self.assertIn("Feature availability", (output_dir / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
