from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.run_b5_ensemble import (
    B2_MODEL,
    B2_WEIGHTS,
    B4_MODEL,
    SEARCH_PROTOCOL,
    aligned_records,
    build_prediction_command,
    classification_metrics,
    frozen_unseen_gate,
    reference_gate,
    score_ensemble_runs,
    score_model_runs,
    select_candidate,
    validation_gate,
)


def artifact(seed: int, probabilities: list[float]) -> dict[str, object]:
    labels = [0, 0, 1, 1]
    return {
        "seed": seed,
        "records": [
            {
                "path": f"{'dirty' if label else 'clean'}/layout_tile_{index}_0.npy",
                "label": label,
                "dirty_probability": probability,
            }
            for index, (label, probability) in enumerate(zip(labels, probabilities))
        ],
    }


def paired_artifacts() -> list[tuple[int, dict[str, object], dict[str, object]]]:
    return [
        (
            seed,
            artifact(seed, [0.1, 0.51, 0.9, 0.8]),
            artifact(seed, [0.2, 0.1, 0.49, 0.9]),
        )
        for seed in (42, 43, 44)
    ]


def summary(seed_values: list[tuple[int, float, float, float]]) -> dict[str, object]:
    runs = []
    for seed, accuracy, recall, f1 in seed_values:
        runs.append(
            {
                "seed": seed,
                "accuracy": accuracy,
                "precision": f1,
                "recall": recall,
                "f1": f1,
                "predicted_class_counts": {"clean": 50, "dirty": 50},
                "confusion_matrix": [[45, 5], [5, 45]],
                "per_layout": {
                    "layout": classification_metrics(
                        [0, 0, 1, 1], [0, 0, 1, 1]
                    )
                },
            }
        )
    from scripts.run_b5_ensemble import summarize_runs

    return summarize_runs(runs)


class B5EnsembleTest(unittest.TestCase):
    def test_alignment_rejects_path_mismatch(self) -> None:
        left = artifact(42, [0.1, 0.2, 0.8, 0.9])
        right = artifact(42, [0.1, 0.2, 0.8, 0.9])
        right["records"][0]["path"] = "clean/other_tile_0_0.npy"
        with self.assertRaisesRegex(ValueError, "not sample-aligned"):
            aligned_records(left, right)

    def test_half_blend_recovers_complementary_errors(self) -> None:
        pairs = paired_artifacts()
        b2 = score_model_runs(pairs, B2_MODEL)
        b4 = score_model_runs(pairs, B4_MODEL)
        blend = score_ensemble_runs(pairs, 0.5)
        self.assertGreater(blend["metrics"]["accuracy"]["mean"], b2["metrics"]["accuracy"]["mean"])
        self.assertGreater(blend["metrics"]["accuracy"]["mean"], b4["metrics"]["accuracy"]["mean"])

    def test_selection_uses_f1_then_accuracy(self) -> None:
        candidates = []
        for weight, f1, accuracy in zip(B2_WEIGHTS, (0.91, 0.92, 0.92), (0.95, 0.93, 0.94)):
            candidates.append(
                {
                    "b2_weight": weight,
                    "metrics": {"f1": {"mean": f1}, "accuracy": {"mean": accuracy}},
                }
            )
        self.assertEqual(select_candidate(candidates)["b2_weight"], 0.75)

    def test_validation_gate_requires_beating_stronger_model(self) -> None:
        b2 = summary([(42, .90, .90, .90), (43, .90, .90, .90), (44, .90, .90, .90)])
        b4 = summary([(42, .92, .92, .92), (43, .92, .92, .92), (44, .92, .92, .92)])
        candidate = summary([(42, .93, .93, .93), (43, .93, .93, .93), (44, .91, .93, .91)])
        self.assertTrue(validation_gate(candidate, b2, b4)["passed"])
        candidate["metrics"]["f1"]["mean"] = .915
        self.assertFalse(validation_gate(candidate, b2, b4)["passed"])

    def test_frozen_gate_requires_accuracy_and_f1_improvement(self) -> None:
        b2 = summary([(42, .90, .90, .90), (43, .90, .90, .90), (44, .90, .90, .90)])
        candidate = summary([(42, .91, .91, .91), (43, .91, .91, .91), (44, .89, .89, .89)])
        self.assertTrue(frozen_unseen_gate(candidate, b2)["passed"])
        candidate["metrics"]["accuracy"]["mean"] = .89
        self.assertFalse(frozen_unseen_gate(candidate, b2)["passed"])

    def test_reference_gate_applies_half_point_tolerance(self) -> None:
        b2 = summary([(42, .92, .92, .92), (43, .92, .92, .92), (44, .92, .92, .92)])
        within = summary([(42, .916, .916, .916), (43, .916, .916, .916), (44, .916, .916, .916)])
        outside = summary([(42, .914, .914, .914), (43, .914, .914, .914), (44, .914, .914, .914)])
        self.assertTrue(reference_gate(within, b2)["passed"])
        self.assertFalse(reference_gate(outside, b2)["passed"])

    def test_validation_command_never_exports_test_split(self) -> None:
        args = argparse.Namespace(
            python=Path("python"),
            exporter=Path("training/export_classifier_predictions.py"),
            dataset=Path("dataset.zip"),
            manifest=Path("manifest.json"),
            b2_checkpoints=Path("b2"),
            b4_checkpoints=Path("b4"),
            output_dir=Path("results"),
            batch_size=32,
            device="auto",
            cpu=False,
        )
        command = build_prediction_command(args, B2_MODEL, SEARCH_PROTOCOL, "validation", 42)
        self.assertIn("validation", command)
        self.assertNotIn("test", command)


if __name__ == "__main__":
    unittest.main()
