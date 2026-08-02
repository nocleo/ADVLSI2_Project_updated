from __future__ import annotations

import unittest

from scripts.run_b3_extension import calibration_summary, quality_preserved, summarize_runs
from training.threshold_calibration import select_threshold, threshold_candidates


def run(seed: int, f1: float, recall: float, accuracy: float = 0.9, epochs: int = 30):
    return {
        "seed": seed,
        "epochs": 30,
        "epochs_completed": epochs,
        "best_epoch": min(epochs, 20),
        "best_validation_metrics": {
            "accuracy": accuracy,
            "precision": f1,
            "recall": recall,
            "f1": f1,
            "loss": 0.2,
            "predicted_class_counts": {"clean": 5, "dirty": 5},
            "confusion_matrix": [[4, 1], [1, 4]],
        },
    }


class B3ExtensionTest(unittest.TestCase):
    def test_threshold_candidates_include_default_and_boundaries(self) -> None:
        candidates = threshold_candidates([0.2, 0.8])
        self.assertIn(0.5, candidates)
        self.assertIn(0.2, candidates)
        self.assertIn(0.8, candidates)

    def test_threshold_selection_maximizes_dirty_f1(self) -> None:
        threshold, metrics = select_threshold([0, 0, 1, 1], [0.1, 0.4, 0.45, 0.9], 0.0)
        self.assertGreater(threshold, 0.4)
        self.assertLessEqual(threshold, 0.45)
        self.assertEqual(metrics["f1"], 1.0)

    def test_threshold_selection_respects_recall_floor(self) -> None:
        _, metrics = select_threshold([0, 1, 1], [0.4, 0.45, 0.9], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_run_summary_records_epoch_efficiency(self) -> None:
        summary = summarize_runs([run(42, 0.9, 0.9, epochs=20), run(43, 0.9, 0.9, epochs=22)])
        self.assertEqual(summary["epochs_completed"]["mean"], 21.0)

    def test_quality_tolerance_accepts_half_point_regression(self) -> None:
        baseline = summarize_runs([run(42, 0.90, 0.90, 0.90), run(43, 0.90, 0.90, 0.90)])
        candidate = summarize_runs([run(42, 0.895, 0.895, 0.895), run(43, 0.895, 0.895, 0.895)])
        self.assertTrue(quality_preserved(candidate, baseline))

    def test_calibration_summary_keeps_default_and_selected_separate(self) -> None:
        items = [
            {"selected_threshold": 0.4, "default_metrics": {"accuracy": 0.8, "recall": 0.7, "f1": 0.75}, "selected_metrics": {"accuracy": 0.82, "recall": 0.8, "f1": 0.81}},
            {"selected_threshold": 0.6, "default_metrics": {"accuracy": 0.9, "recall": 0.8, "f1": 0.85}, "selected_metrics": {"accuracy": 0.91, "recall": 0.85, "f1": 0.88}},
        ]
        summary = calibration_summary(items)
        self.assertEqual(summary["thresholds"], [0.4, 0.6])
        self.assertGreater(summary["selected"]["f1"]["mean"], summary["default"]["f1"]["mean"])


if __name__ == "__main__":
    unittest.main()
