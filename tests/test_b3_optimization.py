from __future__ import annotations

import unittest

from scripts.run_b3_optimization import (
    candidate_id,
    paired_seed_wins,
    select_candidate,
    selected_validation_metrics,
    summarize_validation_candidate,
    test_regressions as find_test_regressions,
)


def make_run(seed: int, f1: float, recall: float) -> dict[str, object]:
    return {
        "seed": seed,
        "best_epoch": 7,
        "best_validation_metrics": {
            "loss": 0.2,
            "accuracy": 0.9,
            "precision": 0.91,
            "recall": recall,
            "f1": f1,
            "predicted_class_counts": {"clean": 50, "dirty": 50},
            "confusion_matrix": [[45, 5], [5, 45]],
        },
    }


class B3OptimizationTest(unittest.TestCase):
    def test_reads_b2_validation_metrics_from_selected_history_epoch(self) -> None:
        run = {
            "best_epoch": 2,
            "history": [
                {
                    "epoch": 1,
                    "validation_loss": 0.4,
                    "validation_accuracy": 0.7,
                    "validation_precision": 0.7,
                    "validation_recall": 0.7,
                    "validation_f1": 0.7,
                    "validation_predicted_class_counts": {"clean": 3, "dirty": 7},
                    "validation_confusion_matrix": [[3, 2], [1, 4]],
                },
                {
                    "epoch": 2,
                    "validation_loss": 0.2,
                    "validation_accuracy": 0.9,
                    "validation_precision": 0.88,
                    "validation_recall": 0.92,
                    "validation_f1": 0.90,
                    "validation_predicted_class_counts": {"clean": 5, "dirty": 5},
                    "validation_confusion_matrix": [[4, 1], [0, 5]],
                },
            ],
        }
        self.assertAlmostEqual(selected_validation_metrics(run)["f1"], 0.90)

    def test_selection_rejects_higher_f1_when_recall_regresses(self) -> None:
        baseline = summarize_validation_candidate(
            "rmsprop",
            0.001,
            [make_run(42, 0.90, 0.90), make_run(43, 0.91, 0.91), make_run(44, 0.89, 0.89)],
        )
        low_recall = summarize_validation_candidate(
            "adam",
            0.001,
            [make_run(42, 0.93, 0.85), make_run(43, 0.94, 0.86), make_run(44, 0.92, 0.87)],
        )
        selected = select_candidate([baseline, low_recall], baseline)
        self.assertEqual(selected["candidate_id"], baseline["candidate_id"])

    def test_paired_seed_wins_requires_seed_aligned_comparison(self) -> None:
        baseline = summarize_validation_candidate(
            "rmsprop",
            0.001,
            [make_run(42, 0.90, 0.90), make_run(43, 0.90, 0.90), make_run(44, 0.90, 0.90)],
        )
        candidate = summarize_validation_candidate(
            "adam",
            0.001,
            [make_run(44, 0.91, 0.91), make_run(42, 0.89, 0.91), make_run(43, 0.92, 0.91)],
        )
        self.assertEqual(paired_seed_wins(candidate, baseline), 2)

    def test_confirmation_flags_accuracy_recall_and_f1_regression(self) -> None:
        baseline = {
            "metrics": {
                "test_accuracy": {"mean": 0.90},
                "test_recall": {"mean": 0.91},
                "test_f1": {"mean": 0.905},
            }
        }
        candidate = {
            "metrics": {
                "test_accuracy": {"mean": 0.90},
                "test_recall": {"mean": 0.90},
                "test_f1": {"mean": 0.91},
            }
        }
        issues = find_test_regressions(candidate, baseline)
        self.assertEqual(len(issues), 1)
        self.assertIn("test_recall", issues[0])

    def test_candidate_identifier_is_filesystem_safe(self) -> None:
        self.assertEqual(candidate_id("adam", 0.0003), "adam_lr_0p0003")

    def test_collapsed_candidate_is_not_selected(self) -> None:
        baseline = summarize_validation_candidate(
            "rmsprop",
            0.001,
            [make_run(42, 0.90, 0.90), make_run(43, 0.90, 0.90), make_run(44, 0.90, 0.90)],
        )
        collapsed_runs = [make_run(42, 0.95, 1.0), make_run(43, 0.95, 1.0), make_run(44, 0.95, 1.0)]
        for run in collapsed_runs:
            run["best_validation_metrics"]["predicted_class_counts"] = {"clean": 0, "dirty": 100}
        collapsed = summarize_validation_candidate("adam", 0.001, collapsed_runs)
        self.assertEqual(collapsed["collapsed_seeds"], [42, 43, 44])
        self.assertEqual(select_candidate([baseline, collapsed], baseline)["candidate_id"], baseline["candidate_id"])


if __name__ == "__main__":
    unittest.main()
