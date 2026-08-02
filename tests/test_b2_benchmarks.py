from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.run_b2_benchmarks import (
    metric_stats,
    pooled_confusion_matrix,
    render_markdown,
    run_acceptance_issues,
    summarize_runs,
    validate_run,
)


def make_run(protocol: str, seed: int, accuracy: float) -> dict[str, object]:
    return {
        "protocol": protocol,
        "seed": seed,
        "best_epoch": 12,
        "test_loss": 1.0 - accuracy,
        "test_accuracy": accuracy,
        "test_precision": accuracy - 0.01,
        "test_recall": accuracy - 0.02,
        "test_f1": accuracy - 0.015,
        "test_confusion_matrix": [[30, 5], [7, 28]],
        "test_per_layout": {
            "layout_a": {
                "samples": 70,
                "loss": 1.0 - accuracy,
                "accuracy": accuracy,
                "precision": accuracy - 0.01,
                "recall": accuracy - 0.02,
                "f1": accuracy - 0.015,
            }
        },
        "weights": f"results/{protocol}_{seed}.pth",
        "weights_sha256": f"sha-{protocol}-{seed}",
        "device": "cpu",
        "runtime": {
            "python": "3.12.0",
            "pytorch": "2.2.0",
            "platform": "test-platform",
        },
        "_metrics_path": f"results/{protocol}_{seed}.json",
        "_weights_path": f"results/{protocol}_{seed}.pth",
    }


class B2BenchmarkTest(unittest.TestCase):
    def test_metric_stats_uses_sample_standard_deviation(self) -> None:
        stats = metric_stats([0.7, 0.8, 0.9])
        self.assertAlmostEqual(stats["mean"], 0.8)
        self.assertAlmostEqual(stats["sample_stddev"], 0.1)

    def test_summary_keeps_protocols_and_per_layout_results_separate(self) -> None:
        args = argparse.Namespace(
            protocols=["tile_random_reference", "unseen_layout_v1"],
            seeds=[42, 43, 44],
            epochs=30,
            batch_size=32,
            learning_rate=0.001,
            no_augmentation=False,
            trainer=Path("training/train_classifier.py"),
        )
        runs = [
            make_run(protocol, seed, 0.75 + 0.01 * index)
            for protocol in args.protocols
            for index, seed in enumerate(args.seeds)
        ]
        summary = summarize_runs(
            runs,
            {"manifest_id": "manifest-test"},
            "dataset-test",
            args,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(
            summary["protocol_results"]["unseen_layout_v1"]["seeds"],
            [42, 43, 44],
        )
        self.assertAlmostEqual(
            summary["protocol_results"]["tile_random_reference"]["metrics"][
                "test_accuracy"
            ]["mean"],
            0.76,
        )
        self.assertIn(
            "layout_a",
            summary["protocol_results"]["unseen_layout_v1"]["per_layout"],
        )
        self.assertEqual(
            summary["protocol_results"]["unseen_layout_v1"]["per_layout"][
                "layout_a"
            ]["samples"],
            70,
        )
        self.assertEqual(
            summary["protocol_results"]["unseen_layout_v1"][
                "pooled_confusion_matrix"
            ],
            [[90, 15], [21, 84]],
        )
        markdown = render_markdown(summary)
        self.assertIn("B2 Dual Classification Baselines", markdown)
        self.assertIn("76.00% ± 1.00%", markdown)
        self.assertIn("| `layout_a` | 70 |", markdown)

    def test_pooled_confusion_matrix_rejects_invalid_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid binary confusion matrix"):
            pooled_confusion_matrix([{"test_confusion_matrix": [[1, 2, 3], [4, 5, 6]]}])

    def test_collapsed_selected_checkpoint_is_reported_as_acceptance_issue(self) -> None:
        metrics = make_run("tile_random_reference", 42, 0.8)
        metrics.update(
            {
                "dataset_archive_sha256": "dataset-test",
                "manifest_id": "manifest-test",
                "epochs": 30,
                "batch_size": 32,
                "learning_rate": 0.001,
                "augmentation": "Manhattan rotations and reflections (training only)",
                "model": "NCSU_DRCNN",
                "model_source_sha256": "model-test",
                "trainer_source_sha256": "trainer-test",
                "optimizer": "RMSprop",
                "best_validation_loss": 0.5,
                "test_predicted_class_counts": {"clean": 30, "dirty": 40},
                "history": [
                    {
                        "epoch": 12,
                        "validation_predicted_class_counts": {"clean": 70, "dirty": 0},
                    }
                ],
            }
        )
        expected = {
            key: metrics[key]
            for key in (
                "dataset_archive_sha256",
                "manifest_id",
                "protocol",
                "seed",
                "epochs",
                "batch_size",
                "learning_rate",
                "augmentation",
                "model",
                "model_source_sha256",
                "trainer_source_sha256",
                "optimizer",
            )
        }
        validate_run(metrics, expected)
        self.assertIn(
            "collapsed on validation",
            " ".join(run_acceptance_issues(metrics)),
        )


if __name__ == "__main__":
    unittest.main()
