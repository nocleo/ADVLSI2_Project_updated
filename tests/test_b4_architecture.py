from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.run_b4_architecture import (
    BASELINE_MODEL,
    COMPACT_MODEL,
    build_training_command,
    cost_gate,
    frozen_test_gate,
    reference_test_gate,
    summarize_validation,
    validation_gate,
)


def validation_run(
    seed: int,
    accuracy: float,
    recall: float,
    f1: float,
    counts: dict[str, int] | None = None,
) -> dict[str, object]:
    return {
        "seed": seed,
        "best_epoch": 12,
        "best_validation_metrics": {
            "accuracy": accuracy,
            "precision": f1,
            "recall": recall,
            "f1": f1,
            "predicted_class_counts": counts or {"clean": 50, "dirty": 50},
        },
    }


def with_runs(summary: dict[str, object], runs: list[dict[str, object]]) -> dict[str, object]:
    summary["_runs"] = runs
    return summary


def test_summary(values: list[tuple[int, float, float, float]]) -> dict[str, object]:
    def stats(index: int) -> dict[str, float | int]:
        series = [value[index] for value in values]
        return {
            "runs": len(series),
            "mean": sum(series) / len(series),
            "sample_stddev": 0.0,
            "min": min(series),
            "max": max(series),
        }

    return {
        "metrics": {
            "test_accuracy": stats(1),
            "test_recall": stats(2),
            "test_f1": stats(3),
            "test_precision": stats(3),
        },
        "per_seed": [
            {"seed": seed, "test_accuracy": accuracy, "test_recall": recall, "test_f1": f1, "test_precision": f1}
            for seed, accuracy, recall, f1 in values
        ],
    }


class B4ArchitectureTest(unittest.TestCase):
    def test_device_selection_prefers_mps_after_cuda(self) -> None:
        try:
            from unittest.mock import patch

            import torch
            from training.runtime_device import select_device
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed in the lightweight test runtime")
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=True),
        ):
            self.assertEqual(str(select_device()), "mps")

    def test_explicit_unavailable_mps_fails(self) -> None:
        try:
            from unittest.mock import patch

            import torch
            from training.runtime_device import select_device
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed in the lightweight test runtime")
        with patch.object(torch.backends.mps, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
                select_device("mps")

    def test_compact_model_shape_and_parameter_reduction(self) -> None:
        try:
            import torch
            from training.classifier_models import build_classifier
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed in the lightweight test runtime")
        baseline = build_classifier(BASELINE_MODEL)
        candidate = build_classifier(COMPACT_MODEL)
        self.assertEqual(tuple(candidate(torch.zeros(2, 1, 200, 200)).shape), (2, 2))
        self.assertLess(
            sum(parameter.numel() for parameter in candidate.parameters()),
            sum(parameter.numel() for parameter in baseline.parameters()),
        )

    def test_validation_gate_requires_two_paired_f1_wins(self) -> None:
        baseline_runs = [
            validation_run(42, 0.90, 0.90, 0.90),
            validation_run(43, 0.90, 0.90, 0.90),
            validation_run(44, 0.90, 0.90, 0.90),
        ]
        candidate_runs = [
            validation_run(42, 0.91, 0.91, 0.91),
            validation_run(43, 0.91, 0.91, 0.91),
            validation_run(44, 0.89, 0.91, 0.89),
        ]
        baseline = with_runs(summarize_validation(baseline_runs), baseline_runs)
        candidate = with_runs(summarize_validation(candidate_runs), candidate_runs)
        self.assertTrue(validation_gate(candidate, baseline)["passed"])

    def test_validation_gate_rejects_collapsed_seed(self) -> None:
        baseline_runs = [validation_run(seed, 0.90, 0.90, 0.90) for seed in (42, 43, 44)]
        candidate_runs = [validation_run(seed, 0.92, 0.92, 0.92) for seed in (42, 43, 44)]
        candidate_runs[2]["best_validation_metrics"]["predicted_class_counts"] = {"clean": 0, "dirty": 100}
        baseline = with_runs(summarize_validation(baseline_runs), baseline_runs)
        candidate = with_runs(summarize_validation(candidate_runs), candidate_runs)
        self.assertFalse(validation_gate(candidate, baseline)["passed"])

    def test_frozen_gate_requires_consistent_f1_and_recall_improvement(self) -> None:
        baseline = test_summary([(42, .90, .90, .90), (43, .90, .90, .90), (44, .90, .90, .90)])
        candidate = test_summary([(42, .91, .91, .91), (43, .91, .91, .91), (44, .895, .89, .89)])
        self.assertTrue(frozen_test_gate(candidate, baseline)["passed"])

    def test_reference_gate_rejects_half_point_f1_regression(self) -> None:
        baseline = test_summary([(42, .92, .92, .92), (43, .92, .92, .92), (44, .92, .92, .92)])
        candidate = test_summary([(42, .92, .92, .91), (43, .92, .92, .91), (44, .92, .92, .91)])
        self.assertFalse(reference_test_gate(candidate, baseline)["passed"])

    def test_cost_gate_caps_both_cpu_backends(self) -> None:
        benchmark = {"models": {
            BASELINE_MODEL: {
                "parameters": 100, "state_dict_bytes": 1000,
                "pytorch_cpu_batch1": {"median_ms": 10.0},
                "onnx_cpu_batch1": {"median_ms": 8.0},
            },
            COMPACT_MODEL: {
                "parameters": 50, "state_dict_bytes": 500,
                "pytorch_cpu_batch1": {"median_ms": 14.0},
                "onnx_cpu_batch1": {"median_ms": 13.0},
            },
        }}
        self.assertFalse(cost_gate(benchmark)["passed"])

    def test_search_command_locks_model_and_test(self) -> None:
        args = argparse.Namespace(
            python=Path("python"), trainer=Path("trainer.py"), dataset=Path("data.zip"),
            manifest=Path("manifest.json"), epochs=30, batch_size=32,
            learning_rate=0.001, output_dir=Path("out"), cpu=False,
        )
        command = build_training_command(args, "unseen_layout_v1", 42)
        self.assertIn(COMPACT_MODEL, command)
        self.assertIn("--skip-test", command)
        self.assertEqual(command[command.index("--optimizer") + 1], "rmsprop")


if __name__ == "__main__":
    unittest.main()
