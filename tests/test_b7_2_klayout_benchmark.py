from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import klayout.db as db

    from training.klayout_performance import (
        benchmark_layout,
        fresh_batch_run,
        percentile,
        report_pair_keys,
        summarize_samples,
    )
except ModuleNotFoundError:
    db = None


@unittest.skipIf(db is None, "KLayout is an optional dependency")
class B72KLayoutBenchmarkTest(unittest.TestCase):
    @staticmethod
    def _write_layout(path: Path) -> None:
        layout = db.Layout()
        layout.dbu = 0.001
        top = layout.create_cell("TOP")
        m1 = layout.layer(68, 20)
        top.shapes(m1).insert(db.Box(0, 0, 400, 400))
        top.shapes(m1).insert(db.Box(500, 0, 900, 400))
        layout.write(str(path))

    def test_percentile_and_summary_are_deterministic(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4, 5], 0.95), 4.8)
        summary = summarize_samples([5, 1, 3])
        self.assertEqual(summary["median"], 3.0)
        self.assertEqual(summary["samples"], 3)

    def test_batch_report_and_repeated_benchmark_match_exact_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            layout_path = root / "synthetic.gds"
            expected_report = root / "expected.rdb"
            output_report = root / "measured.rdb"
            self._write_layout(layout_path)

            first = fresh_batch_run(layout_path, expected_report)
            self.assertEqual(first["violation_count"], 1)
            self.assertEqual(
                len(report_pair_keys(expected_report, 0.001)),
                first["violation_count"],
            )

            result = benchmark_layout(
                layout_path,
                output_report,
                expected_report=expected_report,
                repeats=2,
                warmups=0,
                incremental_samples=1,
            )
            self.assertTrue(result["correctness"]["exact_pair_set_match"])
            self.assertEqual(result["violation_count"], 1)
            self.assertEqual(
                result["batch_fresh_layout"]["total_seconds"]["samples"], 2
            )
            self.assertEqual(
                result["interactive_incremental"]["target_pairs_recovered"], 1
            )


if __name__ == "__main__":
    unittest.main()
