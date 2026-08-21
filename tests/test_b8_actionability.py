from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.b8_actionability import (
    ACTION_BY_ID,
    Action,
    build_flow_command,
    completion_is_valid,
    initialize_manifest,
    matrix_specs,
    render_fastroute_tcl,
    parse_klayout_report,
    smoke_specs,
    summarize_manifest,
)


class B8ActionabilityTest(unittest.TestCase):
    def test_registered_actions_and_schedules_are_complete(self) -> None:
        self.assertEqual(len(ACTION_BY_ID), 9)
        self.assertEqual(len(smoke_specs()), 9)
        matrix = matrix_specs()
        self.assertEqual(len(matrix), 126)
        self.assertEqual(len({spec.run_id for spec in matrix}), 126)

    def test_fastroute_uses_supported_adjustment_and_seed_commands(self) -> None:
        text = render_fastroute_tcl(Action("A11", 0.05, 0.35), 42)
        self.assertIn("set_global_routing_layer_adjustment", text)
        self.assertIn(" 0.35", text)
        self.assertIn("set_global_routing_random -seed 42", text)
        self.assertNotIn("ROUTING_LAYER_ADJUSTMENT", text)

    def test_manifest_is_resume_safe(self) -> None:
        protocol = {"protocol_hash": "abc", "phase": "B8.0"}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = initialize_manifest(root, protocol, "smoke")
            first["runs"][smoke_specs()[0].run_id]["status"] = "complete"
            (root / "abc" / "manifest.json").write_text(json.dumps(first))
            resumed = initialize_manifest(root, protocol, "smoke")
            self.assertEqual(
                resumed["runs"][smoke_specs()[0].run_id]["status"], "complete"
            )
            self.assertEqual(summarize_manifest(resumed, "smoke")["planned_runs"], 9)

    def test_command_serializes_action_without_shell(self) -> None:
        spec = smoke_specs()[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "flow").mkdir()
            command = build_flow_command(
                spec,
                orfs_root=root,
                protocol_hash="abc",
                threads=4,
                executor="native",
            )
            self.assertEqual(command[0], "make")
            self.assertIn("PLACE_DENSITY_LB_ADDON=0.00", command)
            self.assertIn("OR_SEED=42", command)
            self.assertTrue(any(value.startswith("FASTROUTE_TCL=") for value in command))

    def test_docker_command_pins_image_by_digest(self) -> None:
        spec = smoke_specs()[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "flow").mkdir()
            command = build_flow_command(
                spec,
                orfs_root=root,
                protocol_hash="abc",
                threads=4,
                executor="docker-shell",
                container_image="openroad/orfs:test",
                container_digest="sha256:abc",
            )
            self.assertEqual(command[:3], ["env", "OR_IMAGE=openroad/orfs:test@sha256:abc", "util/docker_shell"])

    def test_klayout_report_counts_multiplicity_by_rule(self) -> None:
        report = """<?xml version='1.0'?><report-database><items>
        <item><category>'m1.2'</category><multiplicity>2</multiplicity></item>
        <item><category>'via.1'</category><multiplicity>1</multiplicity></item>
        </items></report-database>"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.rdb"
            path.write_text(report)
            parsed = parse_klayout_report(path)
            self.assertEqual(parsed["total"], 3)
            self.assertEqual(parsed["by_rule"], {"m1.2": 2, "via.1": 1})

    def test_completion_requires_marker_and_matching_artifact_hashes(self) -> None:
        from training.b8_actionability import sha256_file

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = {}
            for name in ("final.gds", "checkpoint.odb", "metadata.json", "exact.rdb"):
                path = root / name
                path.write_text(name)
                files[name] = path
            record = {
                "artifacts": {
                    "final_gds": str(files["final.gds"]),
                    "final_gds_sha256": sha256_file(files["final.gds"]),
                    "checkpoint": str(files["checkpoint.odb"]),
                    "checkpoint_sha256": sha256_file(files["checkpoint.odb"]),
                    "metadata": str(files["metadata.json"]),
                    "metadata_sha256": sha256_file(files["metadata.json"]),
                    "exact_klayout_report": str(files["exact.rdb"]),
                    "exact_klayout_report_sha256": sha256_file(files["exact.rdb"]),
                }
            }
            self.assertFalse(completion_is_valid(record, root))
            (root / "COMPLETE").write_text("ok")
            self.assertTrue(completion_is_valid(record, root))
            files["metadata.json"].write_text("changed")
            self.assertFalse(completion_is_valid(record, root))


if __name__ == "__main__":
    unittest.main()
