from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np

from training.dataset_manifest import build_manifest, parse_sample_path


def npy_bytes(array: np.ndarray) -> bytes:
    stream = BytesIO()
    np.save(stream, array)
    return stream.getvalue()


class DatasetManifestTest(unittest.TestCase):
    def make_registry(self, root: Path) -> tuple[Path, Path]:
        layouts = []
        for layout, family in (
            ("layout_train", "family_train"),
            ("layout_train_variant", "family_train"),
            ("layout_validation", "family_validation"),
            ("layout_test", "family_test"),
        ):
            layouts.append(
                {
                    "name": layout,
                    "family": family,
                    "circuit_type": "test",
                    "include_in_b1": True,
                    "source_url": "https://example.test/source",
                    "license": "Apache-2.0",
                    "provenance_status": "verified",
                }
            )
        registry = root / "registry.json"
        registry.write_text(
            json.dumps({"schema_version": 1, "generation": {}, "layouts": layouts})
        )
        protocols = root / "protocols.json"
        protocols.write_text(
            json.dumps(
                {
                    "protocols": {
                        "tile_random_reference": {
                            "seed": 7,
                            "fractions": {
                                "train": 0.5,
                                "validation": 0.25,
                                "test": 0.25,
                            },
                        },
                        "unseen_layout_v1": {
                            "train_families": ["family_train"],
                            "validation_families": ["family_validation"],
                            "test_families": ["family_test"],
                            "duplicate_resolution_priority": ["test", "validation", "train"],
                        },
                    }
                }
            )
        )
        return registry, protocols

    def make_dataset(self, root: Path, conflict: bool = False) -> Path:
        path = root / "dataset.zip"
        entries: dict[str, np.ndarray] = {}
        counter = 1
        for layout in (
            "layout_train",
            "layout_train_variant",
            "layout_validation",
            "layout_test",
        ):
            for label in ("clean", "dirty"):
                for sample_index in range(4):
                    entries[f"{label}/{layout}_tile_{counter}_{sample_index}.npy"] = (
                        np.random.default_rng(counter * 10 + sample_index).integers(
                            0, 2, size=(8, 8), dtype=np.uint8
                        )
                    )
                counter += 1

        duplicate = np.eye(8, dtype=np.uint8)
        entries["clean/layout_train_tile_100_100.npy"] = duplicate
        entries["clean/layout_test_tile_200_200.npy"] = duplicate
        if conflict:
            entries["dirty/layout_validation_tile_300_300.npy"] = duplicate

        with zipfile.ZipFile(path, "w") as archive:
            for name, array in sorted(entries.items()):
                archive.writestr(name, npy_bytes(array))
        return path

    def test_parse_sample_path_preserves_layout_underscores(self) -> None:
        self.assertEqual(
            parse_sample_path("dirty/tt_um_name_tile_123_456.npy"),
            ("dirty", "tt_um_name", 123, 456),
        )

    def test_grouped_protocol_has_no_family_or_hash_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, protocols = self.make_registry(root)
            manifest = build_manifest(
                self.make_dataset(root), registry, seed=7, protocols_path=protocols
            )
            records = {record["path"]: record for record in manifest["records"]}
            splits = manifest["protocols"]["unseen_layout_v1"]["splits"]

            family_locations: dict[str, set[str]] = {}
            hash_locations: dict[str, set[str]] = {}
            for split, paths in splits.items():
                for path in paths:
                    record = records[path]
                    family_locations.setdefault(record["layout_family"], set()).add(split)
                    hash_locations.setdefault(record["canonical_hash"], set()).add(split)
            self.assertTrue(all(len(locations) == 1 for locations in family_locations.values()))
            self.assertTrue(all(len(locations) == 1 for locations in hash_locations.values()))
            self.assertIn("clean/layout_test_tile_200_200.npy", splits["test"])
            self.assertNotIn("clean/layout_train_tile_100_100.npy", splits["train"])

    def test_contradictory_labels_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, protocols = self.make_registry(root)
            with self.assertRaisesRegex(ValueError, "Contradictory clean/dirty"):
                build_manifest(
                    self.make_dataset(root, conflict=True),
                    registry,
                    seed=7,
                    protocols_path=protocols,
                )


if __name__ == "__main__":
    unittest.main()
