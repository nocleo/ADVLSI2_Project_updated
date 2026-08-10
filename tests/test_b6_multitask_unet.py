from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch

    from training.localization_dataset import (
        B6LocalizationDataset,
        collate_localization_batch,
        load_layout_splits,
    )
    from training.localization_metrics import (
        connected_components,
        evaluate_prediction_records,
        match_raster_components,
        select_validation_thresholds,
    )
    from training.multitask_unet import MultiTaskLoss, MultiTaskUNet, parameter_count
    from training.classifier_models import BASELINE_MODEL, build_classifier
    from training.train_multitask_unet import (
        acceptance_gate,
        evaluate_b2_checkpoint,
        train_seed,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test runtime")
class B6MultiTaskUNetTest(unittest.TestCase):
    @staticmethod
    def _write_layout_artifact(root: Path, layout: str) -> None:
        layout_root = root / "layouts" / layout
        (layout_root / "tiles" / "clean").mkdir(parents=True)
        (layout_root / "tiles" / "dirty").mkdir(parents=True)
        vector = {
            "violation_id": f"{layout}:m1.2:000000",
            "owner_index": [0, 0],
            "midpoint_nm": [640, 640],
            "edge1_nm": [590, 600, 590, 680],
            "edge2_nm": [690, 600, 690, 680],
            "deficit_nm": 40,
        }
        (layout_root / "violations.jsonl").write_text(json.dumps(vector) + "\n")
        records = []
        for label in ("clean", "dirty"):
            image = np.zeros((200, 200), dtype=np.uint8)
            mask = np.zeros((160, 160), dtype=np.uint8)
            ids = []
            if label == "dirty":
                image[95:105, 95:105] = 1
                mask[75:85, 75:85] = 1
                ids = [vector["violation_id"]]
            relative = f"tiles/{label}/{label}.npz"
            np.savez_compressed(layout_root / relative, image=image, mask=mask)
            records.append(
                {
                    "tile_id": f"{layout}__{label}",
                    "layout": layout,
                    "label": label,
                    "grid_index": [0, 0],
                    "output_box_nm": [0, 0, 1280, 1280],
                    "image_path": relative,
                    "mask_pixels": int(mask.sum()),
                    "violation_ids": ids,
                }
            )
        (layout_root / "tiles.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n"
        )

    def test_model_outputs_registered_central_mask_and_classification(self) -> None:
        model = MultiTaskUNet(base_channels=4)
        segmentation, classification = model(torch.zeros(2, 1, 200, 200))
        self.assertEqual(tuple(segmentation.shape), (2, 1, 160, 160))
        self.assertEqual(tuple(classification.shape), (2, 2))
        self.assertLess(parameter_count(model), 1_000_000)

    def test_multitask_loss_is_finite_and_backpropagates(self) -> None:
        model = MultiTaskUNet(base_channels=4)
        masks = torch.zeros(2, 1, 160, 160)
        masks[1, :, 70:90, 70:90] = 1
        segmentation, classification = model(torch.zeros(2, 1, 200, 200))
        total, components = MultiTaskLoss(positive_weight=20)(
            segmentation, classification, masks, torch.tensor([0, 1])
        )
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(set(components), {"bce", "dice", "classification"})
        total.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_connected_components_and_matching(self) -> None:
        target = np.zeros((12, 12), dtype=np.uint8)
        target[1:4, 1:4] = 1
        target[7:10, 7:10] = 1
        prediction = target.copy()
        labels, components = connected_components(target)
        self.assertEqual(labels.shape, target.shape)
        self.assertEqual(len(components), 2)
        self.assertEqual(
            match_raster_components(prediction, target),
            {"true_positive": 2, "false_positive": 0, "false_negative": 0},
        )

    def test_exact_vector_owner_metrics_use_physical_coordinates(self) -> None:
        target = np.zeros((160, 160), dtype=np.uint8)
        target[78:83, 78:83] = 1
        probability = target.astype(np.float16)
        midpoint = [644, 636]
        metadata = {
            "layout": "layout_a",
            "grid_index": [0, 0],
            "output_box_nm": [0, 0, 1280, 1280],
            "vectors": [
                {
                    "violation_id": "layout_a:m1.2:000000",
                    "owner_index": [0, 0],
                    "midpoint_nm": midpoint,
                    "edge1_nm": [594, 600, 594, 672],
                    "edge2_nm": [694, 600, 694, 672],
                    "deficit_nm": 40,
                }
            ],
        }
        metrics = evaluate_prediction_records(
            [
                {
                    "mask_probability": probability,
                    "target_mask": target,
                    "class_probability": 0.9,
                    "label": 1,
                    "metadata": metadata,
                }
            ],
            segmentation_threshold=0.5,
            classification_threshold=0.5,
        )
        self.assertEqual(metrics["raster_objects"]["f1"], 1.0)
        self.assertEqual(metrics["exact_vector_owners"]["recall"], 1.0)
        self.assertLess(metrics["exact_vector_owners"]["centroid_error_nm"]["mean"], 12)

    def test_validation_thresholds_are_selected_without_test_data(self) -> None:
        target = np.zeros((160, 160), dtype=np.uint8)
        target[50:70, 50:70] = 1
        metadata = {
            "layout": "layout_a",
            "grid_index": [0, 0],
            "output_box_nm": [0, 0, 1280, 1280],
            "vectors": [],
        }
        records = [
            {
                "mask_probability": target.astype(np.float16) * 0.6,
                "target_mask": target,
                "class_probability": 0.7,
                "label": 1,
                "metadata": metadata,
            },
            {
                "mask_probability": np.zeros_like(target, dtype=np.float16),
                "target_mask": np.zeros_like(target),
                "class_probability": 0.2,
                "label": 0,
                "metadata": metadata,
            },
        ]
        selected = select_validation_thresholds(records, [0.5, 0.7], [0.3, 0.8])
        self.assertEqual(selected["segmentation"]["threshold"], 0.5)
        self.assertEqual(selected["classification"]["threshold"], 0.3)

    def test_family_split_and_dataset_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = root / "registry.json"
            protocols = root / "protocols.json"
            registry.write_text(
                json.dumps(
                    {
                        "layouts": [
                            {"name": "train_layout", "family": "train_family", "include_in_b1": True},
                            {"name": "val_layout", "family": "val_family", "include_in_b1": True},
                            {"name": "test_layout", "family": "test_family", "include_in_b1": True},
                        ]
                    }
                )
            )
            protocols.write_text(
                json.dumps(
                    {
                        "protocols": {
                            "unseen_layout_v1": {
                                "train_families": ["train_family"],
                                "validation_families": ["val_family"],
                                "test_families": ["test_family"],
                            }
                        }
                    }
                )
            )
            splits = load_layout_splits(registry, protocols)
            self.assertEqual(splits["validation"], ["val_layout"])

            self._write_layout_artifact(root / "dataset", "train_layout")
            dataset = B6LocalizationDataset(root / "dataset", ["train_layout"])
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.class_counts, {"clean": 1, "dirty": 1})
            batch = collate_localization_batch([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch[0].shape), (2, 1, 200, 200))
            self.assertEqual(tuple(batch[1].shape), (2, 1, 160, 160))

    def test_one_epoch_training_checkpoint_and_evaluation_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            for layout in ("train_layout", "val_layout", "test_layout"):
                self._write_layout_artifact(dataset_root, layout)
            args = argparse.Namespace(
                max_samples_per_split=None,
                no_augmentation=True,
                positive_weight_cap=50.0,
                protocol="unseen_layout_v1",
                epochs=1,
                patience=1,
                batch_size=2,
                learning_rate=1e-3,
                weight_decay=1e-4,
                base_channels=4,
                classification_weight=0.25,
                output_dir=root / "results",
                workers=0,
            )
            result = train_seed(
                args,
                dataset_root,
                "synthetic-dataset",
                {
                    "train": ["train_layout"],
                    "validation": ["val_layout"],
                    "test": ["test_layout"],
                },
                seed=42,
                device=torch.device("cpu"),
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["best_epoch"], 1)
            self.assertTrue((root / "results" / "seed_42" / "best.pth").is_file())
            self.assertIn("raster_objects", result["development_test"])

    def test_acceptance_gate_is_pre_registered(self) -> None:
        candidate = {
            "dirty_dice": {"mean": 0.80},
            "raster_object_f1": {"mean": 0.78},
            "exact_vector_owner_recall": {"mean": 0.90},
            "classification_recall": {"mean": 0.89},
        }
        baseline = {"classification_recall": {"mean": 0.90}}
        self.assertTrue(acceptance_gate(candidate, baseline)["passed"])
        candidate["classification_recall"]["mean"] = 0.87
        self.assertFalse(acceptance_gate(candidate, baseline)["passed"])

    def test_b2_full_box_localization_baseline_uses_same_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            self._write_layout_artifact(dataset_root, "layout_a")
            dataset = B6LocalizationDataset(dataset_root, ["layout_a"])
            checkpoint = root / "b2.pth"
            torch.save(build_classifier(BASELINE_MODEL).state_dict(), checkpoint)
            result = evaluate_b2_checkpoint(
                checkpoint,
                dataset,
                batch_size=2,
                workers=0,
                device=torch.device("cpu"),
            )
            self.assertEqual(
                result["localization_baseline"],
                "B2 probability over the full central output box",
            )
            self.assertEqual(result["metrics"]["samples"], 2)
            self.assertIn("raster_objects", result["metrics"])


if __name__ == "__main__":
    unittest.main()
