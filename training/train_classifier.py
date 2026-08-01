"""Train and evaluate the paper-style binary DRC classifier.

This is the script counterpart of the baseline Colab notebook.  It makes the
training step runnable from a clone and provides a small ``--max-samples``
mode for end-to-end smoke tests.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "run_inference_pc_optimized"))

from define_cnn_model import NCSU_DRCNN


class ManhattanAugmentation:
    """Apply only geometry-preserving transforms for an isotropic Manhattan rule."""

    def __call__(self, tile: torch.Tensor) -> torch.Tensor:
        tile = torch.rot90(tile, int(torch.randint(0, 4, ()).item()), dims=(-2, -1))
        if torch.rand(()) < 0.5:
            tile = torch.flip(tile, dims=(-1,))
        if torch.rand(()) < 0.5:
            tile = torch.flip(tile, dims=(-2,))
        return tile


class DRCDataset(Dataset):
    CLASSES = ("clean", "dirty")

    def __init__(self, root_dir: Path, augment: bool = True) -> None:
        self.samples: list[tuple[Path, int]] = []
        for label, class_name in enumerate(self.CLASSES):
            class_dir = root_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing dataset directory: {class_dir}")
            self.samples.extend((path, label) for path in sorted(class_dir.glob("*.npy")))
        if not self.samples:
            raise ValueError(f"No .npy tiles found under {root_dir}")

        self.transform = ManhattanAugmentation() if augment else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        tile = torch.from_numpy(np.load(path).astype(np.float32)).unsqueeze(0)
        if self.transform is not None:
            tile = self.transform(tile)
        return tile, label


def balanced_indices(dataset: DRCDataset, max_samples: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for index, (_, label) in enumerate(dataset.samples):
        by_label[label].append(index)
    per_class = max(1, max_samples // 2)
    selected: list[int] = []
    for indices in by_label.values():
        rng.shuffle(indices)
        selected.extend(indices[:per_class])
    rng.shuffle(selected)
    return selected[:max_samples]


def split_indices(indices: list[int], seed: int) -> tuple[list[int], list[int], list[int]]:
    total = len(indices)
    if total < 3:
        raise ValueError("At least three samples are required")
    test_size = max(1, round(total * 0.05))
    val_size = max(1, round(total * 0.15))
    train_size = total - val_size - test_size
    splits = random_split(
        indices,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return tuple(list(split) for split in splits)


def classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, object]:
    if not labels:
        raise ValueError("Cannot calculate metrics for an empty dataset")
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "predicted_class_counts": {
            "clean": predictions.count(0),
            "dirty": predictions.count(1),
        },
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = count = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
            count += labels.size(0)
    return {"loss": total_loss / count, **classification_metrics(all_labels, all_predictions)}


def collapse_warning(metrics: dict[str, object], split: str, epoch: int | None = None) -> str | None:
    counts = metrics["predicted_class_counts"]
    if not isinstance(counts, dict) or all(counts.get(name, 0) > 0 for name in DRCDataset.CLASSES):
        return None
    where = f" at epoch {epoch}" if epoch is not None else ""
    message = f"Model predicted only one class on the {split} split{where}: {counts}"
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return message


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with tempfile.TemporaryDirectory(prefix="advlsi-training-") as temp_dir:
        data_root = Path(temp_dir)
        with zipfile.ZipFile(args.dataset) as archive:
            archive.extractall(data_root)

        evaluation_dataset = DRCDataset(data_root, augment=False)
        selected_indices = list(range(len(evaluation_dataset)))
        if args.max_samples and args.max_samples < len(evaluation_dataset):
            selected_indices = balanced_indices(evaluation_dataset, args.max_samples, args.seed)

        train_indices, val_indices, test_indices = split_indices(selected_indices, args.seed)
        training_dataset = DRCDataset(data_root, augment=not args.no_augmentation)
        train_set = Subset(training_dataset, train_indices)
        val_set = Subset(evaluation_dataset, val_indices)
        test_set = Subset(evaluation_dataset, test_indices)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size)
        test_loader = DataLoader(test_set, batch_size=args.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        model = NCSU_DRCNN().to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        history: list[dict[str, object]] = []
        run_warnings: list[str] = []
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        best_validation_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = count = 0
            train_labels: list[int] = []
            train_predictions: list[int] = []
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(inputs)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * labels.size(0)
                count += labels.size(0)
                train_labels.extend(labels.cpu().tolist())
                train_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            train_metrics = classification_metrics(train_labels, train_predictions)
            validation_metrics = evaluate(model, val_loader, device)
            epoch_result = {
                "epoch": epoch,
                "train_loss": running_loss / count,
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "train_predicted_class_counts": train_metrics["predicted_class_counts"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_precision": validation_metrics["precision"],
                "validation_recall": validation_metrics["recall"],
                "validation_f1": validation_metrics["f1"],
                "validation_predicted_class_counts": validation_metrics["predicted_class_counts"],
                "validation_confusion_matrix": validation_metrics["confusion_matrix"],
            }
            history.append(epoch_result)
            print(json.dumps(epoch_result))

            warning = collapse_warning(validation_metrics, "validation", epoch)
            if warning:
                run_warnings.append(warning)
            validation_loss = float(validation_metrics["loss"])
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.state_dict().items()
                }

        if best_state is None:
            raise RuntimeError("Training completed without producing a checkpoint")
        model.load_state_dict(best_state)
        test_metrics = evaluate(model, test_loader, device)
        warning = collapse_warning(test_metrics, "test")
        if warning:
            run_warnings.append(warning)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, args.output)

    result = {
        "dataset": str(args.dataset),
        "samples": len(selected_indices),
        "split": {"train": len(train_set), "validation": len(val_set), "test": len(test_set)},
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "augmentation": "none" if args.no_augmentation else "Manhattan rotations and reflections (training only)",
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_predicted_class_counts": test_metrics["predicted_class_counts"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "weights": str(args.output),
        "warnings": run_warnings,
        "history": history,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "training_datasets" / "combined_training_dataset.zip",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "ncsu_drcnn_weights.pth")
    parser.add_argument("--metrics", type=Path, default=PROJECT_ROOT / "training_metrics.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
