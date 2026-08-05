"""Train and evaluate the paper-style binary DRC classifier.

This is the script counterpart of the baseline Colab notebook.  It makes the
training step runnable from a clone and provides a small ``--max-samples``
mode for end-to-end smoke tests.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import tempfile
import time
import warnings
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "run_inference_pc_optimized"))

from training.classifier_models import (
    BASELINE_MODEL,
    MODEL_NAMES,
    build_classifier,
    model_source_path,
)
from training.dataset_manifest import parse_sample_path, sha256_file
from training.runtime_device import DEVICE_CHOICES, select_device


MODEL_SOURCE = model_source_path(BASELINE_MODEL)
OPTIMIZER_LABELS = {"rmsprop": "RMSprop", "adam": "Adam"}


def portable_path(path: Path | None) -> str | None:
    """Render project files without embedding a machine-specific checkout path."""

    if path is None:
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def repository_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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

    def __init__(
        self,
        root_dir: Path,
        augment: bool = True,
        sample_paths: list[str] | None = None,
    ) -> None:
        self.samples: list[tuple[Path, int]] = []
        self.relative_paths: list[str] = []
        if sample_paths is None:
            for label, class_name in enumerate(self.CLASSES):
                class_dir = root_dir / class_name
                if not class_dir.is_dir():
                    raise FileNotFoundError(f"Missing dataset directory: {class_dir}")
                for path in sorted(class_dir.glob("*.npy")):
                    self.samples.append((path, label))
                    self.relative_paths.append(path.relative_to(root_dir).as_posix())
        else:
            for relative_path in sample_paths:
                class_name, _, _, _ = parse_sample_path(relative_path)
                path = root_dir / PurePosixPath(relative_path)
                if not path.is_file():
                    raise FileNotFoundError(f"Manifest sample is missing from archive: {relative_path}")
                self.samples.append((path, self.CLASSES.index(class_name)))
                self.relative_paths.append(relative_path)
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
    return {
        "samples": count,
        "loss": total_loss / count,
        **classification_metrics(all_labels, all_predictions),
    }


def build_optimizer(
    model: torch.nn.Module,
    name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build a supported optimizer without changing any other training factor."""

    parameters = model.parameters()
    if name == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def collapse_warning(metrics: dict[str, object], split: str, epoch: int | None = None) -> str | None:
    counts = metrics["predicted_class_counts"]
    if not isinstance(counts, dict) or all(counts.get(name, 0) > 0 for name in DRCDataset.CLASSES):
        return None
    where = f" at epoch {epoch}" if epoch is not None else ""
    message = f"Model predicted only one class on the {split} split{where}: {counts}"
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return message


def load_protocol(
    manifest_path: Path, dataset_path: Path, protocol_name: str
) -> tuple[dict[str, object], dict[str, list[str]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("dataset", {}).get("archive_sha256")
    actual_hash = sha256_file(dataset_path)
    if expected_hash != actual_hash:
        raise ValueError(
            "Dataset archive does not match the split manifest: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    try:
        protocol = manifest["protocols"][protocol_name]
        splits = protocol["splits"]
    except KeyError as error:
        available = sorted(manifest.get("protocols", {}))
        raise ValueError(
            f"Protocol {protocol_name!r} is absent from {manifest_path}; available: {available}"
        ) from error
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError(f"Protocol {protocol_name!r} must define train/validation/test")
    all_paths = [path for paths in splits.values() for path in paths]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError(f"Protocol {protocol_name!r} contains sample paths in multiple splits")
    return manifest, {name: list(splits[name]) for name in ("train", "validation", "test")}


def per_layout_test_metrics(
    model: torch.nn.Module,
    data_root: Path,
    test_paths: list[str],
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, object]]:
    by_layout: dict[str, list[str]] = {}
    for path in test_paths:
        _, layout, _, _ = parse_sample_path(path)
        by_layout.setdefault(layout, []).append(path)
    results: dict[str, dict[str, object]] = {}
    for layout, paths in sorted(by_layout.items()):
        dataset = DRCDataset(data_root, augment=False, sample_paths=paths)
        results[layout] = evaluate(
            model, DataLoader(dataset, batch_size=batch_size), device
        )
    return results


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    with tempfile.TemporaryDirectory(prefix="advlsi-training-") as temp_dir:
        data_root = Path(temp_dir)
        with zipfile.ZipFile(args.dataset) as archive:
            archive.extractall(data_root)

        manifest: dict[str, object] | None = None
        protocol_splits: dict[str, list[str]] | None = None
        if args.manifest:
            if args.max_samples:
                raise ValueError("--max-samples cannot alter a frozen manifest protocol")
            manifest, protocol_splits = load_protocol(
                args.manifest, args.dataset, args.protocol
            )
            train_set = DRCDataset(
                data_root,
                augment=not args.no_augmentation,
                sample_paths=protocol_splits["train"],
            )
            val_set = DRCDataset(
                data_root, augment=False, sample_paths=protocol_splits["validation"]
            )
            test_set = DRCDataset(
                data_root, augment=False, sample_paths=protocol_splits["test"]
            )
            selected_sample_count = sum(len(paths) for paths in protocol_splits.values())
        else:
            evaluation_dataset = DRCDataset(data_root, augment=False)
            selected_indices = list(range(len(evaluation_dataset)))
            if args.max_samples and args.max_samples < len(evaluation_dataset):
                selected_indices = balanced_indices(evaluation_dataset, args.max_samples, args.seed)

            train_indices, val_indices, test_indices = split_indices(selected_indices, args.seed)
            training_dataset = DRCDataset(data_root, augment=not args.no_augmentation)
            train_set = Subset(training_dataset, train_indices)
            val_set = Subset(evaluation_dataset, val_indices)
            test_set = Subset(evaluation_dataset, test_indices)
            selected_sample_count = len(selected_indices)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        val_loader = DataLoader(val_set, batch_size=args.batch_size)
        test_loader = DataLoader(test_set, batch_size=args.batch_size)

        device = select_device(args.device, args.cpu)
        print(json.dumps({"event": "device_selected", "device": str(device)}), flush=True)
        model = build_classifier(args.model).to(device)
        optimizer = build_optimizer(
            model,
            args.optimizer,
            args.learning_rate,
            args.weight_decay,
        )
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=args.scheduler_factor,
                patience=args.scheduler_patience,
                min_lr=args.scheduler_min_lr,
            )
            if args.scheduler == "plateau"
            else None
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        history: list[dict[str, object]] = []
        run_warnings: list[str] = []
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        stopped_early = False
        training_started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
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
            learning_rate = float(optimizer.param_groups[0]["lr"])
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
                "learning_rate": learning_rate,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            mean_epoch_seconds = (time.perf_counter() - training_started) / epoch
            epoch_result["estimated_remaining_seconds"] = mean_epoch_seconds * (
                args.epochs - epoch
            )
            history.append(epoch_result)
            print(json.dumps(epoch_result), flush=True)

            warning = collapse_warning(validation_metrics, "validation", epoch)
            if warning:
                run_warnings.append(warning)
            validation_loss = float(validation_metrics["loss"])
            if validation_loss < best_validation_loss - args.early_stopping_min_delta:
                best_validation_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
            if scheduler is not None:
                scheduler.step(validation_loss)
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                stopped_early = True
                break

        if best_state is None:
            raise RuntimeError("Training completed without producing a checkpoint")
        model.load_state_dict(best_state)
        test_metrics: dict[str, object] | None = None
        layout_metrics: dict[str, dict[str, object]] = {}
        if not args.skip_test:
            test_metrics = evaluate(model, test_loader, device)
            layout_metrics = (
                per_layout_test_metrics(
                    model,
                    data_root,
                    protocol_splits["test"],
                    args.batch_size,
                    device,
                )
                if protocol_splits is not None
                else {}
            )
            warning = collapse_warning(test_metrics, "test")
            if warning:
                run_warnings.append(warning)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, args.output)

    best_validation_metrics = next(
        record for record in history if int(record["epoch"]) == best_epoch
    )

    result = {
        "dataset": portable_path(args.dataset),
        "dataset_archive_sha256": sha256_file(args.dataset),
        "manifest": portable_path(args.manifest),
        "manifest_id": manifest.get("manifest_id") if manifest else None,
        "protocol": args.protocol if args.manifest else "b0-tile-random",
        "samples": selected_sample_count,
        "split": {"train": len(train_set), "validation": len(val_set), "test": len(test_set)},
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "model": args.model,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_source_sha256": sha256_file(model_source_path(args.model)),
        "trainer_source_sha256": sha256_file(Path(__file__)),
        "optimizer": OPTIMIZER_LABELS[args.optimizer],
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "scheduler_factor": args.scheduler_factor if scheduler is not None else None,
        "scheduler_patience": args.scheduler_patience if scheduler is not None else None,
        "scheduler_min_lr": args.scheduler_min_lr if scheduler is not None else None,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "batch_size": args.batch_size,
        "augmentation": "none" if args.no_augmentation else "Manhattan rotations and reflections (training only)",
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_validation_metrics": {
            "loss": best_validation_metrics["validation_loss"],
            "accuracy": best_validation_metrics["validation_accuracy"],
            "precision": best_validation_metrics["validation_precision"],
            "recall": best_validation_metrics["validation_recall"],
            "f1": best_validation_metrics["validation_f1"],
            "predicted_class_counts": best_validation_metrics[
                "validation_predicted_class_counts"
            ],
            "confusion_matrix": best_validation_metrics[
                "validation_confusion_matrix"
            ],
        },
        "test_evaluated": test_metrics is not None,
        "weights": portable_path(args.output),
        "weights_sha256": sha256_file(args.output),
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
        "repository_commit": repository_commit(),
        "warnings": run_warnings,
        "history": history,
    }
    if test_metrics is not None:
        result.update(
            {
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_predicted_class_counts": test_metrics[
                    "predicted_class_counts"
                ],
                "test_confusion_matrix": test_metrics["confusion_matrix"],
                "test_per_layout": layout_metrics,
            }
        )
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
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default=BASELINE_MODEL,
        help="Classifier architecture (default preserves the B2 baseline).",
    )
    parser.add_argument(
        "--optimizer",
        choices=("rmsprop", "adam"),
        default="rmsprop",
        help="Optimizer to use (default preserves the B2 RMSprop baseline).",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--scheduler",
        choices=("none", "plateau"),
        default="none",
        help="Validation-loss learning-rate scheduler (default preserves B2).",
    )
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving validation epochs; 0 disables it.",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Dataset manifest containing frozen split protocols.",
    )
    parser.add_argument(
        "--protocol",
        default="unseen_layout_v1",
        help="Protocol key inside --manifest (default: unseen_layout_v1).",
    )
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Train/select a checkpoint without evaluating the frozen test split.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Execution backend; auto selects CUDA, then Apple MPS, then CPU.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    if args.weight_decay < 0:
        parser.error("weight decay cannot be negative")
    if not 0 < args.scheduler_factor < 1:
        parser.error("scheduler factor must be between zero and one")
    if args.scheduler_patience < 0 or args.scheduler_min_lr <= 0:
        parser.error("scheduler patience must be non-negative and min LR positive")
    if args.early_stopping_patience < 0 or args.early_stopping_min_delta < 0:
        parser.error("early-stopping values cannot be negative")
    return args


if __name__ == "__main__":
    train(parse_args())
