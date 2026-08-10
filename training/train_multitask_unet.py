"""Train and evaluate the family-disjoint B6.2 multi-task U-Net."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.classifier_models import BASELINE_MODEL, build_classifier
from training.dataset_manifest import sha256_file
from training.localization_dataset import (
    B6LocalizationDataset,
    collate_localization_batch,
    load_layout_splits,
)
from training.localization_metrics import (
    binary_metrics,
    collect_model_outputs,
    evaluate_prediction_records,
    select_validation_thresholds,
)
from training.multitask_unet import MultiTaskLoss, MultiTaskUNet, parameter_count
from training.runtime_device import DEVICE_CHOICES, select_device


DEFAULT_DATASET = PROJECT_ROOT / "training_datasets" / "b6_localization_dataset.zip"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "b6_multitask_unet"
DEFAULT_REGISTRY = PROJECT_ROOT / "data" / "layout_registry.json"
DEFAULT_PROTOCOLS = PROJECT_ROOT / "data" / "evaluation_protocols.json"
DEFAULT_SEEDS = (42, 43, 44)
THRESHOLDS = tuple(round(value / 10, 1) for value in range(1, 10))


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


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def stable_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


@contextmanager
def extracted_dataset(path: Path) -> Iterator[tuple[Path, str]]:
    path = path.resolve()
    if path.is_dir():
        yield path, "unarchived-directory"
        return
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise FileNotFoundError(f"B6 dataset must be a directory or ZIP archive: {path}")
    with tempfile.TemporaryDirectory(prefix="advlsi-b6-") as temp_dir:
        root = Path(temp_dir)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(root)
        yield root, sha256_file(path)


def make_loader(
    dataset: B6LocalizationDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_localization_batch,
    )


def loss_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "bce": 0.0, "dice": 0.0, "classification": 0.0}
    samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, masks, labels, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            segmentation, classification = model(images)
            loss, components = criterion(segmentation, classification, masks, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            count = labels.size(0)
            totals["total"] += float(loss.detach()) * count
            for key, value in components.items():
                totals[key] += float(value.detach()) * count
            samples += count
    return {key: value / samples for key, value in totals.items()}


def _save_checkpoint(path: Path, model: torch.nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def train_seed(
    args: argparse.Namespace,
    dataset_root: Path,
    dataset_sha256: str,
    layout_splits: dict[str, list[str]],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    configure_determinism(seed)
    smoke_limit = args.max_samples_per_split
    train_set = B6LocalizationDataset(
        dataset_root,
        layout_splits["train"],
        augment=not args.no_augmentation,
        max_samples=smoke_limit,
        seed=seed,
    )
    validation_set = B6LocalizationDataset(
        dataset_root,
        layout_splits["validation"],
        augment=False,
        max_samples=smoke_limit,
        seed=seed,
    )
    development_test_set = B6LocalizationDataset(
        dataset_root,
        layout_splits["test"],
        augment=False,
        max_samples=smoke_limit,
        seed=seed,
    )
    positive_pixels, negative_pixels = train_set.mask_pixel_counts()
    positive_weight = min(negative_pixels / positive_pixels, args.positive_weight_cap)
    configuration = {
        "phase": "B6.2",
        "dataset_sha256": dataset_sha256,
        "protocol": args.protocol,
        "seed": seed,
        "epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "base_channels": args.base_channels,
        "classification_loss_weight": args.classification_weight,
        "positive_weight": positive_weight,
        "positive_weight_cap": args.positive_weight_cap,
        "augmentation": not args.no_augmentation,
        "max_samples_per_split": smoke_limit,
        "threshold_grid": THRESHOLDS,
    }
    configuration_id = stable_id(configuration)
    seed_dir = args.output_dir / f"seed_{seed}"
    result_path = seed_dir / "run.json"
    best_path = seed_dir / "best.pth"
    last_path = seed_dir / "last.pt"
    if result_path.is_file() and best_path.is_file():
        completed = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            completed.get("configuration_id") == configuration_id
            and completed.get("checkpoint_sha256") == sha256_file(best_path)
        ):
            print(f"[B6.2] seed {seed}: reusing completed run", flush=True)
            return completed

    model = MultiTaskUNet(args.base_channels).to(device)
    criterion = MultiTaskLoss(positive_weight, args.classification_weight).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_loader = make_loader(
        train_set, args.batch_size, True, seed, args.workers
    )
    validation_loader = make_loader(
        validation_set, args.batch_size, False, seed, args.workers
    )
    development_loader = make_loader(
        development_test_set, args.batch_size, False, seed, args.workers
    )

    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    if last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("configuration_id") == configuration_id:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            history = list(state["history"])
            best_validation_loss = float(state["best_validation_loss"])
            best_epoch = int(state["best_epoch"])
            stale_epochs = int(state["stale_epochs"])
            start_epoch = int(state["epoch"]) + 1
            print(f"[B6.2] seed {seed}: resuming at epoch {start_epoch}", flush=True)

    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = loss_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss = loss_epoch(model, validation_loader, criterion, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        improved = validation_loss["total"] < best_validation_loss - 1e-8
        if improved:
            best_validation_loss = validation_loss["total"]
            best_epoch = epoch
            stale_epochs = 0
            _save_checkpoint(best_path, model)
        else:
            stale_epochs += 1
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "configuration_id": configuration_id,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "best_validation_loss": best_validation_loss,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
            },
            last_path,
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": train_loss["total"],
                    "validation_loss": validation_loss["total"],
                    "best_epoch": best_epoch,
                    "stale_epochs": stale_epochs,
                }
            ),
            flush=True,
        )
        if stale_epochs >= args.patience:
            break

    if not best_path.is_file():
        raise RuntimeError(f"No B6.2 checkpoint was selected for seed {seed}")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    validation_outputs = collect_model_outputs(model, validation_loader, device)
    thresholds = select_validation_thresholds(validation_outputs, THRESHOLDS, THRESHOLDS)
    segmentation_threshold = thresholds["segmentation"]["threshold"]
    classification_threshold = thresholds["classification"]["threshold"]
    validation_metrics = evaluate_prediction_records(
        validation_outputs,
        segmentation_threshold,
        classification_threshold,
        include_per_layout=True,
    )
    del validation_outputs
    development_outputs = collect_model_outputs(model, development_loader, device)
    development_metrics = evaluate_prediction_records(
        development_outputs,
        segmentation_threshold,
        classification_threshold,
        include_per_layout=True,
    )
    del development_outputs

    result: dict[str, Any] = {
        "phase": "B6.2",
        "status": "smoke_only" if smoke_limit else "complete",
        "configuration_id": configuration_id,
        "configuration": configuration,
        "repository_commit": repository_commit(),
        "device": str(device),
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
        "model": {
            "name": "MultiTaskUNet",
            "parameters": parameter_count(model),
            "input_shape": [1, 200, 200],
            "segmentation_output_shape": [1, 160, 160],
        },
        "splits": {
            "train": {
                "layouts": layout_splits["train"],
                "samples": len(train_set),
                "classes": train_set.class_counts,
            },
            "validation": {
                "layouts": layout_splits["validation"],
                "samples": len(validation_set),
                "classes": validation_set.class_counts,
            },
            "development_test": {
                "layouts": layout_splits["test"],
                "samples": len(development_test_set),
                "classes": development_test_set.class_counts,
                "untouched_final_holdout": False,
            },
        },
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "selected_thresholds": thresholds,
        "validation": validation_metrics,
        "development_test": development_metrics,
        "checkpoint": portable_path(best_path),
        "checkpoint_sha256": sha256_file(best_path),
        "history": history,
    }
    seed_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def evaluate_b2_checkpoint(
    checkpoint: Path,
    dataset: B6LocalizationDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> dict[str, Any]:
    model = build_classifier(BASELINE_MODEL).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    records: list[dict[str, Any]] = []
    loader = make_loader(dataset, batch_size, False, 42, workers)
    with torch.no_grad():
        for images, masks, batch_labels, metadata in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            targets = masks.numpy().astype(np.uint8)
            for index, item in enumerate(metadata):
                probability = float(probabilities[index])
                records.append(
                    {
                        "mask_probability": np.full(
                            targets[index, 0].shape, probability, dtype=np.float16
                        ),
                        "class_probability": probability,
                        "target_mask": targets[index, 0],
                        "label": int(batch_labels[index]),
                        "metadata": item,
                    }
                )
    return {
        "checkpoint": portable_path(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "localization_baseline": "B2 probability over the full central output box",
        "metrics": evaluate_prediction_records(
            records,
            segmentation_threshold=0.5,
            classification_threshold=0.5,
            include_per_layout=True,
        ),
    }


def evaluate_b2_baseline(
    checkpoint_dir: Path,
    dataset_root: Path,
    layout_splits: dict[str, list[str]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    validation_set = B6LocalizationDataset(
        dataset_root,
        layout_splits["validation"],
        augment=False,
        max_samples=args.max_samples_per_split,
    )
    test_set = B6LocalizationDataset(
        dataset_root,
        layout_splits["test"],
        augment=False,
        max_samples=args.max_samples_per_split,
    )
    runs = []
    for seed in args.seeds:
        checkpoint = checkpoint_dir / f"unseen_layout_v1_seed_{seed}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing authoritative B2 checkpoint for seed {seed}: {checkpoint}"
            )
        runs.append(
            {
                "seed": seed,
                "validation": evaluate_b2_checkpoint(
                    checkpoint, validation_set, args.batch_size, args.workers, device
                ),
                "development_test": evaluate_b2_checkpoint(
                    checkpoint, test_set, args.batch_size, args.workers, device
                ),
            }
        )
    return {"model": BASELINE_MODEL, "threshold": 0.5, "runs": runs}


def metric_stats(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "runs": len(values),
        "mean": statistics.fmean(values),
        "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize_unet_runs(runs: Sequence[dict[str, Any]], split: str) -> dict[str, Any]:
    paths = {
        "classification_accuracy": ("classification", "accuracy"),
        "classification_precision": ("classification", "precision"),
        "classification_recall": ("classification", "recall"),
        "classification_f1": ("classification", "f1"),
        "dirty_dice": ("pixel_dirty_only", "dice"),
        "dirty_iou": ("pixel_dirty_only", "iou"),
        "raster_object_precision": ("raster_objects", "precision"),
        "raster_object_recall": ("raster_objects", "recall"),
        "raster_object_f1": ("raster_objects", "f1"),
        "exact_vector_owner_recall": ("exact_vector_owners", "recall"),
    }
    return {
        name: metric_stats([float(run[split][section][metric]) for run in runs])
        for name, (section, metric) in paths.items()
    }


def summarize_b2_runs(baseline: dict[str, Any], split: str) -> dict[str, Any]:
    paths = {
        "classification_accuracy": ("classification", "accuracy"),
        "classification_precision": ("classification", "precision"),
        "classification_recall": ("classification", "recall"),
        "classification_f1": ("classification", "f1"),
        "dirty_dice": ("pixel_dirty_only", "dice"),
        "dirty_iou": ("pixel_dirty_only", "iou"),
        "raster_object_precision": ("raster_objects", "precision"),
        "raster_object_recall": ("raster_objects", "recall"),
        "raster_object_f1": ("raster_objects", "f1"),
        "exact_vector_owner_recall": ("exact_vector_owners", "recall"),
    }
    return {
        name: metric_stats(
            [float(run[split]["metrics"][section][metric]) for run in baseline["runs"]]
        )
        for name, (section, metric) in paths.items()
    }


def acceptance_gate(
    unet_development: dict[str, Any],
    b2_development: dict[str, Any] | None,
) -> dict[str, Any]:
    checks = {
        "dirty_dice_at_least_0_75": unet_development["dirty_dice"]["mean"] >= 0.75,
        "raster_object_f1_at_least_0_75": unet_development["raster_object_f1"]["mean"] >= 0.75,
        "exact_vector_owner_recall_at_least_0_85": unet_development[
            "exact_vector_owner_recall"
        ]["mean"]
        >= 0.85,
    }
    if b2_development is not None:
        checks["classification_recall_within_2_points_of_b2"] = (
            unet_development["classification_recall"]["mean"]
            >= b2_development["classification_recall"]["mean"] - 0.02
        )
    return {"passed": all(checks.values()), "checks": checks}


def render_markdown(summary: dict[str, Any]) -> str:
    def percent(metric: dict[str, Any]) -> str:
        return f"{100*metric['mean']:.2f}% ± {100*metric['sample_stddev']:.2f}%"

    unet = summary["unet"]["development_test"]
    b2 = summary.get("b2_on_b6", {}).get("development_test")
    rows = [
        ("Classification accuracy", percent(unet["classification_accuracy"]), percent(b2["classification_accuracy"]) if b2 else "not run"),
        ("Dirty recall", percent(unet["classification_recall"]), percent(b2["classification_recall"]) if b2 else "not run"),
        ("Dirty F1", percent(unet["classification_f1"]), percent(b2["classification_f1"]) if b2 else "not run"),
        ("Dirty-mask Dice", percent(unet["dirty_dice"]), percent(b2["dirty_dice"]) if b2 else "not run"),
        ("Raster-object F1", percent(unet["raster_object_f1"]), percent(b2["raster_object_f1"]) if b2 else "not run"),
        ("Exact-vector owner recall", percent(unet["exact_vector_owner_recall"]), percent(b2["exact_vector_owner_recall"]) if b2 else "not run"),
    ]
    table = "\n".join(f"| {name} | {candidate} | {baseline} |" for name, candidate, baseline in rows)
    status = "passed" if summary["acceptance"]["passed"] else "did not pass"
    return f"""# B6.2 multi-task U-Net

Status: **{summary['status']}**; the pre-registered development gate **{status}**.

The model and both thresholds were selected using the three validation layout
families only. The three historical test families are a development
confirmation because earlier project decisions have already inspected them;
they are not the untouched B9 final holdout.

| Metric | Multi-task U-Net | B2 on the same B6 tiles |
|---|---:|---:|
{table}

Raster-object metrics operate on per-tile connected components. Exact-vector
recall counts each KLayout violation only in its unique owner tile. Full-layout
stitching, unique prediction precision, exact edge recovery, and false alarms
per area remain B7 work.
"""


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device, args.cpu)
    print(json.dumps({"event": "device_selected", "device": str(device)}), flush=True)
    layout_splits = load_layout_splits(args.registry, args.protocols, args.protocol)
    with extracted_dataset(args.dataset) as (dataset_root, dataset_sha256):
        runs = [
            train_seed(args, dataset_root, dataset_sha256, layout_splits, seed, device)
            for seed in args.seeds
        ]
        baseline = None
        if not args.skip_b2_comparison:
            baseline = evaluate_b2_baseline(
                args.b2_checkpoint_dir, dataset_root, layout_splits, args, device
            )

    unet_validation = summarize_unet_runs(runs, "validation")
    unet_development = summarize_unet_runs(runs, "development_test")
    b2_validation = summarize_b2_runs(baseline, "validation") if baseline else None
    b2_development = summarize_b2_runs(baseline, "development_test") if baseline else None
    gate = acceptance_gate(unet_development, b2_development)
    smoke = args.max_samples_per_split is not None
    summary: dict[str, Any] = {
        "phase": "B6.2",
        "status": "smoke_only" if smoke else "complete",
        "official_result": not smoke and baseline is not None,
        "dataset": portable_path(args.dataset),
        "dataset_sha256": dataset_sha256,
        "protocol": args.protocol,
        "seeds": args.seeds,
        "untouched_final_holdout_used": False,
        "unet": {
            "validation": unet_validation,
            "development_test": unet_development,
            "runs": [portable_path(args.output_dir / f"seed_{seed}" / "run.json") for seed in args.seeds],
        },
        "b2_on_b6": {
            "validation": b2_validation,
            "development_test": b2_development,
            "runs": baseline["runs"] if baseline else None,
        },
        "acceptance": gate,
        "repository_commit": repository_commit(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--protocols", type=Path, default=DEFAULT_PROTOCOLS)
    parser.add_argument("--protocol", default="unseen_layout_v1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--b2-checkpoint-dir", type=Path)
    parser.add_argument("--skip-b2-comparison", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--classification-weight", type=float, default=0.25)
    parser.add_argument("--positive-weight-cap", type=float, default=50.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        help="Smoke testing only; any result produced with this option is non-official.",
    )
    args = parser.parse_args(argv)
    if not args.skip_b2_comparison and args.b2_checkpoint_dir is None:
        parser.error("--b2-checkpoint-dir is required unless --skip-b2-comparison is used")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("epochs, patience, and batch size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run_experiment(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
