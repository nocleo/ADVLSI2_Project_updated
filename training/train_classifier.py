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
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "run_inference_pc_optimized"))

from define_cnn_model import NCSU_DRCNN


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

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ]
        ) if augment else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        tile = torch.from_numpy(np.load(path).astype(np.float32)).unsqueeze(0)
        if self.transform is not None:
            tile = self.transform(tile)
        return tile, label


def balanced_subset(dataset: DRCDataset, max_samples: int, seed: int) -> Subset:
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
    return Subset(dataset, selected[:max_samples])


def split_dataset(dataset: Dataset, seed: int) -> tuple[Dataset, Dataset, Dataset]:
    total = len(dataset)
    if total < 3:
        raise ValueError("At least three samples are required")
    test_size = max(1, round(total * 0.05))
    val_size = max(1, round(total * 0.15))
    train_size = total - val_size - test_size
    return tuple(
        random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )
    )


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = correct = count = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / count, correct / count


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with tempfile.TemporaryDirectory(prefix="advlsi-training-") as temp_dir:
        data_root = Path(temp_dir)
        with zipfile.ZipFile(args.dataset) as archive:
            archive.extractall(data_root)

        dataset = DRCDataset(data_root, augment=not args.no_augmentation)
        selected: Dataset = dataset
        if args.max_samples and args.max_samples < len(dataset):
            selected = balanced_subset(dataset, args.max_samples, args.seed)

        train_set, val_set, test_set = split_dataset(selected, args.seed)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size)
        test_loader = DataLoader(test_set, batch_size=args.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        model = NCSU_DRCNN().to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        history: list[dict[str, float]] = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = count = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(inputs), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * labels.size(0)
                count += labels.size(0)
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            epoch_result = {
                "epoch": epoch,
                "train_loss": running_loss / count,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
            }
            history.append(epoch_result)
            print(json.dumps(epoch_result))

        test_loss, test_accuracy = evaluate(model, test_loader, device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.output)

    result = {
        "dataset": str(args.dataset),
        "samples": len(selected),
        "split": {"train": len(train_set), "validation": len(val_set), "test": len(test_set)},
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "weights": str(args.output),
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
