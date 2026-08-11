"""Family-disjoint PyTorch dataset for the B6 localization artifact."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_NAMES = ("train", "validation", "test")


def load_layout_splits(
    registry_path: Path,
    protocols_path: Path,
    protocol_name: str = "unseen_layout_v1",
) -> dict[str, list[str]]:
    """Resolve the frozen family split and prove that no family crosses splits."""

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    protocols = json.loads(protocols_path.read_text(encoding="utf-8"))
    try:
        protocol = protocols["protocols"][protocol_name]
    except KeyError as error:
        raise ValueError(f"Unknown evaluation protocol: {protocol_name}") from error

    included = {
        item["name"]: item["family"]
        for item in registry["layouts"]
        if item.get("include_in_b1", False)
    }
    family_to_layouts: dict[str, list[str]] = {}
    for layout, family in included.items():
        family_to_layouts.setdefault(family, []).append(layout)

    splits: dict[str, list[str]] = {}
    seen_families: set[str] = set()
    for split in SPLIT_NAMES:
        families = list(protocol[f"{split}_families"])
        overlap = seen_families.intersection(families)
        if overlap:
            raise ValueError(f"Layout families cross B6 splits: {sorted(overlap)}")
        seen_families.update(families)
        missing = sorted(set(families) - set(family_to_layouts))
        if missing:
            raise ValueError(f"Protocol families are absent from the registry: {missing}")
        splits[split] = sorted(
            layout for family in families for layout in family_to_layouts[family]
        )

    expected = set(included)
    assigned = {layout for layouts in splits.values() for layout in layouts}
    if assigned != expected:
        raise ValueError(
            "B6 protocol does not cover every registered layout: "
            f"missing={sorted(expected-assigned)}, extra={sorted(assigned-expected)}"
        )
    return splits


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class PairedManhattanAugmentation:
    """Apply one physical symmetry identically to the image and central mask."""

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        turns = int(torch.randint(0, 4, ()).item())
        image = torch.rot90(image, turns, dims=(-2, -1))
        mask = torch.rot90(mask, turns, dims=(-2, -1))
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(-2,))
            mask = torch.flip(mask, dims=(-2,))
        return image, mask


class B6LocalizationDataset(Dataset):
    """Load aligned 200x200 inputs, 160x160 masks, and exact vectors."""

    def __init__(
        self,
        root: Path,
        layouts: Sequence[str],
        augment: bool = False,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.root = root.resolve()
        self.transform = PairedManhattanAugmentation() if augment else None
        self.records: list[dict[str, Any]] = []
        self.vectors: dict[str, dict[str, Any]] = {}

        for layout in sorted(layouts):
            layout_root = self.root / "layouts" / layout
            if not layout_root.is_dir():
                raise FileNotFoundError(f"Missing B6 layout artifact: {layout_root}")
            for vector in _read_jsonl(layout_root / "violations.jsonl"):
                violation_id = vector["violation_id"]
                if violation_id in self.vectors:
                    raise ValueError(f"Duplicate exact violation ID: {violation_id}")
                self.vectors[violation_id] = vector
            for record in _read_jsonl(layout_root / "tiles.jsonl"):
                if record["layout"] != layout:
                    raise ValueError(f"Tile/layout mismatch in {layout_root}")
                item = dict(record)
                item["array_path"] = str(layout_root / record["image_path"])
                self.records.append(item)

        if not self.records:
            raise ValueError("The selected B6 split contains no tiles")
        if max_samples is not None and max_samples < len(self.records):
            self.records = _balanced_sample(self.records, max_samples, seed)
        self._validate_records()

    def _validate_records(self) -> None:
        ids: set[str] = set()
        labels: set[str] = set()
        for record in self.records:
            tile_id = record["tile_id"]
            if tile_id in ids:
                raise ValueError(f"Duplicate B6 tile ID: {tile_id}")
            ids.add(tile_id)
            labels.add(record["label"])
            path = Path(record["array_path"])
            if not path.is_file() or self.root not in path.resolve().parents:
                raise FileNotFoundError(f"Invalid B6 tile path: {path}")
            missing = set(record["violation_ids"]) - set(self.vectors)
            if missing:
                raise ValueError(f"Tile references missing vectors: {sorted(missing)[:3]}")
            if record["label"] not in {"clean", "dirty"}:
                raise ValueError(f"Unsupported B6 label: {record['label']}")
        if labels != {"clean", "dirty"}:
            raise ValueError(f"B6 split must contain both classes, got {sorted(labels)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]]:
        record = self.records[index]
        with np.load(record["array_path"]) as arrays:
            image_array = arrays["image"].astype(np.float32, copy=True)
            mask_array = arrays["mask"].astype(np.float32, copy=True)
        if image_array.shape != (200, 200) or mask_array.shape != (160, 160):
            raise ValueError(
                f"Unexpected B6 shapes for {record['tile_id']}: "
                f"image={image_array.shape}, mask={mask_array.shape}"
            )
        image = torch.from_numpy(image_array).unsqueeze(0)
        mask = torch.from_numpy(mask_array).unsqueeze(0)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        label = int(record["label"] == "dirty")
        if bool(mask.any()) != bool(label):
            raise ValueError(f"Mask/label mismatch for {record['tile_id']}")
        metadata = {key: value for key, value in record.items() if key != "array_path"}
        metadata["vectors"] = [self.vectors[item] for item in record["violation_ids"]]
        return image, mask, label, metadata

    @property
    def class_counts(self) -> dict[str, int]:
        return {
            label: sum(record["label"] == label for record in self.records)
            for label in ("clean", "dirty")
        }

    def mask_pixel_counts(self) -> tuple[int, int]:
        positive = sum(int(record["mask_pixels"]) for record in self.records)
        total = len(self.records) * 160 * 160
        return positive, total - positive


def _balanced_sample(
    records: Sequence[dict[str, Any]], max_samples: int, seed: int
) -> list[dict[str, Any]]:
    """Deterministic smoke-only subset that retains both classes and layouts."""

    if max_samples < 2:
        raise ValueError("--max-samples-per-split must be at least 2")
    rng = random.Random(seed)
    by_label = {
        label: [record for record in records if record["label"] == label]
        for label in ("clean", "dirty")
    }
    per_label = max_samples // 2
    selected: list[dict[str, Any]] = []
    for label in ("clean", "dirty"):
        values = list(by_label[label])
        rng.shuffle(values)
        selected.extend(values[:per_label])
    if len(selected) < max_samples:
        remainder = [record for record in records if record not in selected]
        rng.shuffle(remainder)
        selected.extend(remainder[: max_samples - len(selected)])
    return sorted(selected, key=lambda record: record["tile_id"])


def collate_localization_batch(
    batch: Iterable[tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    images, masks, labels, metadata = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(masks),
        torch.tensor(labels, dtype=torch.long),
        list(metadata),
    )
