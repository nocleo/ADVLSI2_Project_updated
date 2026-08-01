"""Build and validate leakage-aware manifests for DRC tile datasets."""

from __future__ import annotations

import hashlib
import json
import random
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 1
CLASS_NAMES = ("clean", "dirty")
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_PRIORITY = {"train": 0, "validation": 1, "test": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    """Hash logical array content independently of the .npy container header."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def canonical_manhattan_hash(array: np.ndarray) -> str:
    """Return one hash for all 90-degree rotations/reflections of a tile."""

    hashes: list[str] = []
    for rotation in range(4):
        rotated = np.rot90(array, rotation)
        hashes.append(array_hash(rotated))
        hashes.append(array_hash(np.fliplr(rotated)))
    return min(hashes)


def parse_sample_path(path: str) -> tuple[str, str, int, int]:
    """Parse ``class/layout_tile_x_y.npy`` without guessing layout underscores."""

    parts = PurePosixPath(path).parts
    if len(parts) != 2 or parts[0] not in CLASS_NAMES:
        raise ValueError(f"Expected clean/dirty sample path, got {path!r}")
    class_name, filename = parts
    if not filename.endswith(".npy") or "_tile_" not in filename:
        raise ValueError(f"Expected <layout>_tile_<x>_<y>.npy, got {path!r}")
    layout, coordinates = filename[:-4].rsplit("_tile_", 1)
    try:
        x_text, y_text = coordinates.split("_", 1)
        x, y = int(x_text), int(y_text)
    except ValueError as error:
        raise ValueError(f"Invalid tile coordinates in {path!r}") from error
    return class_name, layout, x, y


def load_catalog(path: Path, protocols_path: Path | None = None) -> dict[str, Any]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported catalog schema {catalog.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    layouts = catalog.get("layouts")
    if isinstance(layouts, list):
        source_collection = catalog.get("source_collection", {})
        source_repository = source_collection.get("repository")
        source_revision = source_collection.get("revision")
        source_license = source_collection.get("license")
        normalized_layouts: dict[str, dict[str, Any]] = {}
        for item in layouts:
            metadata = dict(item)
            name = metadata.pop("name")
            repository = metadata.get("source_repository", source_repository)
            revision = metadata.get("source_revision", source_revision)
            source_path = metadata.get("source_path")
            if repository and revision and source_path:
                metadata.setdefault(
                    "source_url",
                    f"https://github.com/{repository}/blob/{revision}/{source_path}",
                )
            metadata.setdefault("license", source_license)
            normalized_layouts[name] = metadata
        layouts = normalized_layouts
        catalog["layouts"] = layouts
    if not isinstance(layouts, dict) or not layouts:
        raise ValueError("Catalog must define non-empty layout metadata")

    if protocols_path is not None:
        protocol_config = json.loads(protocols_path.read_text(encoding="utf-8"))
        catalog["protocol_config"] = protocol_config
        grouped = protocol_config["protocols"]["unseen_layout_v1"]
        family_to_split = {
            family: split
            for split in SPLIT_NAMES
            for family in grouped[f"{split}_families"]
        }
        for name, metadata in layouts.items():
            family = metadata.get("family")
            if family not in family_to_split:
                if metadata.get("include_in_b1", True):
                    raise ValueError(
                        f"Layout family {family!r} ({name}) is missing from unseen_layout_v1"
                    )
                metadata["split"] = "train"
            else:
                metadata["split"] = family_to_split[family]
        tile_reference = protocol_config["protocols"]["tile_random_reference"]
        catalog["paper_aligned_ratios"] = tile_reference["fractions"]
    for name, metadata in layouts.items():
        for field in ("family", "circuit_type", "license", "split"):
            if not metadata.get(field):
                raise ValueError(f"Catalog layout {name!r} is missing {field!r}")
        if metadata.get("include_in_b1", True) and not metadata.get("source_url"):
            raise ValueError(f"Catalog layout {name!r} is missing 'source_url'")
        if metadata["split"] not in SPLIT_NAMES:
            raise ValueError(f"Catalog layout {name!r} has invalid split {metadata['split']!r}")
    return catalog


def _stratified_group_split(
    records: list[dict[str, Any]], seed: int, ratios: dict[str, float]
) -> dict[str, list[str]]:
    """Stratify canonical-content groups without changing class prevalence."""

    if set(ratios) != set(SPLIT_NAMES) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must define train/validation/test and sum to 1")
    rng = random.Random(seed)
    splits = {name: [] for name in SPLIT_NAMES}
    by_label_and_hash: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_label_and_hash[record["label"]][record["canonical_hash"]].append(record)
    for label in CLASS_NAMES:
        groups = [
            sorted(group, key=lambda record: record["path"])
            for _, group in sorted(by_label_and_hash[label].items())
        ]
        rng.shuffle(groups)
        total = sum(len(group) for group in groups)
        targets = {name: total * ratios[name] for name in SPLIT_NAMES}
        assigned = {name: 0 for name in SPLIT_NAMES}
        if len(groups) < len(SPLIT_NAMES):
            raise ValueError(f"Not enough independent {label} groups for a three-way split")
        for split_name, group in zip(("test", "validation", "train"), groups[:3]):
            splits[split_name].extend(record["path"] for record in group)
            assigned[split_name] += len(group)
        for group in groups[3:]:
            split_name = max(
                SPLIT_NAMES,
                key=lambda name: (targets[name] - assigned[name], ratios[name]),
            )
            splits[split_name].extend(record["path"] for record in group)
            assigned[split_name] += len(group)
        if any(assigned[name] == 0 for name in SPLIT_NAMES):
            raise ValueError(f"Not enough {label} records for a three-way split")
    return {name: sorted(paths) for name, paths in splits.items()}


def _grouped_split(
    records: list[dict[str, Any]], layout_catalog: dict[str, dict[str, Any]]
) -> tuple[dict[str, list[str]], int]:
    """Keep one canonical tile, preferring held-out layouts over train copies."""

    by_canonical_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_canonical_hash[record["canonical_hash"]].append(record)

    splits = {name: [] for name in SPLIT_NAMES}
    removed = 0
    for group in by_canonical_hash.values():
        chosen_split = max(
            (layout_catalog[record["source_layout"]]["split"] for record in group),
            key=lambda split_name: SPLIT_PRIORITY[split_name],
        )
        candidates = [
            record
            for record in group
            if layout_catalog[record["source_layout"]]["split"] == chosen_split
        ]
        retained = min(candidates, key=lambda record: record["path"])
        splits[chosen_split].append(retained["path"])
        removed += len(group) - 1
    return {name: sorted(paths) for name, paths in splits.items()}, removed


def _count_paths(
    records_by_path: dict[str, dict[str, Any]], paths: Iterable[str]
) -> dict[str, Any]:
    selected = [records_by_path[path] for path in paths]
    return {
        "samples": len(selected),
        "classes": dict(sorted(Counter(record["label"] for record in selected).items())),
        "layouts": dict(sorted(Counter(record["source_layout"] for record in selected).items())),
        "families": sorted({record["layout_family"] for record in selected}),
    }


def _assert_protocol_integrity(
    records_by_path: dict[str, dict[str, Any]], protocol: dict[str, Any], grouped: bool
) -> None:
    split_sets = {name: set(protocol["splits"][name]) for name in SPLIT_NAMES}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(f"Sample paths cross {left}/{right}: {sorted(overlap)[:3]}")

    canonical_splits: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, paths in split_sets.items():
        for path in paths:
            record = records_by_path[path]
            canonical_splits[record["canonical_hash"]].add(split_name)
            family_splits[record["layout_family"]].add(split_name)
    leaked_hashes = [key for key, values in canonical_splits.items() if len(values) > 1]
    if leaked_hashes:
        raise ValueError(f"Canonical content hashes cross splits: {leaked_hashes[:3]}")
    if grouped:
        leaked_families = [key for key, values in family_splits.items() if len(values) > 1]
        if leaked_families:
            raise ValueError(f"Layout families cross grouped splits: {leaked_families}")


def build_manifest(
    dataset_zip: Path,
    catalog_path: Path,
    seed: int = 42,
    protocols_path: Path | None = None,
) -> dict[str, Any]:
    dataset_zip = dataset_zip.resolve()
    catalog_path = catalog_path.resolve()
    protocols_path = protocols_path.resolve() if protocols_path is not None else None
    catalog = load_catalog(catalog_path, protocols_path)
    layout_catalog: dict[str, dict[str, Any]] = catalog["layouts"]
    records: list[dict[str, Any]] = []

    with zipfile.ZipFile(dataset_zip) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".npy"))
        if not names:
            raise ValueError(f"No .npy samples found in {dataset_zip}")
        for path in names:
            label, source_layout, x, y = parse_sample_path(path)
            if source_layout not in layout_catalog:
                raise ValueError(
                    f"Dataset layout {source_layout!r} is absent from {catalog_path}"
                )
            array = np.load(BytesIO(archive.read(path)), allow_pickle=False)
            if array.ndim != 2:
                raise ValueError(f"Expected a 2-D tile at {path}, got shape {array.shape}")
            metadata = layout_catalog[source_layout]
            records.append(
                {
                    "path": path,
                    "label": label,
                    "label_index": CLASS_NAMES.index(label),
                    "source_layout": source_layout,
                    "layout_family": metadata["family"],
                    "variant_type": metadata.get("variant_type", "source"),
                    "derived_from": metadata.get("derived_from"),
                    "coordinates_dbu": [x, y],
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "metal_density": float(np.count_nonzero(array) / array.size),
                    "content_hash": array_hash(array),
                    "canonical_hash": canonical_manhattan_hash(array),
                }
            )

    exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    canonical_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        exact_groups[record["content_hash"]].append(record)
        canonical_groups[record["canonical_hash"]].append(record)

    exact_conflicts = [
        group for group in exact_groups.values() if len({record["label"] for record in group}) > 1
    ]
    canonical_conflicts = [
        group
        for group in canonical_groups.values()
        if len({record["label"] for record in group}) > 1
    ]
    if exact_conflicts or canonical_conflicts:
        examples = [record["path"] for group in (exact_conflicts or canonical_conflicts)[:3] for record in group]
        raise ValueError(f"Contradictory clean/dirty content labels detected: {examples[:8]}")

    eligible_records = [
        record
        for record in records
        if layout_catalog[record["source_layout"]].get("include_in_b1", True)
    ]
    ratios = catalog.get(
        "paper_aligned_ratios", {"train": 0.80, "validation": 0.15, "test": 0.05}
    )
    paper_splits = _stratified_group_split(eligible_records, seed, ratios)
    grouped_splits, grouped_removed = _grouped_split(eligible_records, layout_catalog)
    records_by_path = {record["path"]: record for record in records}

    grouped_missing_splits = [name for name, paths in grouped_splits.items() if not paths]
    protocols = {
        "tile_random_reference": {
            "status": catalog.get("paper_aligned_status", "candidate"),
            "description": (
                "Stratified tile-level split with each Manhattan-equivalent group kept "
                "entirely in one split. This is a B0-compatible reference, not a direct "
                "reproduction of the paper."
            ),
            "seed": seed,
            "ratios": ratios,
            "deduplication": "Manhattan-equivalent groups remain intact within one split",
            "removed_duplicate_samples": 0,
            "splits": paper_splits,
        },
        "unseen_layout_v1": {
            "status": (
                "blocked-missing-layout-data"
                if grouped_missing_splits
                else catalog.get("grouped_split_status", "candidate")
            ),
            "description": (
                "Layout-family-disjoint split with one representative per "
                "Manhattan-equivalent content group."
            ),
            "seed": seed,
            "deduplication": (
                "test > validation > train for cross-split canonical groups; one "
                "deterministic representative is retained"
            ),
            "removed_duplicate_samples": grouped_removed,
            "missing_splits": grouped_missing_splits,
            "splits": grouped_splits,
        },
    }
    for protocol_name, protocol in protocols.items():
        grouped = protocol_name == "unseen_layout_v1"
        _assert_protocol_integrity(records_by_path, protocol, grouped=grouped)
        protocol["summary"] = {
            split: _count_paths(records_by_path, protocol["splits"][split])
            for split in SPLIT_NAMES
        }

    inventory: dict[str, Any] = {}
    for layout_name in sorted(layout_catalog):
        layout_records = [record for record in records if record["source_layout"] == layout_name]
        if not layout_records:
            continue
        inventory[layout_name] = {
            "family": layout_catalog[layout_name]["family"],
            "circuit_type": layout_catalog[layout_name]["circuit_type"],
            "samples": len(layout_records),
            "classes": dict(sorted(Counter(record["label"] for record in layout_records).items())),
            "density": {
                "minimum": min(record["metal_density"] for record in layout_records),
                "mean": sum(record["metal_density"] for record in layout_records) / len(layout_records),
                "maximum": max(record["metal_density"] for record in layout_records),
            },
        }

    project_root = catalog_path.parent.parent
    try:
        dataset_display_path = dataset_zip.relative_to(project_root).as_posix()
    except ValueError:
        dataset_display_path = str(dataset_zip)

    provenance_inventory: dict[str, Any] = {}
    for name, metadata in sorted(layout_catalog.items()):
        local_path = project_root / "real_layouts_tt" / f"{name}.oas"
        actual_hash = sha256_file(local_path) if local_path.exists() else None
        expected_hash = metadata.get("expected_sha256")
        provenance_inventory[name] = {
            "present": local_path.exists(),
            "actual_sha256": actual_hash,
            "expected_sha256": expected_hash,
            "verified": bool(actual_hash and expected_hash and actual_hash == expected_hash),
            "status": metadata.get("provenance_status"),
        }
    selected_not_acquired = sorted(
        name
        for name, metadata in layout_catalog.items()
        if metadata.get("include_in_b1", True) and name not in inventory
    )
    unresolved_provenance = sorted(
        name
        for name, metadata in layout_catalog.items()
        if name in inventory
        and metadata.get("include_in_b1", True)
        and not provenance_inventory[name]["verified"]
    )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "path": dataset_display_path,
            "archive_sha256": sha256_file(dataset_zip),
            "samples": len(records),
            "classes": dict(sorted(Counter(record["label"] for record in records).items())),
            "layout_count": len(inventory),
            "family_count": len({item["family"] for item in inventory.values()}),
            "eligible_samples": len(eligible_records),
        },
        "generation": catalog.get("generation", {}),
        "layouts": layout_catalog,
        "inventory": inventory,
        "audit": {
            "exact_unique": len(exact_groups),
            "exact_duplicate_groups": sum(len(group) > 1 for group in exact_groups.values()),
            "exact_duplicate_samples": sum(len(group) - 1 for group in exact_groups.values()),
            "manhattan_unique": len(canonical_groups),
            "manhattan_duplicate_groups": sum(
                len(group) > 1 for group in canonical_groups.values()
            ),
            "manhattan_duplicate_samples": sum(
                len(group) - 1 for group in canonical_groups.values()
            ),
            "exact_label_conflicts": 0,
            "manhattan_label_conflicts": 0,
            "unresolved_provenance": unresolved_provenance,
            "selected_layouts_not_acquired": selected_not_acquired,
            "b1_gate_ready": not (
                exact_conflicts
                or canonical_conflicts
                or grouped_missing_splits
                or unresolved_provenance
                or selected_not_acquired
            ),
        },
        "provenance": provenance_inventory,
        "protocols": protocols,
        "records": records,
    }
    identity_payload = {
        "archive_sha256": manifest["dataset"]["archive_sha256"],
        "catalog": layout_catalog,
        "protocols": protocols,
        "records": records,
    }
    manifest["manifest_id"] = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return manifest


def manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "manifest_id": manifest["manifest_id"],
        "dataset": manifest["dataset"],
        "inventory": manifest["inventory"],
        "audit": manifest["audit"],
        "provenance": manifest["provenance"],
        "protocols": {
            name: {
                key: value
                for key, value in protocol.items()
                if key not in {"splits"}
            }
            for name, protocol in manifest["protocols"].items()
        },
    }
