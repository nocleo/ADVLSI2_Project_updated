"""Resume-safe B6.1 build across the registered B1 layout families."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import (  # noqa: E402
    add_generate_training_dataset_scripts_to_syspath,
    dataset_output_dir,
    drc_report_path,
    ensure_dir,
    extracted_m1_gds,
    injected_m1_gds,
    layout_oas,
)

add_generate_training_dataset_scripts_to_syspath()

from extract_m1 import extract_m1  # noqa: E402
from generate_localization_dataset import (  # noqa: E402
    LocalizationConfig,
    build_layout_dataset,
)
from inject_drc_error import inject_isolated_m1_2_errors  # noqa: E402
from run_full_drc import run_full_drc  # noqa: E402


DEFAULT_REGISTRY = PROJECT_ROOT / "data" / "layout_registry.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "training_datasets" / "b6_localization_dataset"
DEFAULT_RESULT = PROJECT_ROOT / "results" / "b6_localization_dataset" / "summary.json"


def layout_seed(base_seed: int, layout_name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{layout_name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def registered_layouts(registry_path: Path) -> list[str]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    return [
        item["name"]
        for item in registry["layouts"]
        if item.get("include_in_b1", False)
    ]


def ensure_layout_intermediates(layout_name: str, seed: int) -> tuple[Path, Path]:
    injected = injected_m1_gds(layout_name)
    report = drc_report_path(layout_name)
    if not injected.exists():
        source = layout_oas(layout_name)
        if not source.exists():
            raise FileNotFoundError(source)
        output_dir = ensure_dir(dataset_output_dir(layout_name))
        extract_m1(source, output_dir)
        inject_isolated_m1_2_errors(
            extracted_m1_gds(layout_name),
            num_errors=400,
            seed=layout_seed(seed, layout_name),
        )
    if not injected.exists():
        raise RuntimeError(f"Injection did not create {injected}")
    if not report.exists() and not run_full_drc(injected, report):
        raise RuntimeError(f"DRC did not create {report}")
    return injected, report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_reproducible_archive(root: Path) -> Path:
    archive = root.with_suffix(".zip")
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as output:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            output.writestr(info, path.read_bytes(), compresslevel=9)
    return archive


def _write_compact_result(summary: dict, result_path: Path) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    compact = {
        key: summary[key]
        for key in (
            "schema_version",
            "phase",
            "status",
            "config",
            "layout_count",
            "exact_violation_count",
            "unique_owner_count",
            "dirty_tile_count",
            "clean_tile_count",
            "total_tile_count",
            "surrogate_target_count",
            "omitted_nonowner_subpixel_fragment_count",
            "visual_audit_categories",
            "archive_sha256",
        )
    }
    compact["layouts"] = [
        {
            key: layout[key]
            for key in (
                "layout",
                "exact_violation_count",
                "unique_owner_count",
                "dirty_tile_count",
                "clean_tile_count",
                "surrogate_target_count",
                "omitted_nonowner_subpixel_fragment_count",
            )
        }
        for layout in summary["layouts"]
    ]
    result_path.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_registry(
    registry_path: Path = DEFAULT_REGISTRY,
    output_root: Path = DEFAULT_OUTPUT,
    result_path: Path = DEFAULT_RESULT,
    seed: int = 42,
    force: bool = False,
    make_archive: bool = True,
) -> dict:
    config = LocalizationConfig(seed=seed)
    layout_summaries = []
    for layout_name in registered_layouts(registry_path):
        print(f"[B6.1] {layout_name}", flush=True)
        injected, report = ensure_layout_intermediates(layout_name, seed)
        layout_summaries.append(
            build_layout_dataset(
                layout_name,
                injected,
                output_root / "layouts" / layout_name,
                config,
                force=force,
                report_path=report,
            )
        )
    exact_count = sum(item["exact_violation_count"] for item in layout_summaries)
    owner_count = sum(item["unique_owner_count"] for item in layout_summaries)
    dirty_count = sum(item["dirty_tile_count"] for item in layout_summaries)
    clean_count = sum(item["clean_tile_count"] for item in layout_summaries)
    if exact_count != owner_count:
        raise RuntimeError(f"Unique ownership failed: {owner_count}/{exact_count}")
    categories = sorted(
        set.intersection(*(set(item["visual_audit"]) for item in layout_summaries))
    )
    required = {"clean", "dense", "sparse", "horizontal", "vertical", "boundary", "near_threshold"}
    if not required.issubset(categories):
        raise RuntimeError(f"Missing visual-audit categories: {sorted(required - set(categories))}")

    summary = {
        "schema_version": 1,
        "phase": "B6.1",
        "status": "complete",
        "config": {
            **config.__dict__,
            "nm_per_pixel": config.nm_per_pixel,
        },
        "layout_count": len(layout_summaries),
        "exact_violation_count": exact_count,
        "unique_owner_count": owner_count,
        "dirty_tile_count": dirty_count,
        "clean_tile_count": clean_count,
        "total_tile_count": dirty_count + clean_count,
        "surrogate_target_count": sum(item["surrogate_target_count"] for item in layout_summaries),
        "omitted_nonowner_subpixel_fragment_count": sum(
            item["omitted_nonowner_subpixel_fragment_count"] for item in layout_summaries
        ),
        "visual_audit_categories": categories,
        "layouts": layout_summaries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "index.json"
    index_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if make_archive:
        archive = make_reproducible_archive(output_root)
        summary["archive"] = str(archive)
        summary["archive_sha256"] = _sha256(archive)
    else:
        summary["archive_sha256"] = None
    index_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_compact_result(summary, result_path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args(argv)
    summary = build_registry(
        args.registry,
        args.output,
        args.result,
        args.seed,
        args.force,
        not args.no_archive,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
