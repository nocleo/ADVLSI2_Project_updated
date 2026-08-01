import os
import shutil
import time
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import (
    LAYOUTS_DIR,
    COMBINED_DATASET_DIR,
    layout_oas,
    layout_output_dir,
    dataset_output_dir,
    training_dataset_dir,
    extracted_m1_gds,
    injected_m1_gds,
    drc_report_path,
    drc_mask_path,
    ensure_dir,
)

# Import core functions from your scripts
from run_full_drc import run_full_drc
from extract_m1 import extract_m1
from extract_drc_mask_from_rdb import extract_drc_mask_from_rdb
from inject_drc_error import inject_isolated_m1_2_errors
from generate_training_dataset import generate_dataset
from visualize_dataset_matrices import visualize_dataset_matrices

# --- CONFIGURATION ---
DEFAULT_LAYOUT_NAME = "tt_um_cmos_inverter"
ERROR_COUNT = 400
DEFAULT_SEED = 42


def build_paths(layout_name):
    base_out_dir = layout_output_dir(layout_name)
    dataset_output = dataset_output_dir(layout_name)
    training_dataset = training_dataset_dir(layout_name)
    return dict(
        base_out_dir=str(base_out_dir),
        dataset_output=str(dataset_output),
        training_dataset=str(training_dataset),
        original_gds=str(layout_oas(layout_name)),
        extracted_m1_gds=str(extracted_m1_gds(layout_name)),
        injected_gds=str(injected_m1_gds(layout_name)),
        drc_report=str(drc_report_path(layout_name)),
        mask_file=str(drc_mask_path(layout_name)),
    )


def clean_layout_folder(paths):
    """Delete old data to prevent dataset contamination, then recreate folders."""
    if os.path.exists(paths["base_out_dir"]):
        print(f"[*] Cleaning up old '{paths['base_out_dir']}' directory...")
        try:
            shutil.rmtree(paths["base_out_dir"])
        except OSError as e:
            print(f"[*] Warning: Could not fully remove '{paths['base_out_dir']}' ({e}). Will try to continue.")
    os.makedirs(paths["dataset_output"], exist_ok=True)
    os.makedirs(paths["training_dataset"], exist_ok=True)
    time.sleep(0.5)


def validate_generated_tiles(training_dir, tile_prefix, expected_dirty, expected_clean):
    """Fail when generated files do not exactly match the reported build."""
    root = Path(training_dir)
    expected = {"dirty": expected_dirty, "clean": expected_clean}
    for label, expected_count in expected.items():
        files = sorted((root / label).glob("*.npy"))
        if len(files) != expected_count:
            raise RuntimeError(
                f"{root / label} contains {len(files)} tiles, but generation "
                f"reported {expected_count}"
            )
        if tile_prefix:
            invalid = [path.name for path in files if not path.name.startswith(tile_prefix)]
            if invalid:
                raise RuntimeError(
                    f"{root / label} contains tiles without prefix {tile_prefix!r}: "
                    + ", ".join(invalid[:5])
                )


def layout_seed(base_seed, layout_name):
    """Derive a stable independent seed for one source layout."""
    digest = hashlib.sha256(f"{base_seed}:{layout_name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def run_single_layout(layout_name, tile_prefix="", seed=DEFAULT_SEED):
    """Run the full DRC pipeline for one layout. Returns (v_count, c_count) or None on failure."""
    paths = build_paths(layout_name)

    print("=" * 60)
    print(f" STARTING FULL AUTONOMOUS DRC PIPELINE FOR: {layout_name}")
    print("=" * 60)

    clean_layout_folder(paths)

    print("\n>>> STEP 1: Extracting M1 (Layer 68/20) from original layout <<<")
    extract_m1(paths["original_gds"], paths["dataset_output"])

    print("\n>>> STEP 2: Injecting NEW DRC Errors into the M1 layout <<<")
    inject_isolated_m1_2_errors(
        paths["extracted_m1_gds"],
        num_errors=ERROR_COUNT,
        seed=layout_seed(seed, layout_name),
    )

    print(f"\n>>> STEP 3: Running KLayout DRC Engine on {paths['injected_gds']} <<<")
    drc_success = run_full_drc(paths["injected_gds"], paths["drc_report"])
    if not drc_success:
        print(f"Pipeline aborted for {layout_name} due to DRC failure.")
        return None

    print("\n>>> STEP 4: Extracting verified violations from RDB to Mask <<<")
    total_drc_errors_found = extract_drc_mask_from_rdb(paths["drc_report"], paths["mask_file"])

    print("\n>>> STEP 5: Generating Dataset Tiles (Matrix conversion) <<<")
    v_count, c_count = generate_dataset(
        paths["injected_gds"],
        paths["training_dataset"],
        tile_prefix=tile_prefix,
        seed=layout_seed(seed, f"{layout_name}:clean-sampling"),
    )
    validate_generated_tiles(
        paths["training_dataset"], tile_prefix, v_count, c_count
    )

    print("\n>>> STEP 6: Visualizing Results (Sanity Check) <<<")
    visualize_dataset_matrices(
        f"{paths['training_dataset']}/clean",
        f"{paths['training_dataset']}/dirty",
    )

    print("\n" + "=" * 60)
    print(f" PIPELINE COMPLETED: {layout_name}")
    print(f"   * Layout Errors Injected:    {ERROR_COUNT}")
    print(f"   * Layout Errors Found (DRC): {total_drc_errors_found}")
    print(f"   * Violation Images (Dirty):  {v_count}")
    print(f"   * Clean Images Generated:    {c_count}")
    print(f"   * Total Training Samples:    {v_count + c_count}")
    print("=" * 60)

    return v_count, c_count, paths["training_dataset"]


def registry_layout_names(registry_path):
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    return [
        layout["name"]
        for layout in registry["layouts"]
        if layout.get("include_in_b1", False)
    ]


def combine_layout_datasets(layout_names):
    """Rebuild the combined dataset from completed per-layout outputs."""
    if os.path.exists(COMBINED_DATASET_DIR):
        print("[*] Cleaning up old combined dataset directory...")
        shutil.rmtree(COMBINED_DATASET_DIR)
    ensure_dir(COMBINED_DATASET_DIR / "clean")
    ensure_dir(COMBINED_DATASET_DIR / "dirty")

    grand_v = 0
    grand_c = 0
    missing = []
    for layout_name in layout_names:
        training_dir = training_dataset_dir(layout_name)
        counts = {}
        for split in ("clean", "dirty"):
            src_dir = training_dir / split
            files = sorted(src_dir.glob("*.npy")) if src_dir.exists() else []
            counts[split] = len(files)
            if not files:
                missing.append(f"{layout_name}/{split}")
                continue
            dst_dir = COMBINED_DATASET_DIR / split
            expected_prefix = f"{layout_name}_"
            for npy_file in files:
                if not npy_file.name.startswith(expected_prefix):
                    raise RuntimeError(
                        f"Unexpected unprefixed tile for {layout_name}: {npy_file.name}"
                    )
                destination = dst_dir / npy_file.name
                if destination.exists():
                    raise FileExistsError(
                        f"Duplicate combined-dataset filename: {destination.name}"
                    )
                shutil.copy2(npy_file, destination)
        grand_c += counts["clean"]
        grand_v += counts["dirty"]

    if missing:
        raise FileNotFoundError(
            "Cannot combine incomplete per-layout datasets: " + ", ".join(missing)
        )

    combined_counts = {
        split: len(list((COMBINED_DATASET_DIR / split).glob("*.npy")))
        for split in ("clean", "dirty")
    }
    if combined_counts != {"clean": grand_c, "dirty": grand_v}:
        raise RuntimeError(
            f"Combined counts {combined_counts} do not match per-layout totals "
            f"{{'clean': {grand_c}, 'dirty': {grand_v}}}"
        )

    zip_path = shutil.make_archive(
        str(COMBINED_DATASET_DIR), "zip", root_dir=str(COMBINED_DATASET_DIR)
    )
    print("\n" + "=" * 60)
    print(" COMBINED DATASET SUMMARY")
    print("=" * 60)
    print(f"   * Layouts combined:          {len(layout_names)}")
    print(f"   * Total Violation (Dirty):   {grand_v}")
    print(f"   * Total Clean:               {grand_c}")
    print(f"   * Total Training Samples:    {grand_v + grand_c}")
    print(f"   * Combined archive:          {zip_path}")
    print("=" * 60)
    return grand_v, grand_c, zip_path


def run_all_layouts(seed=DEFAULT_SEED, registry_path=None):
    """Run the pipeline for every .oas layout in LAYOUTS_DIR and produce one combined zip."""
    oas_files = sorted(glob.glob(str(LAYOUTS_DIR / "*.oas")))
    if not oas_files:
        print(f"[!] No .oas files found in '{LAYOUTS_DIR}'. Aborting.")
        return

    available_names = {os.path.splitext(os.path.basename(f))[0] for f in oas_files}
    if registry_path:
        layout_names = registry_layout_names(registry_path)
        missing = sorted(set(layout_names) - available_names)
        if missing:
            raise FileNotFoundError(
                "B1 registry layouts have not been acquired: " + ", ".join(missing)
            )
    else:
        layout_names = sorted(available_names)
    print(f"[*] Found {len(layout_names)} layouts: {', '.join(layout_names)}")

    failed = []

    for layout_name in layout_names:
        tile_prefix = f"{layout_name}_"
        result = run_single_layout(layout_name, tile_prefix=tile_prefix, seed=seed)
        if result is None:
            failed.append(layout_name)
            continue

        _, _, _ = result

    completed = [name for name in layout_names if name not in failed]
    grand_v, grand_c, zip_path = combine_layout_datasets(completed)

    print("\n" + "=" * 60)
    print(" ALL LAYOUTS PROCESSED — COMBINED DATASET SUMMARY")
    print("=" * 60)
    print(f"   * Layouts processed:         {len(layout_names) - len(failed)}/{len(layout_names)}")
    if failed:
        print(f"   * Failed layouts:            {', '.join(failed)}")
    print(f"   * Total Violation (Dirty):   {grand_v}")
    print(f"   * Total Clean:               {grand_c}")
    print(f"   * Total Training Samples:    {grand_v + grand_c}")
    print(f"   * Combined archive:          {zip_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full DRC training-data pipeline."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--layout",
        metavar="NAME",
        help=f"Name of a single layout in {LAYOUTS_DIR}/ (without extension). "
             f"Defaults to '{DEFAULT_LAYOUT_NAME}'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base seed used to derive a deterministic injection seed per layout.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="With --all, process only include_in_b1 layouts from this registry.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help=f"Run the pipeline on every .oas file in {LAYOUTS_DIR}/ and "
             "produce one combined training dataset zip.",
    )
    group.add_argument(
        "--combine-only",
        action="store_true",
        help="Rebuild the combined ZIP from completed per-layout datasets.",
    )
    args = parser.parse_args()

    if args.combine_only:
        if not args.registry:
            parser.error("--combine-only requires --registry")
        combine_layout_datasets(registry_layout_names(args.registry))
    elif args.all:
        run_all_layouts(seed=args.seed, registry_path=args.registry)
    else:
        layout_name = args.layout if args.layout else DEFAULT_LAYOUT_NAME
        run_single_layout(
            layout_name,
            tile_prefix=f"{layout_name}_",
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
