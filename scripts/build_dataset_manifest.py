"""Build the B1 integrity manifest and versioned evaluation protocols."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_manifest import build_manifest, manifest_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "training_datasets" / "combined_training_dataset.zip",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "data" / "layout_registry.json",
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation_protocols.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "b1_current_audit" / "manifest.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        args.dataset,
        args.registry,
        seed=args.seed,
        protocols_path=args.protocol_config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary_path = args.output.with_name(f"{args.output.stem}.summary.json")
    summary = manifest_summary(manifest)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"manifest": str(args.output), "summary": str(summary_path), **summary},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
