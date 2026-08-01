"""Acquire and verify the additional B1 layout sources.

The registry pins every source to a repository revision and SHA-256 digest.
Existing layouts are left untouched unless ``--include-existing`` is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_bytes(checkout: Path, revision: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(checkout), "show", f"{revision}:{path}"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout


def acquire(args: argparse.Namespace) -> dict[str, object]:
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    source = registry["source_collection"]
    selected_statuses = {"selected_for_b1"}
    if args.include_existing:
        selected_statuses.add("existing")

    layouts = [
        layout
        for layout in registry["layouts"]
        if layout["status"] in selected_statuses and layout.get("source_path")
    ]
    args.destination.mkdir(parents=True, exist_ok=True)

    repositories = {
        layout.get("source_repository", source["repository"]) for layout in layouts
    }
    checkouts: dict[str, Path] = {}
    results: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="advlsi-b1-layouts-") as temp_dir:
        temp_root = Path(temp_dir)
        for index, repository in enumerate(sorted(repositories)):
            checkout = temp_root / f"source-{index}"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    f"https://github.com/{repository}.git",
                    str(checkout),
                ],
                check=True,
            )
            checkouts[repository] = checkout

        for layout in layouts:
            repository = layout.get("source_repository", source["repository"])
            revision = layout.get("source_revision", source["revision"])
            checkout = checkouts[repository]
            subprocess.run(
                ["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", revision],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            payload = git_bytes(checkout, revision, layout["source_path"])
            actual_hash = sha256(payload)
            if actual_hash != layout["expected_sha256"]:
                raise RuntimeError(
                    f"Source hash mismatch for {layout['name']}: "
                    f"expected {layout['expected_sha256']}, got {actual_hash}"
                )

            license_path = source["license_path_template"].format(name=layout["name"])
            license_payload = git_bytes(checkout, revision, license_path)
            if b"Apache License" not in license_payload:
                raise RuntimeError(f"Unexpected license for {layout['name']}: {license_path}")

            destination = args.destination / f"{layout['name']}.oas"
            action = "created"
            if destination.exists():
                existing_hash = sha256(destination.read_bytes())
                if existing_hash == actual_hash:
                    action = "already_verified"
                elif not args.replace_mismatched:
                    raise RuntimeError(
                        f"Refusing to replace mismatched {destination}. "
                        "Pass --replace-mismatched after reviewing the registry."
                    )
                else:
                    action = "replaced"
            if action in {"created", "replaced"}:
                destination.write_bytes(payload)
            results.append(
                {
                    "name": layout["name"],
                    "action": action,
                    "sha256": actual_hash,
                    "bytes": len(payload),
                    "repository": repository,
                    "revision": revision,
                    "license": source["license"],
                }
            )

    summary = {"layouts": results, "count": len(results)}
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "data" / "layout_registry.json",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=PROJECT_ROOT / "real_layouts_tt",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Also verify/acquire layouts marked existing.",
    )
    parser.add_argument(
        "--replace-mismatched",
        action="store_true",
        help="Replace an existing layout only after its pinned source hash is verified.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    acquire(parse_args())
