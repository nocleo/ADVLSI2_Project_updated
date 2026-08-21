#!/usr/bin/env python3
"""Plan and execute the preregistered B8.0 ORFS actionability pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.b8_actionability import (
    atomic_write_json,
    execute_one,
    frozen_protocol,
    initialize_manifest,
    pending_specs,
    summarize_manifest,
)


DEFAULT_OUTPUT = Path(
    "/content/drive/MyDrive/ADVLSI2 2026 Project/experiments/"
    "B8_action_control/b8_0_actionability"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("plan", "run", "status"))
    result.add_argument("--stage", choices=("smoke", "matrix"), default="smoke")
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--orfs-root", type=Path)
    result.add_argument("--container-image", default="openroad/orfs")
    result.add_argument("--container-digest")
    result.add_argument(
        "--deck", type=Path, default=PROJECT_ROOT / "sky130_drc_deck" / "sky130A_mr.drc"
    )
    result.add_argument("--executor", choices=("native", "docker-shell"), default="docker-shell")
    result.add_argument("--threads", type=int, default=8)
    result.add_argument("--timeout-seconds", type=int, default=4 * 60 * 60)
    result.add_argument("--protocol-hash")
    result.add_argument("--max-runs", type=int)
    result.add_argument("--rerun-failed", action="store_true")
    return result


def _resolve_protocol(args: argparse.Namespace) -> dict:
    if args.protocol_hash:
        manifest_path = args.output_root / args.protocol_hash / "manifest.json"
        return json.loads(manifest_path.read_text(encoding="utf-8"))["protocol"]
    if args.orfs_root is None or args.container_digest is None:
        raise SystemExit(
            "--orfs-root and immutable --container-digest sha256:... are required "
            "when creating a protocol"
        )
    return frozen_protocol(
        orfs_root=args.orfs_root,
        container_image=args.container_image,
        container_digest=args.container_digest,
        deck_path=args.deck,
        threads=args.threads,
        timeout_seconds=args.timeout_seconds,
        executor=args.executor,
    )


def main() -> None:
    args = parser().parse_args()
    protocol = _resolve_protocol(args)
    manifest = initialize_manifest(args.output_root, protocol, args.stage)
    protocol_root = args.output_root / protocol["protocol_hash"]
    if args.command == "plan":
        print(json.dumps(summarize_manifest(manifest, args.stage), indent=2))
        print(f"Manifest: {protocol_root / 'manifest.json'}")
        return
    if args.command == "status":
        print(json.dumps(summarize_manifest(manifest, args.stage), indent=2))
        return
    specs = pending_specs(manifest, args.stage, args.rerun_failed, protocol_root)
    if args.max_runs is not None:
        specs = specs[: args.max_runs]
    for index, spec in enumerate(specs, 1):
        print(f"[B8.0] {index}/{len(specs)} {spec.run_id}", flush=True)
        record = execute_one(
            output_root=args.output_root,
            protocol=protocol,
            spec=spec,
            rerun_failed=args.rerun_failed,
        )
        print(f"[B8.0] {spec.run_id}: {record['status']}", flush=True)
    manifest = json.loads((protocol_root / "manifest.json").read_text(encoding="utf-8"))
    summary = summarize_manifest(manifest, args.stage)
    atomic_write_json(protocol_root / f"{args.stage}_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
