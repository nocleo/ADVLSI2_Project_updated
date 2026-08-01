"""Create a deterministic clean/dirty visual audit grid from a B1 manifest."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import klayout.db as db

from project_paths import drc_mask_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CELL_WIDTH = 360
CELL_HEIGHT = 250
TILE_SIZE = 190
MARGIN = 20
BACKGROUND = "#111827"
TEXT = "#F9FAFB"
MUTED = "#9CA3AF"
LABEL_COLORS = {"clean": "#2DD4BF", "dirty": "#FB7185"}
MASK_LAYER = (255, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "training_datasets" / "combined_training_dataset.zip",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "b1_current_audit" / "manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "b1_current_audit" / "label_audit_grid.png",
    )
    return parser.parse_args()


def select_examples(records: list[dict]) -> list[dict]:
    """Select the median-density clean and dirty tile for every layout."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["source_layout"], record["label"])].append(record)

    selected = []
    layouts = sorted({record["source_layout"] for record in records})
    for layout in layouts:
        for label in ("clean", "dirty"):
            candidates = sorted(
                grouped[(layout, label)],
                key=lambda record: (record["metal_density"], record["path"]),
            )
            if not candidates:
                raise ValueError(f"No {label} sample exists for {layout}")
            selected.append(candidates[len(candidates) // 2])
    return selected


def mask_overlay(record: dict, cache: dict[str, db.Region]) -> Image.Image:
    """Rasterize verified DRC mask geometry over one selected dirty tile."""
    overlay = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    if record["label"] != "dirty":
        return overlay
    layout_name = record["source_layout"]
    if layout_name not in cache:
        layout = db.Layout()
        layout.read(str(drc_mask_path(layout_name)))
        layer_index = layout.find_layer(*MASK_LAYER)
        cache[layout_name] = db.Region(
            layout.top_cell().begin_shapes_rec(layer_index)
        )
    x_origin, y_origin = record["coordinates_dbu"]
    tile_box = db.Box(x_origin, y_origin, x_origin + 1600, y_origin + 1600)
    region = cache[layout_name] & db.Region(tile_box)
    draw = ImageDraw.Draw(overlay)
    scale = 200 / 1600
    for polygon in region.each():
        points = [
            (
                round((point.x - x_origin) * scale),
                round(200 - (point.y - y_origin) * scale),
            )
            for point in polygon.each_point_hull()
        ]
        if len(points) >= 3:
            draw.polygon(points, fill=(251, 113, 133, 190))
    return overlay


def render(dataset: Path, manifest_path: Path, output: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = select_examples(manifest["records"])
    columns = 4
    rows = (len(selected) + columns - 1) // columns
    header_height = 90
    canvas = Image.new(
        "RGB", (columns * CELL_WIDTH, header_height + rows * CELL_HEIGHT), BACKGROUND
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((MARGIN, 18), "B1 deterministic label audit", fill=TEXT, font=font)
    draw.text(
        (MARGIN, 44),
        f"{manifest['dataset']['samples']:,} tiles | manifest {manifest['manifest_id'][:12]}",
        fill=MUTED,
        font=font,
    )

    mask_cache: dict[str, db.Region] = {}
    with zipfile.ZipFile(dataset) as archive:
        for index, record in enumerate(selected):
            column = index % columns
            row = index // columns
            left = column * CELL_WIDTH + MARGIN
            top = header_height + row * CELL_HEIGHT + MARGIN
            array = np.load(BytesIO(archive.read(record["path"])), allow_pickle=False)
            pixels = (np.asarray(array, dtype=np.uint8) * 255)
            tile = Image.fromarray(pixels, mode="L").convert("RGBA")
            tile = Image.alpha_composite(tile, mask_overlay(record, mask_cache))
            tile = tile.resize(
                (TILE_SIZE, TILE_SIZE), Image.Resampling.NEAREST
            )
            border = LABEL_COLORS[record["label"]]
            draw.rectangle(
                (left - 2, top - 2, left + TILE_SIZE + 1, top + TILE_SIZE + 1),
                outline=border,
                width=2,
            )
            canvas.paste(tile.convert("RGB"), (left, top))
            text_x = left + TILE_SIZE + 14
            label_text = record["label"].upper()
            if record["label"] == "dirty":
                label_text += " + DRC MASK"
            draw.text((text_x, top + 4), label_text, fill=border, font=font)
            layout = record["source_layout"]
            if len(layout) > 22:
                layout = layout[:20] + "..."
            draw.text((text_x, top + 29), layout, fill=TEXT, font=font)
            draw.text(
                (text_x, top + 52),
                f"density {record['metal_density']:.3f}",
                fill=MUTED,
                font=font,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    print(output)


def main() -> None:
    args = parse_args()
    render(args.dataset, args.manifest, args.output)


if __name__ == "__main__":
    main()
