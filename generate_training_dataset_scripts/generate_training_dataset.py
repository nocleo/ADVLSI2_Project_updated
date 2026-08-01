import klayout.db as db
import numpy as np
from PIL import Image, ImageDraw
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import layout_oas

METAL_LAYER = (68, 20)
MASK_LAYER = (255, 0)

# --- PHYSICAL & IMAGE RESOLUTION SETTINGS ---
#PHYSICAL_SIZE = 1600
PHYSICAL_SIZE = 1600
#STRIDE = 400 
STRIDE = 1500
#IMAGE_SIZE = 200  
IMAGE_SIZE = 200

#MARGIN = 600
MARGIN = 100
OUTPUT_DIR = "training_dataset"
DEFAULT_LAYOUT = "tt_um_yen"
INPUT_LAYOUT = layout_oas(DEFAULT_LAYOUT)


def _candidate_axis_indices(positions, geometry_min, geometry_max, margin):
    """Return grid indices whose inset window can intersect a geometry bbox."""
    if not positions:
        return range(0)
    low = geometry_min - (PHYSICAL_SIZE - margin)
    high = geometry_max - margin
    first = max(0, -((positions.start - low) // positions.step))
    last = min(len(positions) - 1, (high - positions.start) // positions.step)
    return range(first, last + 1) if first <= last else range(0)


def _mask_candidate_windows(mask_region, x_range, y_range, margin):
    """Enumerate only grid windows near violation geometry."""
    candidates = set()
    for polygon in mask_region.each():
        bbox = polygon.bbox()
        x_indices = _candidate_axis_indices(x_range, bbox.left, bbox.right, margin)
        y_indices = _candidate_axis_indices(y_range, bbox.bottom, bbox.top, margin)
        for x_index in x_indices:
            x = x_range[x_index]
            for y_index in y_indices:
                candidates.add((x, y_range[y_index]))
    return candidates


def _rasterize_tile(metal_region, x, y, scale_factor):
    tile_box = db.Box(x, y, x + PHYSICAL_SIZE, y + PHYSICAL_SIZE)
    tile_metal = metal_region & db.Region(tile_box)
    if tile_metal.is_empty():
        return None

    img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), 0)
    draw = ImageDraw.Draw(img)
    for poly in tile_metal.each():
        shifted_poly = poly.moved(-x, -y)
        pts = [
            (pt.x * scale_factor, pt.y * scale_factor)
            for pt in shifted_poly.each_point_hull()
        ]
        if len(pts) >= 3:
            draw.polygon(pts, fill=1)
    matrix = np.flipud(np.array(img, dtype=np.uint8))
    metal_density = np.sum(matrix) / (IMAGE_SIZE * IMAGE_SIZE)
    if metal_density < 0.03 or metal_density > 0.85:
        return None
    return matrix


def _save_matrix(output_dir, label, tile_prefix, x, y, matrix):
    fname = f"{tile_prefix}tile_{x}_{y}.npy" if tile_prefix else f"tile_{x}_{y}.npy"
    np.save(f"{output_dir}/{label}/{fname}", matrix)


def generate_dataset(input_layout=INPUT_LAYOUT, output_dir=OUTPUT_DIR, tile_prefix="", seed=42):
    input_layout = str(Path(input_layout))
    output_dir = str(Path(output_dir))

    # Keep each build self-contained even after an interrupted or older run.
    # The pipeline also cleans the parent layout folder, but this local guard
    # prevents stale class tiles when the function is called independently.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(f"{output_dir}/clean")
    os.makedirs(f"{output_dir}/dirty")

    layout_metal = db.Layout()
    layout_metal.read(input_layout)
    top_cell_metal = layout_metal.top_cell()
    
    metal_idx = layout_metal.find_layer(*METAL_LAYER)
    metal_region = db.Region(top_cell_metal.begin_shapes_rec(metal_idx))

    # Mask layout is the same as the input layout in our unified file approach
    mask_idx = layout_metal.find_layer(*MASK_LAYER)
    mask_region = db.Region(top_cell_metal.begin_shapes_rec(mask_idx))

    bbox = metal_region.bbox()

    scale_factor = IMAGE_SIZE / PHYSICAL_SIZE 

    print(f"Starting Tiling... Window: {PHYSICAL_SIZE}nm, Res: {PHYSICAL_SIZE/IMAGE_SIZE}nm/px")

    x_range = range(bbox.left, bbox.right - PHYSICAL_SIZE, STRIDE)
    y_range = range(bbox.bottom, bbox.top - PHYSICAL_SIZE, STRIDE)
    if not x_range or not y_range:
        raise ValueError("Layout is smaller than one configured physical tile")

    full_candidates = _mask_candidate_windows(mask_region, x_range, y_range, margin=0)
    safe_candidates = _mask_candidate_windows(mask_region, x_range, y_range, margin=MARGIN)
    full_error_windows = set()
    for x, y in sorted(full_candidates):
        tile_region = db.Region(db.Box(x, y, x + PHYSICAL_SIZE, y + PHYSICAL_SIZE))
        if not (mask_region & tile_region).is_empty():
            full_error_windows.add((x, y))
    dirty_windows = set()
    for x, y in sorted(safe_candidates):
        safe_region = db.Region(
            db.Box(
                x + MARGIN,
                y + MARGIN,
                x + PHYSICAL_SIZE - MARGIN,
                y + PHYSICAL_SIZE - MARGIN,
            )
        )
        if not (mask_region & safe_region).is_empty():
            dirty_windows.add((x, y))

    dirty_count = 0
    for x, y in sorted(dirty_windows):
        matrix = _rasterize_tile(metal_region, x, y, scale_factor)
        if matrix is not None:
            _save_matrix(output_dir, "dirty", tile_prefix, x, y, matrix)
            dirty_count += 1

    # Sample clean windows in a stable random order. This avoids the spatial
    # bias and full-layout raster scan of the former x/y traversal.
    clean_target = dirty_count + 100
    total_steps = len(x_range) * len(y_range)
    order = list(range(total_steps))
    random.Random(seed).shuffle(order)
    clean_count = 0
    for flat_index in order:
        x_index, y_index = divmod(flat_index, len(y_range))
        x, y = x_range[x_index], y_range[y_index]
        if (x, y) in full_error_windows:
            continue
        tile_region = db.Region(db.Box(x, y, x + PHYSICAL_SIZE, y + PHYSICAL_SIZE))
        if not (mask_region & tile_region).is_empty():
            continue
        matrix = _rasterize_tile(metal_region, x, y, scale_factor)
        if matrix is None:
            continue
        _save_matrix(output_dir, "clean", tile_prefix, x, y, matrix)
        clean_count += 1
        if clean_count >= clean_target:
            break

    discard_count = len(full_error_windows - dirty_windows)
    print(
        f"[>] Targeted windows: {len(dirty_windows)} dirty candidates, "
        f"{len(full_error_windows)} touching a violation, {total_steps} total grid windows"
    )

    zip_path = shutil.make_archive(output_dir, "zip", root_dir=output_dir)
    print(f"Created archive: {zip_path}")
    print(f"Total Dirty: {dirty_count} | Total Clean: {clean_count} | Discarded Edge Errors: {discard_count}")
    return dirty_count, clean_count

if __name__ == "__main__":
    generate_dataset()
