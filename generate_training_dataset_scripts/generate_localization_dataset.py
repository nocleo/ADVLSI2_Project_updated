"""Build coverage-correct B6 localization tiles from exact M1 spacing geometry.

The raster mask is a training target.  The vector edge pair stored in JSONL is
the authoritative geometry for coordinate and spacing evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import klayout.db as db
import klayout.rdb as rdb
import numpy as np
from PIL import Image, ImageChops, ImageDraw


M1_LAYER = (68, 20)
M1_2_RULE_NM = 140


@dataclass(frozen=True)
class LocalizationConfig:
    input_nm: int = 1600
    halo_nm: int = 160
    output_nm: int = 1280
    stride_nm: int = 1280
    input_px: int = 200
    output_px: int = 160
    min_rule_nm: int = M1_2_RULE_NM
    seed: int = 42

    def validate(self) -> None:
        if self.input_nm - 2 * self.halo_nm != self.output_nm:
            raise ValueError("input_nm - 2*halo_nm must equal output_nm")
        if self.stride_nm != self.output_nm:
            raise ValueError("stride_nm must equal output_nm for unique gap-free ownership")
        if self.input_nm % self.input_px or self.output_nm % self.output_px:
            raise ValueError("physical sizes must divide evenly by raster sizes")
        if self.input_nm // self.input_px != self.output_nm // self.output_px:
            raise ValueError("input and output rasters must use the same nm/pixel")
        if self.halo_nm <= self.min_rule_nm:
            raise ValueError("halo_nm must be greater than the spacing rule")

    @property
    def nm_per_pixel(self) -> int:
        self.validate()
        return self.input_nm // self.input_px


@dataclass(frozen=True)
class CoverageGrid:
    x0_nm: int
    y0_nm: int
    nx: int
    ny: int
    stride_nm: int = 1280

    @classmethod
    def from_bbox(cls, bbox_nm: Sequence[int], stride_nm: int = 1280) -> "CoverageGrid":
        left, bottom, right, top = map(int, bbox_nm)
        if right <= left or top <= bottom:
            raise ValueError(f"Invalid layout bbox: {bbox_nm}")
        x0 = math.floor(left / stride_nm) * stride_nm
        y0 = math.floor(bottom / stride_nm) * stride_nm
        nx = max(1, math.ceil((right - x0) / stride_nm))
        ny = max(1, math.ceil((top - y0) / stride_nm))
        return cls(x0, y0, nx, ny, stride_nm)

    def output_box(self, ix: int, iy: int) -> tuple[int, int, int, int]:
        self._check_index(ix, iy)
        left = self.x0_nm + ix * self.stride_nm
        bottom = self.y0_nm + iy * self.stride_nm
        return left, bottom, left + self.stride_nm, bottom + self.stride_nm

    def input_box(self, ix: int, iy: int, halo_nm: int) -> tuple[int, int, int, int]:
        left, bottom, right, top = self.output_box(ix, iy)
        return left - halo_nm, bottom - halo_nm, right + halo_nm, top + halo_nm

    def owner(self, x_nm: float, y_nm: float) -> tuple[int, int]:
        ix = math.floor((x_nm - self.x0_nm) / self.stride_nm)
        iy = math.floor((y_nm - self.y0_nm) / self.stride_nm)
        return min(max(ix, 0), self.nx - 1), min(max(iy, 0), self.ny - 1)

    def each_index(self) -> Iterator[tuple[int, int]]:
        for ix in range(self.nx):
            for iy in range(self.ny):
                yield ix, iy

    def _check_index(self, ix: int, iy: int) -> None:
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            raise IndexError((ix, iy))


def _nm_per_dbu(layout: db.Layout) -> float:
    return float(layout.dbu) * 1000.0


def _to_dbu(value_nm: float, nm_per_dbu: float) -> int:
    return int(round(value_nm / nm_per_dbu))


def _to_nm(value_dbu: int, nm_per_dbu: float) -> int:
    return int(round(value_dbu * nm_per_dbu))


def _box_to_nm(box: db.Box, nm_per_dbu: float) -> tuple[int, int, int, int]:
    return tuple(_to_nm(v, nm_per_dbu) for v in (box.left, box.bottom, box.right, box.top))


def _edge_to_nm(edge: db.Edge, nm_per_dbu: float) -> list[int]:
    return [
        _to_nm(edge.p1.x, nm_per_dbu),
        _to_nm(edge.p1.y, nm_per_dbu),
        _to_nm(edge.p2.x, nm_per_dbu),
        _to_nm(edge.p2.y, nm_per_dbu),
    ]


def _edge_orientation(edge: db.Edge) -> str:
    dx = abs(edge.p2.x - edge.p1.x)
    dy = abs(edge.p2.y - edge.p1.y)
    if dx > dy:
        return "horizontal"
    if dy > dx:
        return "vertical"
    return "diagonal"


def load_m1_region(layout_path: Path | str) -> tuple[db.Layout, db.Region, float]:
    layout = db.Layout()
    layout.read(str(layout_path))
    top = layout.top_cell()
    if top is None:
        raise ValueError(f"Layout has no top cell: {layout_path}")
    layer_index = layout.find_layer(*M1_LAYER)
    if layer_index is None:
        raise ValueError(f"Layout has no M1 {M1_LAYER}: {layout_path}")
    region = db.Region(top.begin_shapes_rec(layer_index))
    region.merge()
    if region.is_empty():
        raise ValueError(f"Layout has no M1 geometry: {layout_path}")
    return layout, region, _nm_per_dbu(layout)


def extract_exact_violations(
    metal_region: db.Region,
    nm_per_dbu: float,
    layout_name: str,
    grid: CoverageGrid,
    rule_nm: int = M1_2_RULE_NM,
) -> tuple[list[dict], dict[str, db.Polygon]]:
    pairs = metal_region.space_check(_to_dbu(rule_nm, nm_per_dbu))
    records: list[dict] = []
    polygons: dict[str, db.Polygon] = {}
    for index, pair in enumerate(pairs.each()):
        spacing_nm = _to_nm(pair.distance(), nm_per_dbu)
        edge1 = _edge_to_nm(pair.first, nm_per_dbu)
        edge2 = _edge_to_nm(pair.second, nm_per_dbu)
        midpoint = [
            int(round((edge1[0] + edge1[2] + edge2[0] + edge2[2]) / 4)),
            int(round((edge1[1] + edge1[3] + edge2[1] + edge2[3]) / 4)),
        ]
        owner_ix, owner_iy = grid.owner(*midpoint)
        violation_id = f"{layout_name}:m1.2:{index:06d}"
        polygon = pair.polygon(0)
        polygons[violation_id] = polygon
        records.append(
            {
                "violation_id": violation_id,
                "layout": layout_name,
                "rule": "m1.2",
                "rule_min_nm": rule_nm,
                "spacing_nm": spacing_nm,
                "deficit_nm": rule_nm - spacing_nm,
                "edge1_nm": edge1,
                "edge2_nm": edge2,
                "midpoint_nm": midpoint,
                "bbox_nm": list(_box_to_nm(polygon.bbox(), nm_per_dbu)),
                "orientation": _edge_orientation(pair.first),
                "owner_index": [owner_ix, owner_iy],
                "owned_subpixel_surrogate": False,
            }
        )
    return records, polygons


def extract_rdb_violations(
    report_path: Path | str,
    layout_dbu: float,
    nm_per_dbu: float,
    layout_name: str,
    grid: CoverageGrid,
    rule_nm: int = M1_2_RULE_NM,
) -> tuple[list[dict], dict[str, db.Polygon]]:
    """Read authoritative m1.2 edge pairs from a KLayout report database."""
    report = rdb.ReportDatabase("B6.1")
    report.load(str(report_path))
    pairs: list[db.EdgePair] = []
    for item in report.each_item():
        category = report.category_by_id(item.category_id())
        name = category.name().replace("'", "").strip()
        if name != "m1.2":
            continue
        for value in item.each_value():
            if not value.is_edge_pair():
                raise ValueError(f"m1.2 report item is not an edge pair: {report_path}")
            pairs.append(value.edge_pair().to_itype(layout_dbu))

    records: list[dict] = []
    polygons: dict[str, db.Polygon] = {}
    for index, pair in enumerate(pairs):
        spacing_nm = _to_nm(pair.distance(), nm_per_dbu)
        edge1 = _edge_to_nm(pair.first, nm_per_dbu)
        edge2 = _edge_to_nm(pair.second, nm_per_dbu)
        midpoint = [
            int(round((edge1[0] + edge1[2] + edge2[0] + edge2[2]) / 4)),
            int(round((edge1[1] + edge1[3] + edge2[1] + edge2[3]) / 4)),
        ]
        owner_ix, owner_iy = grid.owner(*midpoint)
        violation_id = f"{layout_name}:m1.2:{index:06d}"
        polygon = pair.polygon(0)
        polygons[violation_id] = polygon
        records.append(
            {
                "violation_id": violation_id,
                "layout": layout_name,
                "rule": "m1.2",
                "rule_min_nm": rule_nm,
                "spacing_nm": spacing_nm,
                "deficit_nm": rule_nm - spacing_nm,
                "edge1_nm": edge1,
                "edge2_nm": edge2,
                "midpoint_nm": midpoint,
                "bbox_nm": list(_box_to_nm(polygon.bbox(), nm_per_dbu)),
                "orientation": _edge_orientation(pair.first),
                "owner_index": [owner_ix, owner_iy],
                "owned_subpixel_surrogate": False,
            }
        )
    if not records:
        raise RuntimeError(f"No m1.2 edge pairs in {report_path}")
    return records, polygons


def nm_to_pixel(value_nm: float, origin_nm: float, nm_per_pixel: int) -> float:
    return (value_nm - origin_nm) / nm_per_pixel


def pixel_to_nm(value_px: float, origin_nm: float, nm_per_pixel: int) -> float:
    return origin_nm + value_px * nm_per_pixel


def _rasterize_region(
    region: db.Region,
    box_nm: Sequence[int],
    size_px: int,
    nm_per_dbu: float,
) -> np.ndarray:
    left_nm, bottom_nm, right_nm, top_nm = map(int, box_nm)
    crop = db.Box(*(_to_dbu(v, nm_per_dbu) for v in box_nm))
    clipped = region & db.Region(crop)
    image = Image.new("L", (size_px, size_px), 0)
    draw = ImageDraw.Draw(image)
    width_nm = right_nm - left_nm
    height_nm = top_nm - bottom_nm
    for polygon in clipped.each():
        points = []
        for point in polygon.each_point_hull():
            x_nm = point.x * nm_per_dbu
            y_nm = point.y * nm_per_dbu
            points.append(
                (
                    (x_nm - left_nm) * size_px / width_nm,
                    (y_nm - bottom_nm) * size_px / height_nm,
                )
            )
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    return np.flipud(np.asarray(image, dtype=np.uint8)).copy()


def rasterize_registered_tile(
    metal_region: db.Region,
    marker_region: db.Region,
    input_box_nm: Sequence[int],
    output_box_nm: Sequence[int],
    config: LocalizationConfig,
    nm_per_dbu: float,
) -> tuple[np.ndarray, np.ndarray]:
    image = _rasterize_region(metal_region, input_box_nm, config.input_px, nm_per_dbu)
    mask = _rasterize_region(marker_region, output_box_nm, config.output_px, nm_per_dbu)
    if image.shape != (config.input_px, config.input_px):
        raise AssertionError(image.shape)
    if mask.shape != (config.output_px, config.output_px):
        raise AssertionError(mask.shape)
    return image, mask


def _candidate_indices_for_bbox(
    bbox_nm: Sequence[int], grid: CoverageGrid
) -> Iterator[tuple[int, int]]:
    left, bottom, right, top = map(int, bbox_nm)
    epsilon = 1e-9
    ix0, iy0 = grid.owner(left, bottom)
    ix1, iy1 = grid.owner(right - epsilon, top - epsilon)
    for ix in range(min(ix0, ix1), max(ix0, ix1) + 1):
        for iy in range(min(iy0, iy1), max(iy0, iy1) + 1):
            yield ix, iy


def _tile_id(layout_name: str, ix: int, iy: int) -> str:
    return f"{layout_name}__x{ix:05d}_y{iy:05d}"


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[1]
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _completed_summary_is_valid(
    summary: dict,
    output_dir: Path,
    layout_path: Path,
    report_path: Path | None,
    config: LocalizationConfig,
) -> bool:
    try:
        if summary["schema_version"] != 1 or summary["phase"] != "B6.1":
            return False
        if summary["config"] != asdict(config):
            return False
        if summary["source_sha256"] != _sha256(layout_path):
            return False
        expected_report_hash = _sha256(report_path) if report_path is not None else None
        if summary.get("report_sha256") != expected_report_hash:
            return False
        with (output_dir / "violations.jsonl").open(encoding="utf-8") as handle:
            vector_count = sum(1 for _ in handle)
        with (output_dir / "tiles.jsonl").open(encoding="utf-8") as handle:
            tile_count = sum(1 for _ in handle)
        dirty_files = len(list((output_dir / "tiles" / "dirty").glob("*.npz")))
        clean_files = len(list((output_dir / "tiles" / "clean").glob("*.npz")))
        return (
            vector_count == summary["exact_violation_count"]
            and tile_count == summary["dirty_tile_count"] + summary["clean_tile_count"]
            and dirty_files == summary["dirty_tile_count"]
            and clean_files == summary["clean_tile_count"]
            and (output_dir / "visual_audit.png").is_file()
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        return False


def _tile_record(
    layout_name: str,
    ix: int,
    iy: int,
    grid: CoverageGrid,
    config: LocalizationConfig,
    label: str,
    relative_path: str,
    image: np.ndarray,
    mask: np.ndarray,
    violation_ids: Sequence[str],
    surrogate_ids: Sequence[str],
    omitted_subpixel_ids: Sequence[str],
) -> dict:
    return {
        "tile_id": _tile_id(layout_name, ix, iy),
        "layout": layout_name,
        "label": label,
        "grid_index": [ix, iy],
        "input_box_nm": list(grid.input_box(ix, iy, config.halo_nm)),
        "output_box_nm": list(grid.output_box(ix, iy)),
        "image_path": relative_path,
        "image_shape": list(image.shape),
        "mask_shape": list(mask.shape),
        "metal_density": float(image.mean()),
        "mask_pixels": int(mask.sum()),
        "violation_ids": list(violation_ids),
        "surrogate_violation_ids": list(surrogate_ids),
        "omitted_subpixel_violation_ids": list(omitted_subpixel_ids),
    }


def _save_tile(path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, image=image, mask=mask)


def _intersecting_violation_ids(
    records: Sequence[dict],
    polygons: dict[str, db.Polygon],
    output_box_nm: Sequence[int],
    nm_per_dbu: float,
) -> list[str]:
    output_region = db.Region(db.Box(*(_to_dbu(v, nm_per_dbu) for v in output_box_nm)))
    result = []
    for record in records:
        bbox = record["bbox_nm"]
        if bbox[2] <= output_box_nm[0] or bbox[0] >= output_box_nm[2]:
            continue
        if bbox[3] <= output_box_nm[1] or bbox[1] >= output_box_nm[3]:
            continue
        if not (db.Region(polygons[record["violation_id"]]) & output_region).is_empty():
            result.append(record["violation_id"])
    return result


def _apply_owned_subpixel_surrogate(
    mask: np.ndarray,
    records_by_id: dict[str, dict],
    violation_ids: Sequence[str],
    ix: int,
    iy: int,
    output_box_nm: Sequence[int],
    config: LocalizationConfig,
    polygons: dict[str, db.Polygon],
    nm_per_dbu: float,
) -> tuple[list[str], list[str]]:
    surrogates: list[str] = []
    omitted: list[str] = []
    left, bottom, _, _ = output_box_nm
    for violation_id in violation_ids:
        individual = _rasterize_region(
            db.Region(polygons[violation_id]),
            output_box_nm,
            config.output_px,
            nm_per_dbu,
        )
        if individual.any():
            continue
        if records_by_id[violation_id]["owner_index"] != [ix, iy]:
            omitted.append(violation_id)
            continue
        x_nm, y_nm = records_by_id[violation_id]["midpoint_nm"]
        col = min(max(int((x_nm - left) // config.nm_per_pixel), 0), config.output_px - 1)
        row_from_bottom = min(
            max(int((y_nm - bottom) // config.nm_per_pixel), 0), config.output_px - 1
        )
        mask[config.output_px - 1 - row_from_bottom, col] = 1
        records_by_id[violation_id]["owned_subpixel_surrogate"] = True
        surrogates.append(violation_id)
    return surrogates, omitted


def _audit_categories(
    tile_records: Sequence[dict], violation_records: Sequence[dict]
) -> dict[str, str]:
    clean = [record for record in tile_records if record["label"] == "clean"]
    dirty = [record for record in tile_records if record["label"] == "dirty"]
    if not clean or not dirty:
        raise RuntimeError("Visual audit requires both clean and dirty tiles")
    records_by_id = {record["violation_id"]: record for record in violation_records}
    density_sorted = sorted(tile_records, key=lambda record: record["metal_density"])
    result = {
        "clean": clean[0]["tile_id"],
        "sparse": density_sorted[0]["tile_id"],
        "dense": density_sorted[-1]["tile_id"],
    }
    for orientation in ("horizontal", "vertical"):
        match = next(
            (
                tile
                for tile in dirty
                if any(records_by_id[v]["orientation"] == orientation for v in tile["violation_ids"])
            ),
            None,
        )
        if match is None:
            raise RuntimeError(f"No {orientation} visual-audit example")
        result[orientation] = match["tile_id"]
    boundary = next(
        (
            tile
            for tile in dirty
            if any(records_by_id[v]["owner_index"] != tile["grid_index"] for v in tile["violation_ids"])
            or tile["surrogate_violation_ids"]
        ),
        None,
    )
    if boundary is None:
        raise RuntimeError("No boundary-crossing visual-audit example")
    result["boundary"] = boundary["tile_id"]
    near = min(
        dirty,
        key=lambda tile: min(records_by_id[v]["deficit_nm"] for v in tile["violation_ids"]),
    )
    result["near_threshold"] = near["tile_id"]
    return result


def _write_visual_audit(
    output_path: Path, categories: dict[str, str], tile_records: Sequence[dict], root: Path
) -> None:
    by_id = {record["tile_id"]: record for record in tile_records}
    labels = list(categories)
    panel = Image.new("L", (len(labels) * 200, 220), 0)
    for column, label in enumerate(labels):
        record = by_id[categories[label]]
        arrays = np.load(root / record["image_path"])
        image = Image.fromarray((arrays["image"] * 180).astype(np.uint8), mode="L")
        if record["label"] == "dirty":
            mask = Image.fromarray((arrays["mask"] * 255).astype(np.uint8), mode="L").resize((160, 160))
            image.paste(ImageChops.lighter(image.crop((20, 20, 180, 180)), mask), (20, 20))
        panel.paste(image, (column * 200, 20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def build_layout_dataset(
    layout_name: str,
    layout_path: Path | str,
    output_dir: Path | str,
    config: LocalizationConfig | None = None,
    force: bool = False,
    report_path: Path | str | None = None,
) -> dict:
    config = config or LocalizationConfig()
    config.validate()
    layout_path = Path(layout_path)
    output_dir = Path(output_dir)
    report_path = Path(report_path) if report_path is not None else None
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not force:
        completed = json.loads(summary_path.read_text(encoding="utf-8"))
        if _completed_summary_is_valid(
            completed, output_dir, layout_path, report_path, config
        ):
            return completed
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    layout, metal_region, nm_per_dbu = load_m1_region(layout_path)
    grid = CoverageGrid.from_bbox(_box_to_nm(metal_region.bbox(), nm_per_dbu), config.stride_nm)
    if report_path is not None:
        violations, polygons = extract_rdb_violations(
            report_path,
            layout.dbu,
            nm_per_dbu,
            layout_name,
            grid,
            config.min_rule_nm,
        )
        geometry_source = "klayout_rdb"
    else:
        violations, polygons = extract_exact_violations(
            metal_region, nm_per_dbu, layout_name, grid, config.min_rule_nm
        )
        geometry_source = "region_space_check"
    if not violations:
        raise RuntimeError(f"No exact m1.2 violations found for {layout_name}")
    records_by_id = {record["violation_id"]: record for record in violations}
    marker_region = db.Region()
    dirty_indices: set[tuple[int, int]] = set()
    for record in violations:
        polygon = polygons[record["violation_id"]]
        marker_region.insert(polygon)
        for index in _candidate_indices_for_bbox(record["bbox_nm"], grid):
            output_box = grid.output_box(*index)
            output_region = db.Region(
                db.Box(*(_to_dbu(v, nm_per_dbu) for v in output_box))
            )
            if not (db.Region(polygon) & output_region).is_empty():
                dirty_indices.add(index)

    tile_records: list[dict] = []
    for ix, iy in sorted(dirty_indices):
        input_box = grid.input_box(ix, iy, config.halo_nm)
        output_box = grid.output_box(ix, iy)
        image, mask = rasterize_registered_tile(
            metal_region, marker_region, input_box, output_box, config, nm_per_dbu
        )
        ids = _intersecting_violation_ids(
            violations, polygons, output_box, nm_per_dbu
        )
        surrogates, omitted_subpixel = _apply_owned_subpixel_surrogate(
            mask, records_by_id, ids, ix, iy, output_box, config, polygons, nm_per_dbu
        )
        if not mask.any():
            # A non-owner subpixel fragment remains exact in JSONL but is not a
            # useful raster target in this neighboring output tile.
            continue
        tile_id = _tile_id(layout_name, ix, iy)
        relative_path = f"tiles/dirty/{tile_id}.npz"
        _save_tile(output_dir / relative_path, image, mask)
        tile_records.append(
            _tile_record(
                layout_name, ix, iy, grid, config, "dirty", relative_path,
                image, mask, ids, surrogates, omitted_subpixel,
            )
        )

    clean_target = len(tile_records)
    clean_indices = [index for index in grid.each_index() if index not in dirty_indices]
    random.Random(config.seed ^ int(hashlib.sha256(layout_name.encode()).hexdigest()[:8], 16)).shuffle(clean_indices)
    for ix, iy in clean_indices:
        input_box = grid.input_box(ix, iy, config.halo_nm)
        output_box = grid.output_box(ix, iy)
        image, mask = rasterize_registered_tile(
            metal_region, marker_region, input_box, output_box, config, nm_per_dbu
        )
        if not image.any() or mask.any():
            continue
        tile_id = _tile_id(layout_name, ix, iy)
        relative_path = f"tiles/clean/{tile_id}.npz"
        _save_tile(output_dir / relative_path, image, mask)
        tile_records.append(
            _tile_record(
                layout_name, ix, iy, grid, config, "clean", relative_path,
                image, mask, [], [], [],
            )
        )
        if sum(record["label"] == "clean" for record in tile_records) >= clean_target:
            break
    clean_count = sum(record["label"] == "clean" for record in tile_records)
    if clean_count != clean_target:
        raise RuntimeError(
            f"{layout_name}: found {clean_count} clean tiles for {clean_target} dirty tiles"
        )

    owner_ids = {
        violation_id
        for record in tile_records
        if record["label"] == "dirty"
        for violation_id in record["violation_ids"]
        if records_by_id[violation_id]["owner_index"] == record["grid_index"]
    }
    missing_owners = sorted(set(records_by_id) - owner_ids)
    if missing_owners:
        raise RuntimeError(f"{layout_name}: {len(missing_owners)} violations lack owner tiles")

    _write_jsonl(output_dir / "violations.jsonl", violations)
    _write_jsonl(output_dir / "tiles.jsonl", tile_records)
    categories = _audit_categories(tile_records, violations)
    _write_visual_audit(output_dir / "visual_audit.png", categories, tile_records, output_dir)
    summary = {
        "schema_version": 1,
        "phase": "B6.1",
        "layout": layout_name,
        "source": _portable_path(layout_path),
        "source_sha256": _sha256(layout_path),
        "geometry_source": geometry_source,
        "report_sha256": _sha256(report_path) if report_path is not None else None,
        "config": asdict(config),
        "nm_per_pixel": config.nm_per_pixel,
        "coverage_grid": asdict(grid),
        "layout_bbox_nm": list(_box_to_nm(metal_region.bbox(), nm_per_dbu)),
        "exact_violation_count": len(violations),
        "unique_owner_count": len(owner_ids),
        "dirty_tile_count": len(tile_records) - clean_count,
        "clean_tile_count": clean_count,
        "surrogate_target_count": sum(len(r["surrogate_violation_ids"]) for r in tile_records),
        "omitted_nonowner_subpixel_fragment_count": sum(
            len(r["omitted_subpixel_violation_ids"]) for r in tile_records
        ),
        "visual_audit": categories,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-name", required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rdb", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    summary = build_layout_dataset(
        args.layout_name,
        args.layout,
        args.output,
        LocalizationConfig(seed=args.seed),
        force=args.force,
        report_path=args.rdb,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
