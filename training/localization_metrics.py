"""B6.2 pixel, object, vector, classification, and slice metrics."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Iterable, Sequence

import numpy as np
import torch


NM_PER_PIXEL = 8.0
OBJECT_IOU_THRESHOLD = 0.10
VECTOR_MATCH_TOLERANCE_NM = 140.0


def binary_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Binary metrics require equally sized non-empty inputs")
    tn = sum(a == 0 and b == 0 for a, b in zip(labels, predictions))
    fp = sum(a == 0 and b == 1 for a, b in zip(labels, predictions))
    fn = sum(a == 1 and b == 0 for a, b in zip(labels, predictions))
    tp = sum(a == 1 and b == 1 for a, b in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "predicted_class_counts": {
            "clean": predictions.count(0),
            "dirty": predictions.count(1),
        },
    }


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return 8-connected labels and component geometry, using OpenCV when present."""

    binary = np.asarray(mask, dtype=np.uint8)
    try:
        import cv2

        count, labels = cv2.connectedComponents(binary, connectivity=8)
        components = []
        for identifier in range(1, count):
            rows, columns = np.where(labels == identifier)
            components.append(
                {
                    "id": identifier,
                    "area": int(len(rows)),
                    "centroid_rc": [float(rows.mean()), float(columns.mean())],
                }
            )
        return labels.astype(np.int32, copy=False), components
    except ModuleNotFoundError:
        pass

    labels = np.zeros(binary.shape, dtype=np.int32)
    components: list[dict[str, Any]] = []
    identifier = 0
    height, width = binary.shape
    for row, column in zip(*np.where(binary)):
        if labels[row, column]:
            continue
        identifier += 1
        queue = deque([(int(row), int(column))])
        labels[row, column] = identifier
        pixels: list[tuple[int, int]] = []
        while queue:
            current_row, current_column = queue.popleft()
            pixels.append((current_row, current_column))
            for delta_row in (-1, 0, 1):
                for delta_column in (-1, 0, 1):
                    if delta_row == 0 and delta_column == 0:
                        continue
                    next_row = current_row + delta_row
                    next_column = current_column + delta_column
                    if not (0 <= next_row < height and 0 <= next_column < width):
                        continue
                    if binary[next_row, next_column] and not labels[next_row, next_column]:
                        labels[next_row, next_column] = identifier
                        queue.append((next_row, next_column))
        coordinates = np.asarray(pixels, dtype=np.float64)
        components.append(
            {
                "id": identifier,
                "area": len(pixels),
                "centroid_rc": coordinates.mean(axis=0).tolist(),
            }
        )
    return labels, components


def match_raster_components(
    prediction: np.ndarray,
    target: np.ndarray,
    minimum_iou: float = OBJECT_IOU_THRESHOLD,
) -> dict[str, int]:
    predicted_labels, predicted = connected_components(prediction)
    target_labels, targets = connected_components(target)
    candidates: list[tuple[float, int, int]] = []
    for predicted_component in predicted:
        predicted_id = predicted_component["id"]
        predicted_mask = predicted_labels == predicted_id
        overlapping = np.unique(target_labels[predicted_mask])
        for target_id in overlapping:
            if target_id == 0:
                continue
            target_mask = target_labels == target_id
            intersection = int(np.logical_and(predicted_mask, target_mask).sum())
            union = int(np.logical_or(predicted_mask, target_mask).sum())
            iou = intersection / union if union else 0.0
            if iou >= minimum_iou:
                candidates.append((iou, predicted_id, int(target_id)))
    matches = 0
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    for _, predicted_id, target_id in sorted(candidates, reverse=True):
        if predicted_id in used_predictions or target_id in used_targets:
            continue
        used_predictions.add(predicted_id)
        used_targets.add(target_id)
        matches += 1
    return {
        "true_positive": matches,
        "false_positive": len(predicted) - matches,
        "false_negative": len(targets) - matches,
    }


def _component_centroid_nm(component: dict[str, Any], output_box_nm: Sequence[int]) -> np.ndarray:
    row, column = component["centroid_rc"]
    left, _, _, top = output_box_nm
    return np.asarray(
        [left + (column + 0.5) * NM_PER_PIXEL, top - (row + 0.5) * NM_PER_PIXEL],
        dtype=np.float64,
    )


def _point_segment_distance(point: np.ndarray, edge: Sequence[float]) -> float:
    start = np.asarray(edge[:2], dtype=np.float64)
    end = np.asarray(edge[2:], dtype=np.float64)
    delta = end - start
    if float(delta @ delta) == 0:
        return float(np.linalg.norm(point - start))
    position = float(np.clip(((point - start) @ delta) / (delta @ delta), 0.0, 1.0))
    return float(np.linalg.norm(point - (start + position * delta)))


def _vector_matches(
    prediction: np.ndarray, metadata: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, predicted_components = connected_components(prediction)
    predicted_points = [
        _component_centroid_nm(component, metadata["output_box_nm"])
        for component in predicted_components
    ]
    owner_vectors = [
        vector
        for vector in metadata["vectors"]
        if vector["owner_index"] == metadata["grid_index"]
    ]
    candidates: list[tuple[float, int, int]] = []
    for vector_index, vector in enumerate(owner_vectors):
        midpoint = np.asarray(vector["midpoint_nm"], dtype=np.float64)
        for prediction_index, point in enumerate(predicted_points):
            distance = float(np.linalg.norm(point - midpoint))
            if distance <= VECTOR_MATCH_TOLERANCE_NM:
                candidates.append((distance, prediction_index, vector_index))
    used_predictions: set[int] = set()
    used_vectors: set[int] = set()
    matched: list[dict[str, Any]] = []
    for distance, prediction_index, vector_index in sorted(candidates):
        if prediction_index in used_predictions or vector_index in used_vectors:
            continue
        used_predictions.add(prediction_index)
        used_vectors.add(vector_index)
        vector = owner_vectors[vector_index]
        point = predicted_points[prediction_index]
        distance1 = _point_segment_distance(point, vector["edge1_nm"])
        distance2 = _point_segment_distance(point, vector["edge2_nm"])
        matched.append(
            {
                "violation_id": vector["violation_id"],
                "centroid_error_nm": distance,
                "edge_pair_bisector_error_nm": abs(distance1 - distance2) / 2,
                "deficit_nm": float(vector["deficit_nm"]),
                "boundary_distance_nm": _boundary_distance(vector, metadata),
            }
        )
    missed = [
        vector for index, vector in enumerate(owner_vectors) if index not in used_vectors
    ]
    return matched, missed


def _boundary_distance(vector: dict[str, Any], metadata: dict[str, Any]) -> float:
    x, y = vector["midpoint_nm"]
    left, bottom, right, top = metadata["output_box_nm"]
    return float(min(x - left, right - x, y - bottom, top - y))


def _safe_distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def _ratio_metrics(true_positive: int, false_positive: int, false_negative: int) -> dict[str, Any]:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _slice_recall(matched: Sequence[dict[str, Any]], missed: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"matched": 0, "total": 0})
    for item, was_matched in [(item, True) for item in matched] + [
        ({
            "deficit_nm": float(item["deficit_nm"]),
            "boundary_distance_nm": float(item.get("_boundary_distance_nm", math.inf)),
        }, False)
        for item in missed
    ]:
        deficit = float(item["deficit_nm"])
        severity = "near_threshold" if deficit <= 20 else "medium" if deficit <= 60 else "severe"
        boundary = "boundary" if float(item["boundary_distance_nm"]) <= 140 else "interior"
        for key in (f"severity:{severity}", f"boundary:{boundary}"):
            counts[key]["total"] += 1
            counts[key]["matched"] += int(was_matched)
    return {
        key: {
            **value,
            "recall": value["matched"] / value["total"] if value["total"] else 0.0,
        }
        for key, value in sorted(counts.items())
    }


def evaluate_prediction_records(
    records: Sequence[dict[str, Any]],
    segmentation_threshold: float,
    classification_threshold: float,
    include_per_layout: bool = False,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot evaluate an empty prediction set")
    labels: list[int] = []
    predictions: list[int] = []
    pixel_intersection = pixel_prediction = pixel_target = pixel_union = 0
    dirty_intersection = dirty_prediction = dirty_target = dirty_union = 0
    object_counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    vector_matched: list[dict[str, Any]] = []
    vector_missed: list[dict[str, Any]] = []

    for record in records:
        target = record["target_mask"].astype(bool, copy=False)
        prediction = record["mask_probability"] >= segmentation_threshold
        label = int(record["label"])
        labels.append(label)
        predictions.append(int(record["class_probability"] >= classification_threshold))
        intersection = int(np.logical_and(prediction, target).sum())
        union = int(np.logical_or(prediction, target).sum())
        pixel_intersection += intersection
        pixel_prediction += int(prediction.sum())
        pixel_target += int(target.sum())
        pixel_union += union
        if label:
            dirty_intersection += intersection
            dirty_prediction += int(prediction.sum())
            dirty_target += int(target.sum())
            dirty_union += union
        matched_components = match_raster_components(prediction, target)
        for key in object_counts:
            object_counts[key] += matched_components[key]
        matched_vectors, missed_vectors = _vector_matches(prediction, record["metadata"])
        vector_matched.extend(matched_vectors)
        for vector in missed_vectors:
            item = dict(vector)
            item["_boundary_distance_nm"] = _boundary_distance(vector, record["metadata"])
            vector_missed.append(item)

    def overlap(intersection: int, predicted: int, target: int, union: int) -> dict[str, float]:
        return {
            "dice": 2 * intersection / (predicted + target) if predicted + target else 1.0,
            "iou": intersection / union if union else 1.0,
        }

    vector_total = len(vector_matched) + len(vector_missed)
    result: dict[str, Any] = {
        "samples": len(records),
        "segmentation_threshold": segmentation_threshold,
        "classification_threshold": classification_threshold,
        "classification": binary_metrics(labels, predictions),
        "pixel_all": overlap(pixel_intersection, pixel_prediction, pixel_target, pixel_union),
        "pixel_dirty_only": overlap(
            dirty_intersection, dirty_prediction, dirty_target, dirty_union
        ),
        "raster_objects": _ratio_metrics(**object_counts),
        "exact_vector_owners": {
            "total": vector_total,
            "matched": len(vector_matched),
            "recall": len(vector_matched) / vector_total if vector_total else 0.0,
            "centroid_error_nm": _safe_distribution(
                [item["centroid_error_nm"] for item in vector_matched]
            ),
            "edge_pair_bisector_error_nm": _safe_distribution(
                [item["edge_pair_bisector_error_nm"] for item in vector_matched]
            ),
            "slices": _slice_recall(vector_matched, vector_missed),
        },
    }
    if include_per_layout:
        by_layout: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_layout[record["metadata"]["layout"]].append(record)
        result["per_layout"] = {
            layout: evaluate_prediction_records(
                values,
                segmentation_threshold,
                classification_threshold,
                include_per_layout=False,
            )
            for layout, values in sorted(by_layout.items())
        }
    return result


def select_validation_thresholds(
    records: Sequence[dict[str, Any]],
    segmentation_thresholds: Sequence[float],
    classification_thresholds: Sequence[float],
) -> dict[str, Any]:
    segmentation_results = []
    for threshold in segmentation_thresholds:
        metrics = evaluate_prediction_records(records, threshold, 0.5)
        segmentation_results.append(
            {
                "threshold": threshold,
                "raster_object_f1": metrics["raster_objects"]["f1"],
                "dirty_dice": metrics["pixel_dirty_only"]["dice"],
                "vector_owner_recall": metrics["exact_vector_owners"]["recall"],
            }
        )
    selected_segmentation = max(
        segmentation_results,
        key=lambda item: (
            item["raster_object_f1"],
            item["dirty_dice"],
            item["vector_owner_recall"],
            -item["threshold"],
        ),
    )

    classification_results = []
    labels = [int(record["label"]) for record in records]
    probabilities = [float(record["class_probability"]) for record in records]
    for threshold in classification_thresholds:
        classification = binary_metrics(
            labels, [int(probability >= threshold) for probability in probabilities]
        )
        classification_results.append(
            {"threshold": threshold, **classification}
        )
    selected_classification = max(
        classification_results,
        key=lambda item: (
            item["f1"], item["recall"], item["accuracy"], -abs(item["threshold"] - 0.5)
        ),
    )
    return {
        "segmentation": selected_segmentation,
        "classification": selected_classification,
        "segmentation_sweep": segmentation_results,
        "classification_sweep": classification_results,
    }


def collect_model_outputs(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, masks, labels, metadata in loader:
            segmentation_logits, classification_logits = model(images.to(device))
            mask_probabilities = torch.sigmoid(segmentation_logits).cpu().numpy().astype(np.float16)
            class_probabilities = (
                torch.softmax(classification_logits, dim=1)[:, 1].cpu().numpy()
            )
            target_masks = masks.numpy().astype(np.uint8)
            for index, item in enumerate(metadata):
                records.append(
                    {
                        "mask_probability": mask_probabilities[index, 0],
                        "class_probability": float(class_probabilities[index]),
                        "target_mask": target_masks[index, 0],
                        "label": int(labels[index]),
                        "metadata": item,
                    }
                )
    return records
