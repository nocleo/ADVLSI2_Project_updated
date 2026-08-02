"""Pure decision-threshold selection helpers used by B3 calibration."""

from __future__ import annotations

def classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, object]:
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "predicted_class_counts": {"clean": predictions.count(0), "dirty": predictions.count(1)},
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def threshold_candidates(probabilities: list[float]) -> list[float]:
    values = sorted(set(float(value) for value in probabilities))
    candidates = {0.5, 1e-6, 1.0 - 1e-6}
    candidates.update(values)
    candidates.update((left + right) / 2 for left, right in zip(values, values[1:]))
    return sorted(value for value in candidates if 0 < value < 1)


def select_threshold(
    labels: list[int],
    probabilities: list[float],
    recall_floor: float,
) -> tuple[float, dict[str, object]]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be non-empty and aligned")
    scored = []
    for threshold in threshold_candidates(probabilities):
        metrics = classification_metrics(
            labels, [int(probability >= threshold) for probability in probabilities]
        )
        if float(metrics["recall"]) + 1e-12 >= recall_floor:
            scored.append((threshold, metrics))
    if not scored:
        raise ValueError("No threshold satisfies the validation recall floor")
    return max(
        scored,
        key=lambda item: (
            float(item[1]["f1"]),
            float(item[1]["recall"]),
            float(item[1]["accuracy"]),
            -abs(item[0] - 0.5),
        ),
    )
