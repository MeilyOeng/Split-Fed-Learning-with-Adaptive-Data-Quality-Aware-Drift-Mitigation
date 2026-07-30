from __future__ import annotations

import math
from typing import Optional, Sequence


def _safe_ratio(numerator: int | float, denominator: int | float) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def _percentile(values: Sequence[int | float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aggregate_detector_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Aggregate detector-client classifications, excluding ineligible drift rows."""
    drifted = [row for row in rows if bool(row["drifted_client"])]
    clean = [row for row in rows if not bool(row["drifted_client"])]
    eligible = [row for row in drifted if row["eligible_for_detection"] is True]
    ineligible = [row for row in drifted if row["eligible_for_detection"] is not True]

    true_positives = sum(bool(row["successful_detection"]) for row in eligible)
    false_negatives = len(eligible) - true_positives
    false_positives = sum(bool(row["clean_client_false_positive"]) for row in clean)
    true_negatives = len(clean) - false_positives
    classified = true_positives + false_negatives + false_positives + true_negatives

    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    specificity = _safe_ratio(true_negatives, true_negatives + false_positives)
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    delays = [float(row["detection_delay"]) for row in eligible if row["detection_delay"] is not None]
    observations = sum(int(row["observations_seen"]) for row in rows)
    clean_observations = sum(int(row["observations_seen"]) for row in clean)
    drifted_observations = sum(int(row["observations_seen"]) for row in drifted)
    alarms = sum(int(row["alarm_count"]) for row in rows)
    clean_alarms = sum(int(row["alarm_count"]) for row in clean)
    drifted_alarms = sum(int(row["alarm_count"]) for row in drifted)

    return {
        "detector_client_rows": len(rows),
        "drifted_client_rows": len(drifted),
        "eligible_drifted_client_rows": len(eligible),
        "ineligible_drifted_client_rows": len(ineligible),
        "clean_client_rows": len(clean),
        "confusion_matrix": {
            "true_positive": true_positives,
            "false_negative": false_negatives,
            "false_positive": false_positives,
            "true_negative": true_negatives,
        },
        "accuracy": _safe_ratio(true_positives + true_negatives, classified),
        "balanced_accuracy": ((recall + specificity) / 2.0) if recall is not None and specificity is not None else None,
        "precision": precision,
        "recall_post_drift_detection_rate": recall,
        "specificity": specificity,
        "false_positive_rate": _safe_ratio(false_positives, false_positives + true_negatives),
        "false_negative_rate": _safe_ratio(false_negatives, true_positives + false_negatives),
        "f1_score": _safe_ratio(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None,
        "drifted_client_pre_or_inactive_false_alarm_rate": _safe_ratio(
            sum(int(row["pre_or_inactive_false_alarm_count"]) > 0 for row in drifted), len(drifted)
        ),
        "observations_seen": observations,
        "total_alarms": alarms,
        "clean_client_alarms": clean_alarms,
        "drifted_client_alarms": drifted_alarms,
        "alarms_per_1000_observations": _safe_ratio(1000 * alarms, observations),
        "clean_alarms_per_1000_observations": _safe_ratio(1000 * clean_alarms, clean_observations),
        "drifted_alarms_per_1000_observations": _safe_ratio(1000 * drifted_alarms, drifted_observations),
        "detection_delay": {
            "successful_detections": len(delays),
            "mean": (sum(delays) / len(delays)) if delays else None,
            "median": _percentile(delays, 0.5),
            "p90": _percentile(delays, 0.9),
            "minimum": min(delays) if delays else None,
            "maximum": max(delays) if delays else None,
        },
    }
