"""Print a deterministic synthetic AR timeline around every temporal boundary."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .configuration import load_drift_config
from .pipeline import inject_timeline
from .schedule import resolve_experiment_schedule
from .windowing import assert_horizon_one_consistency, build_forecasting_windows


def _clean_ar(length: int) -> np.ndarray:
    values = np.empty(length, dtype=float)
    values[0] = 2.0
    residual_pattern = (-0.1, 0.0, 0.1)
    for index in range(1, length):
        values[index] = 1.0 + 0.5 * values[index - 1] + residual_pattern[(index - 1) % 3]
    return values


def _temporal(kind: str) -> dict[str, object]:
    if kind in {"incremental", "gradual"}:
        return {"type": kind, "start_index": 12, "end_index": 18, "transition": "linear"}
    if kind == "recurring":
        return {
            "type": "recurring", "start_index": 12,
            "cycle_length": 6, "new_duration": 3, "old_duration": 3,
        }
    return {"type": "sudden", "start_index": 12}


def _config(kind: str):
    return load_drift_config({
        "schema_version": 2,
        "experiment_id": f"synthetic-ar-{kind}",
        "enabled": True,
        "experiment_seed": 42,
        "clients": {"ids": ["demo"]},
        "modes": ["test"],
        "temporal": _temporal(kind),
        "source": {
            "type": "process",
            "old": {
                "intercept": 1.0,
                "ar_coefficients": [0.5],
                "innovation_sigma": 0.1,
            },
            "changes": {
                "intercept_delta": 0.4,
                "ar_coefficient_delta": [0.1],
            },
            "innovation_mode": "clean_residual",
        },
        "windowing": {"window_size": 4, "forecast_horizon": 1, "target_feature_index": 0},
    })


def _run(kind: str, length: int) -> dict[str, object]:
    clean = _clean_ar(length)
    timestamps = list(pd.date_range("2024-01-01", periods=length, freq="h"))
    config = _config(kind)
    resolved = resolve_experiment_schedule(
        config, {("demo-cluster", "demo"): (list(range(length)), timestamps)}
    )
    result = inject_timeline(
        clean, timestamps=timestamps, config=config, resolved_schedule=resolved,
        cluster_id="demo-cluster", client_id="demo",
    )
    windows = build_forecasting_windows(
        result.observed_timeline, result.target_timeline,
        window_size=4, forecast_horizon=1, target_feature_index=0,
    )
    assert_horizon_one_consistency(windows.X, windows.y, target_feature_index=0)
    schedule = resolved.for_client("demo-cluster", "demo")
    boundary_indices = sorted({
        nearby
        for point in schedule if point.change_point_event
        for nearby in (point.absolute_index - 1, point.absolute_index, point.absolute_index + 1)
        if 0 <= nearby < length
    })
    window_by_target = {
        int(target): position for position, target in enumerate(windows.target_indices)
    }
    rows = []
    for index in boundary_indices:
        point = schedule[index]
        manifest = result.manifest[index]
        window_position = window_by_target.get(index)
        rows.append({
            "absolute_index": index,
            "timestamp": point.timestamp,
            "clean_value": float(clean[index]),
            "runtime_value": float(result.latent_timeline[index, 0]),
            "drift_weight": point.drift_weight,
            "active_concept_id": point.active_concept_id,
            "regime_active": point.regime_active,
            "change_point_event": point.change_point_event,
            "new_regime_selected": point.new_regime_selected,
            "transformation_applied": manifest.transformation_applied,
            "window_X": (
                windows.X[window_position, :, 0].tolist() if window_position is not None else None
            ),
            "target_y": (
                float(windows.y[window_position]) if window_position is not None else None
            ),
        })
    return {
        "drift_type": kind,
        "config_hash": config.config_hash,
        "resolved_schedule_hash": resolved.resolved_schedule_hash,
        "autoregressive_consistency_expected": result.autoregressive_consistency_expected,
        "boundary_rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=32)
    args = parser.parse_args()
    if args.length < 25:
        parser.error("--length must be at least 25")
    payload = [_run(kind, args.length) for kind in ("sudden", "incremental", "gradual", "recurring")]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
