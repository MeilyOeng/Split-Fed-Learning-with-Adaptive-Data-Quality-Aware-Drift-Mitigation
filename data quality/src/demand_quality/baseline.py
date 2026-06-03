"""Historical baseline calibration for wide-format demand quality scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .io import load_wide_demand_csv, save_json
from .schema import WideDemandDataset
from .statistics import (
    autocorrelation_at_lags,
    histogram_probs,
    make_histogram_bin_edges,
    peak_offpeak_ratio,
    safe_mean,
    safe_std,
    segment_mean_std,
)


def calibrate_client_baselines(
    dataset: WideDemandDataset,
    config: Optional[dict] = None,
) -> dict:
    """Calibrate per-client historical reference statistics from clean demand data."""
    config = config or {}
    client_cfg = dict(config.get("clients", {}) or {})
    selected = list(client_cfg.get("selected", []) or dataset.client_cols)
    accuracy_cfg = dict(config.get("accuracy", {}) or {})
    tc_cfg = dict(config.get("temporal_consistency", {}) or {})
    skew_cfg = dict(config.get("skewness", {}) or {})

    bins = int(skew_cfg.get("jsd_bins", 20))
    strategy = str(skew_cfg.get("histogram_strategy", "quantile"))
    epsilon = float(skew_cfg.get("epsilon", 1e-6))
    peak_hours = list(skew_cfg.get("peak_hours", [17, 18, 19, 20, 21]))
    autocorr_lags = list(tc_cfg.get("autocorr_lags", [1, 24, 168]))
    stationarity_segments = int(tc_cfg.get("stationarity_segments", 4))
    upper_multiplier = float(accuracy_cfg.get("upper_multiplier", 1.5))
    lower_bound = float(accuracy_cfg.get("lower_bound", 0.0))

    baselines: dict[str, dict] = {}
    for client_id in selected:
        if client_id not in dataset.frame.columns:
            continue
        series = pd.to_numeric(dataset.frame[client_id], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue

        edges = make_histogram_bin_edges(valid.to_numpy(dtype=float), bins=bins, strategy=strategy)
        probs = histogram_probs(valid.to_numpy(dtype=float), edges, epsilon=epsilon)
        baselines[str(client_id)] = {
            "sample_count": int(valid.shape[0]),
            "mean": safe_mean(valid),
            "std": safe_std(valid),
            "lower_bound": lower_bound,
            "upper_bound": float(valid.quantile(0.999) * upper_multiplier),
            "peak_ratio": peak_offpeak_ratio(series, dataset.frame[dataset.timestamp_col], peak_hours, epsilon=epsilon),
            "autocorr": {str(k): float(v) for k, v in autocorrelation_at_lags(series, autocorr_lags).items()},
            "stationarity_sigma_ref": segment_mean_std(series, n_segments=stationarity_segments),
            "histogram_edges": [float(x) for x in edges.tolist()],
            "histogram_probs": [float(x) for x in probs.tolist()],
        }

    return {
        "timestamp_col": dataset.timestamp_col,
        "frequency": str(dataset.frequency),
        "autocorr_lags": [int(lag) for lag in autocorr_lags],
        "stationarity_segments": stationarity_segments,
        "peak_hours": [int(hour) for hour in peak_hours],
        "clients": baselines,
    }


def calibrate_demand_baseline_from_csv(
    input_path: Path | str,
    output_path: Path | str,
    config: Optional[dict] = None,
    timestamp_col: Optional[str] = None,
) -> dict:
    """Calibrate a baseline artifact from a clean historical demand CSV."""
    config = config or {}
    configured_freq = config.get("frequency")
    dataset = load_wide_demand_csv(
        input_path,
        timestamp_col=timestamp_col,
        configured_freq=configured_freq,
    )
    payload = calibrate_client_baselines(dataset, config=config)
    save_json(payload, output_path)
    return payload
