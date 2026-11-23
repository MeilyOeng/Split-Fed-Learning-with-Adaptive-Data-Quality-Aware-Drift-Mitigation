"""Demand-specific feature extraction for per-client window scoring models."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from .statistics import (
    autocorrelation_at_lags,
    calculate_psi,
    histogram_probs,
    peak_offpeak_ratio,
    rolling_median_mad_anomaly_mask,
    safe_std,
    stuck_segment_mask,
)
from .windowing import ClientWindowSlice


def _to_numeric_series(values: pd.Series | pd.Index | np.ndarray) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _raw_window_series(window: ClientWindowSlice) -> tuple[pd.Series, pd.Series]:
    raw_ts = pd.to_datetime(window.current_raw_window[window.timestamp_col], errors="coerce")
    raw_values = pd.to_numeric(window.current_raw_window.get(window.client_id), errors="coerce")
    return raw_ts, raw_values


def _raw_history_series(window: ClientWindowSlice) -> tuple[pd.Series, pd.Series]:
    raw_ts = pd.to_datetime(window.current_raw_history[window.timestamp_col], errors="coerce")
    raw_values = pd.to_numeric(window.current_raw_history.get(window.client_id), errors="coerce")
    return raw_ts, raw_values


def _aligned_window_values(window: ClientWindowSlice) -> pd.Series:
    aligned = pd.Series(index=window.expected_timestamps, dtype=float)
    if window.client_id not in window.current_window.frame.columns:
        return aligned

    current = window.current_window.frame[[window.timestamp_col, window.client_id]].copy()
    current[window.timestamp_col] = pd.to_datetime(current[window.timestamp_col], errors="coerce")
    current[window.client_id] = pd.to_numeric(current[window.client_id], errors="coerce")
    current = current.dropna(subset=[window.timestamp_col])
    if current.empty:
        return aligned

    collapsed = current.groupby(window.timestamp_col, sort=True)[window.client_id].mean()
    common = aligned.index.intersection(collapsed.index)
    if len(common) > 0:
        aligned.loc[common] = collapsed.loc[common].to_numpy(dtype=float)
    return aligned


def _cyclical_time_features(ts: pd.Timestamp) -> dict[str, float]:
    if pd.isna(ts):
        return {
            "meta_hour_sin": 0.0,
            "meta_hour_cos": 0.0,
            "meta_dow_sin": 0.0,
            "meta_dow_cos": 0.0,
            "meta_month_sin": 0.0,
            "meta_month_cos": 0.0,
        }
    hour = ts.hour / 24.0
    dow = ts.dayofweek / 7.0
    month = (ts.month - 1) / 12.0
    return {
        "meta_hour_sin": float(math.sin(2.0 * math.pi * hour)),
        "meta_hour_cos": float(math.cos(2.0 * math.pi * hour)),
        "meta_dow_sin": float(math.sin(2.0 * math.pi * dow)),
        "meta_dow_cos": float(math.cos(2.0 * math.pi * dow)),
        "meta_month_sin": float(math.sin(2.0 * math.pi * month)),
        "meta_month_cos": float(math.cos(2.0 * math.pi * month)),
    }


def _longest_run(mask: pd.Series) -> int:
    longest = 0
    current = 0
    for flag in mask.fillna(False).astype(bool).tolist():
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _baseline_for_client(
    baseline_artifact: Optional[dict],
    client_id: str,
) -> dict:
    if not baseline_artifact:
        return {}
    clients = baseline_artifact.get("clients", {}) or {}
    payload = clients.get(str(client_id), {}) or {}
    return payload if isinstance(payload, dict) else {}


def _history_freshness_features(
    window: ClientWindowSlice,
    config: Optional[dict],
) -> dict[str, float]:
    config = config or {}
    freshness_cfg = dict((config.get("freshness", {}) or {}))
    recency_window = pd.to_timedelta(freshness_cfg.get("window_recency_span", "168h"))
    reference_time = pd.Timestamp(window.window_end)
    history_ts, history_values = _raw_history_series(window)
    valid_history = history_ts.notna() & history_values.notna() & np.isfinite(history_values)
    available_start = history_ts.loc[history_ts.notna()].min() if history_ts.notna().any() else pd.NaT

    if valid_history.any():
        latest_valid_ts = history_ts.loc[valid_history].max()
        age_hours = float((reference_time - latest_valid_ts) / pd.Timedelta(hours=1))
    else:
        age_hours = float(recency_window / pd.Timedelta(hours=1)) if recency_window > pd.Timedelta(0) else np.nan

    if recency_window > pd.Timedelta(0) and pd.notna(available_start):
        recent_start = max(reference_time - recency_window, pd.Timestamp(available_start))
        expected_recent_count = int(
            len(pd.date_range(start=recent_start, end=reference_time, freq=window.frequency))
        )
        raw_recent_mask = valid_history & (history_ts >= recent_start) & (history_ts <= reference_time)
        recent_valid_count = int(history_ts.loc[raw_recent_mask].nunique())
        recent_valid_ratio = float(recent_valid_count / max(1, expected_recent_count))
    else:
        expected_recent_count = 0
        recent_valid_count = 0
        recent_valid_ratio = 0.0

    aligned = _aligned_window_values(window)
    trailing_missing_run = _longest_run(aligned.isna()[::-1]) if len(aligned) else 0
    latest_expected_ts = pd.Timestamp(window.expected_timestamps[-1]) if len(window.expected_timestamps) else pd.NaT
    has_latest_value = (
        float(bool(pd.notna(aligned.loc[latest_expected_ts]))) if pd.notna(latest_expected_ts) and latest_expected_ts in aligned.index else 0.0
    )
    return {
        "fresh_age_hours": age_hours,
        "fresh_recent_window_hours": float(recency_window / pd.Timedelta(hours=1))
        if recency_window > pd.Timedelta(0)
        else 0.0,
        "fresh_recent_valid_count": float(recent_valid_count),
        "fresh_recent_expected_count": float(expected_recent_count),
        "fresh_recent_valid_ratio": float(min(1.0, max(0.0, recent_valid_ratio))),
        "fresh_trailing_missing_run": float(trailing_missing_run),
        "fresh_has_latest_value": has_latest_value,
    }


def extract_client_window_features(
    window: ClientWindowSlice,
    *,
    config: Optional[dict] = None,
    baseline_artifact: Optional[dict] = None,
) -> dict[str, float]:
    """Extract model features from a client-window without using labels."""
    config = config or {}
    accuracy_cfg = dict((config.get("accuracy", {}) or {}))
    consistency_cfg = dict((config.get("consistency", {}) or {}))
    validity_cfg = dict((config.get("validity", {}) or {}))

    raw_ts, raw_values = _raw_window_series(window)
    aligned = _aligned_window_values(window)
    valid_raw_ts = raw_ts.dropna().reset_index(drop=True)
    finite_mask = raw_values.notna() & np.isfinite(raw_values)
    finite_values = raw_values.loc[finite_mask]
    non_null_count = int(raw_values.notna().sum())
    finite_count = int(finite_mask.sum())
    expected_count = int(len(window.expected_timestamps))
    observed_unique_count = int(valid_raw_ts.nunique())
    duplicate_count = max(0, int(valid_raw_ts.shape[0] - observed_unique_count))
    missing_expected_count = max(0, expected_count - observed_unique_count)

    features: dict[str, float] = {
        "meta_expected_count": float(expected_count),
        "meta_frequency_hours": float(window.frequency / pd.Timedelta(hours=1)),
        "meta_window_span_hours": float((window.window_end - window.window_start) / pd.Timedelta(hours=1)),
        "meta_raw_row_count": float(len(window.current_raw_window)),
        "meta_valid_timestamp_count": float(valid_raw_ts.shape[0]),
        "meta_unique_timestamp_count": float(observed_unique_count),
        "meta_duplicate_timestamp_count": float(duplicate_count),
        "meta_duplicate_timestamp_ratio": float(duplicate_count / max(1, valid_raw_ts.shape[0])),
        "comp_non_null_count": float(non_null_count),
        "comp_finite_value_count": float(finite_count),
        "comp_non_null_ratio_expected": float(non_null_count / max(1, expected_count)),
        "comp_finite_ratio_expected": float(finite_count / max(1, expected_count)),
        "comp_finite_ratio_observed": float(finite_count / max(1, len(window.current_raw_window))),
        "comp_missing_expected_count": float(missing_expected_count),
        "comp_missing_expected_ratio": float(missing_expected_count / max(1, expected_count)),
        "comp_null_count": float(len(window.current_raw_window) - non_null_count),
        "comp_null_ratio_observed": float((len(window.current_raw_window) - non_null_count) / max(1, len(window.current_raw_window))),
    }
    features.update(_cyclical_time_features(pd.Timestamp(window.window_end)))

    if valid_raw_ts.shape[0] >= 2:
        deltas = valid_raw_ts.diff().dropna()
        delta_hours = deltas / pd.Timedelta(hours=1)
        tolerance = max(pd.Timedelta(seconds=1), window.frequency * 0.1)
        regular_ratio = float((deltas.sub(window.frequency).abs() <= tolerance).mean())
        monotonic_ratio = float((deltas > pd.Timedelta(0)).mean())
        max_gap_steps = float((delta_hours.max() / max(1e-9, window.frequency / pd.Timedelta(hours=1))))
        features.update(
            {
                "time_interval_min_hours": float(delta_hours.min()),
                "time_interval_median_hours": float(delta_hours.median()),
                "time_interval_max_hours": float(delta_hours.max()),
                "time_regular_interval_ratio": regular_ratio,
                "time_monotonic_ratio": monotonic_ratio,
                "time_max_gap_steps": max_gap_steps,
            }
        )
    else:
        features.update(
            {
                "time_interval_min_hours": 0.0,
                "time_interval_median_hours": 0.0,
                "time_interval_max_hours": 0.0,
                "time_regular_interval_ratio": 0.0,
                "time_monotonic_ratio": 0.0,
                "time_max_gap_steps": 0.0,
            }
        )

    if finite_count > 0:
        values = finite_values.to_numpy(dtype=float)
        q05 = float(np.quantile(values, 0.05))
        q25 = float(np.quantile(values, 0.25))
        q75 = float(np.quantile(values, 0.75))
        q95 = float(np.quantile(values, 0.95))
        value_mean = float(np.mean(values))
        value_std = float(np.std(values))
        value_range = float(np.max(values) - np.min(values))
        features.update(
            {
                "val_negative_ratio": float((values < 0).mean()),
                "val_zero_ratio": float((np.abs(values) <= 1e-9).mean()),
                "val_mean": value_mean,
                "val_std": value_std,
                "val_min": float(np.min(values)),
                "val_max": float(np.max(values)),
                "val_median": float(np.median(values)),
                "val_q05": q05,
                "val_q25": q25,
                "val_q75": q75,
                "val_q95": q95,
                "val_iqr": float(q75 - q25),
                "val_range": value_range,
                "val_coeff_var": float(value_std / max(abs(value_mean), 1e-6)),
            }
        )
    else:
        features.update(
            {
                "val_negative_ratio": 0.0,
                "val_zero_ratio": 0.0,
                "val_mean": np.nan,
                "val_std": np.nan,
                "val_min": np.nan,
                "val_max": np.nan,
                "val_median": np.nan,
                "val_q05": np.nan,
                "val_q25": np.nan,
                "val_q75": np.nan,
                "val_q95": np.nan,
                "val_iqr": np.nan,
                "val_range": np.nan,
                "val_coeff_var": np.nan,
            }
        )

    valid_aligned = aligned.dropna()
    if valid_aligned.shape[0] >= 2:
        diffs = valid_aligned.diff().dropna()
        stuck_mask = stuck_segment_mask(
            aligned,
            min_len=int(consistency_cfg.get("freeze_min_len", 6)),
            atol=float(consistency_cfg.get("freeze_atol", 0.0)),
        )
        anomaly_local = pd.Series(False, index=aligned.index)
        applicable_local = aligned.notna() & aligned.shift(1).notna() & aligned.shift(-1).notna()
        if int(applicable_local.sum()) > 0:
            sigma_ref = max(
                safe_std(valid_aligned.to_numpy(dtype=float)),
                float(accuracy_cfg.get("min_threshold_abs", 0.0)),
                1e-6,
            )
            kappa = float(
                accuracy_cfg.get(
                    "local_spike_kappa",
                    accuracy_cfg.get("seasonal_threshold_multiplier", 3.0),
                )
            )
            x_hat = (aligned.shift(1) + aligned.shift(-1)) / 2.0
            anomaly_local.loc[applicable_local] = (
                (aligned.loc[applicable_local] - x_hat.loc[applicable_local]).abs() > (kappa * sigma_ref)
            )
        anomaly_robust, applicable_robust, _ = rolling_median_mad_anomaly_mask(
            aligned,
            window=int(accuracy_cfg.get("rolling_window", 24)),
            mad_multiplier=float(accuracy_cfg.get("mad_multiplier", 4.0)),
            min_history=int(accuracy_cfg.get("min_history", 6)),
            min_threshold_abs=float(accuracy_cfg.get("min_threshold_abs", 0.0)),
        )
        autocorr = autocorrelation_at_lags(valid_aligned.reset_index(drop=True), [1, 24])
        x_axis = np.arange(valid_aligned.shape[0], dtype=float)
        slope = float(np.polyfit(x_axis, valid_aligned.to_numpy(dtype=float), deg=1)[0]) if valid_aligned.shape[0] >= 2 else 0.0
        features.update(
            {
                "dyn_abs_diff_mean": float(diffs.abs().mean()),
                "dyn_abs_diff_std": float(diffs.abs().std(ddof=0)),
                "dyn_abs_diff_max": float(diffs.abs().max()),
                "dyn_positive_step_ratio": float((diffs > 0).mean()),
                "dyn_zero_step_ratio": float((diffs.abs() <= 1e-9).mean()),
                "dyn_negative_step_ratio": float((diffs < 0).mean()),
                "dyn_stuck_ratio": float(stuck_mask.loc[aligned.notna()].mean()) if aligned.notna().any() else 0.0,
                "dyn_longest_stuck_run": float(_longest_run(stuck_mask)),
                "dyn_autocorr_lag1": float(autocorr.get(1, 0.0)),
                "dyn_autocorr_lag24": float(autocorr.get(24, 0.0)),
                "dyn_linear_slope": slope,
                "acc_local_anomaly_ratio": float(anomaly_local.loc[applicable_local].mean()) if int(applicable_local.sum()) > 0 else 0.0,
                "acc_robust_anomaly_ratio": float(anomaly_robust.loc[applicable_robust].mean()) if int(applicable_robust.sum()) > 0 else 0.0,
                "acc_noise_mafd": float(diffs.abs().mean()),
            }
        )
    else:
        features.update(
            {
                "dyn_abs_diff_mean": 0.0,
                "dyn_abs_diff_std": 0.0,
                "dyn_abs_diff_max": 0.0,
                "dyn_positive_step_ratio": 0.0,
                "dyn_zero_step_ratio": 0.0,
                "dyn_negative_step_ratio": 0.0,
                "dyn_stuck_ratio": 0.0,
                "dyn_longest_stuck_run": 0.0,
                "dyn_autocorr_lag1": 0.0,
                "dyn_autocorr_lag24": 0.0,
                "dyn_linear_slope": 0.0,
                "acc_local_anomaly_ratio": 0.0,
                "acc_robust_anomaly_ratio": 0.0,
                "acc_noise_mafd": 0.0,
            }
        )

    features.update(_history_freshness_features(window, config=config))

    baseline = _baseline_for_client(baseline_artifact, window.client_id)
    features["base_has_baseline"] = 1.0 if baseline else 0.0
    if baseline:
        baseline_mean = float(baseline.get("mean", 0.0))
        baseline_std = float(baseline.get("std", 0.0))
        baseline_q99 = float(baseline.get("q99", 0.0))
        baseline_mafd = float(baseline.get("mafd", 0.0))
        baseline_diff_mad = float(baseline.get("diff_mad", 0.0))
        lower_bound = float(baseline.get("lower_bound", validity_cfg.get("lower_bound", 0.0)))
        upper_bound = float(baseline.get("upper_bound", baseline_q99))
        peak_hours = baseline_artifact.get("peak_hours", [17, 18, 19, 20, 21]) if baseline_artifact else [17, 18, 19, 20, 21]
        current_peak_ratio = peak_offpeak_ratio(raw_values, raw_ts, peak_hours)
        baseline_peak_ratio = float(baseline.get("peak_ratio", 0.0))
        z = (finite_values - baseline_mean) / max(baseline_std, 1e-6) if finite_count > 0 else pd.Series(dtype=float)
        hist_edges = baseline.get("histogram_edges", []) or []
        hist_probs = baseline.get("histogram_probs", []) or []
        if len(hist_edges) >= 2 and len(hist_probs) == len(hist_edges) - 1 and finite_count > 0:
            current_hist = histogram_probs(finite_values.to_numpy(dtype=float), hist_edges)
            psi = calculate_psi(np.asarray(hist_probs, dtype=float), current_hist)
        else:
            psi = 0.0

        if finite_count > 0:
            above_upper = float((finite_values > upper_bound).mean())
            below_lower = float((finite_values < lower_bound).mean())
            max_over_q99 = float(finite_values.max() / max(abs(baseline_q99), 1e-6))
            mean_over_mean = float(finite_values.mean() / max(abs(baseline_mean), 1e-6))
            mean_abs_zscore = float(np.mean(np.abs(z.to_numpy(dtype=float))))
            max_abs_zscore = float(np.max(np.abs(z.to_numpy(dtype=float))))
        else:
            above_upper = 0.0
            below_lower = 0.0
            max_over_q99 = np.nan
            mean_over_mean = np.nan
            mean_abs_zscore = np.nan
            max_abs_zscore = np.nan

        if valid_aligned.shape[0] >= 2:
            diffs = valid_aligned.diff().abs().dropna()
            current_diff_mad = float(np.median(np.abs(diffs - np.median(diffs)))) if not diffs.empty else 0.0
        else:
            current_diff_mad = 0.0

        features.update(
            {
                "base_mean": baseline_mean,
                "base_std": baseline_std,
                "base_median": float(baseline.get("median", 0.0)),
                "base_q95": float(baseline.get("q95", 0.0)),
                "base_q99": baseline_q99,
                "base_mafd": baseline_mafd,
                "base_mean_over_mean": mean_over_mean,
                "base_max_over_q99": max_over_q99,
                "base_min_minus_lower": float(finite_values.min() - lower_bound) if finite_count > 0 else np.nan,
                "base_share_above_upper": above_upper,
                "base_share_below_lower": below_lower,
                "base_mean_abs_zscore": mean_abs_zscore,
                "base_max_abs_zscore": max_abs_zscore,
                "base_mafd_over_ref": float(features["acc_noise_mafd"] / max(baseline_mafd, 1e-6)),
                "base_diff_mad_over_ref": float(current_diff_mad / max(baseline_diff_mad, 1e-6)),
                "base_peak_ratio_over_ref": float(current_peak_ratio / max(abs(baseline_peak_ratio), 1e-6)),
                "base_hist_psi": float(psi),
                "base_autocorr_delta_lag1": float(features["dyn_autocorr_lag1"] - float((baseline.get("autocorr", {}) or {}).get("1", 0.0))),
                "base_autocorr_delta_lag24": float(features["dyn_autocorr_lag24"] - float((baseline.get("autocorr", {}) or {}).get("24", 0.0))),
            }
        )
    else:
        features.update(
            {
                "base_mean": np.nan,
                "base_std": np.nan,
                "base_median": np.nan,
                "base_q95": np.nan,
                "base_q99": np.nan,
                "base_mafd": np.nan,
                "base_mean_over_mean": np.nan,
                "base_max_over_q99": np.nan,
                "base_min_minus_lower": np.nan,
                "base_share_above_upper": np.nan,
                "base_share_below_lower": np.nan,
                "base_mean_abs_zscore": np.nan,
                "base_max_abs_zscore": np.nan,
                "base_mafd_over_ref": np.nan,
                "base_diff_mad_over_ref": np.nan,
                "base_peak_ratio_over_ref": np.nan,
                "base_hist_psi": np.nan,
                "base_autocorr_delta_lag1": np.nan,
                "base_autocorr_delta_lag24": np.nan,
            }
        )

    return {key: float(value) if pd.notna(value) else np.nan for key, value in features.items()}
