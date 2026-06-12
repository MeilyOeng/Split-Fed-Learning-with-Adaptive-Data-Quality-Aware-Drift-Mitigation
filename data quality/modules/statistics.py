"""Statistics helpers for demand fault injection and scoring."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def safe_mean(values: pd.Series | np.ndarray) -> float:
    """Return the mean of finite values, or 0.0 when unavailable."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def safe_std(values: pd.Series | np.ndarray, ddof: int = 0) -> float:
    """Return the std of finite values, or 0.0 when unavailable."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(arr.std(ddof=ddof))


def median_absolute_deviation(values: pd.Series | np.ndarray) -> float:
    """Compute the unscaled MAD of finite values."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def make_histogram_bin_edges(
    reference_values: pd.Series | np.ndarray,
    bins: int = 20,
    strategy: str = "quantile",
) -> np.ndarray:
    """Create stable histogram edges from reference values."""
    ref = np.asarray(reference_values, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.array([0.0, 1.0], dtype=float)

    if np.allclose(ref.min(), ref.max()):
        center = float(ref[0])
        pad = max(abs(center) * 0.1, 1.0)
        return np.array([center - pad, center + pad], dtype=float)

    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(ref, quantiles)
        edges = np.unique(edges)
        if edges.size >= 2:
            return edges

    return np.linspace(ref.min(), ref.max(), bins + 1, dtype=float)


def histogram_probs(
    values: pd.Series | np.ndarray,
    bin_edges: Sequence[float],
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Convert values to a smoothed histogram probability vector."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    num_bins = max(1, len(bin_edges) - 1)
    counts = np.zeros(num_bins, dtype=float)
    if arr.size > 0:
        counts, _ = np.histogram(arr, bins=np.asarray(bin_edges, dtype=float))
    counts = counts + float(epsilon)
    return counts / counts.sum()


def calculate_psi(
    expected_probs: np.ndarray,
    actual_probs: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """Calculate PSI between two discrete distributions."""
    exp = np.asarray(expected_probs, dtype=float) + float(epsilon)
    act = np.asarray(actual_probs, dtype=float) + float(epsilon)
    exp = exp / exp.sum()
    act = act / act.sum()
    return float(np.sum((act - exp) * np.log(act / exp)))


def calculate_psi_from_values(
    expected_values: pd.Series | np.ndarray,
    actual_values: pd.Series | np.ndarray,
    bin_edges: Sequence[float],
    epsilon: float = 1e-6,
) -> float:
    """Calculate PSI directly from value arrays."""
    expected_probs = histogram_probs(expected_values, bin_edges, epsilon=epsilon)
    actual_probs = histogram_probs(actual_values, bin_edges, epsilon=epsilon)
    return calculate_psi(expected_probs, actual_probs, epsilon=epsilon)


def autocorrelation_at_lags(
    series: pd.Series,
    lags: Iterable[int],
) -> dict[int, float]:
    """Compute autocorrelation at specific lags for one numeric series."""
    values = pd.to_numeric(series, errors="coerce")
    out: dict[int, float] = {}
    for lag in lags:
        lag_int = int(lag)
        if lag_int <= 0:
            continue
        try:
            corr = values.autocorr(lag=lag_int)
        except Exception:
            corr = np.nan
        if np.isfinite(corr):
            out[lag_int] = float(corr)
    return out


def segment_mean_std(
    series: pd.Series,
    n_segments: int,
) -> float:
    """Compute the std of segment means over contiguous non-null values."""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return 0.0
    n_segments = max(1, int(n_segments))
    parts = [part for part in np.array_split(values, n_segments) if len(part) > 0]
    if len(parts) <= 1:
        return 0.0
    means = np.asarray([float(np.mean(part)) for part in parts], dtype=float)
    return safe_std(means)


def rolling_median_mad_anomaly_mask(
    series: pd.Series,
    window: int,
    mad_multiplier: float,
    min_history: int = 6,
    min_threshold_abs: float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Detect anomalies using a trailing rolling median and MAD."""
    values = pd.to_numeric(series, errors="coerce")
    history = values.shift(1)
    expected = history.rolling(window=window, min_periods=min_history).median()

    def _mad(arr: pd.Series) -> float:
        return median_absolute_deviation(arr.to_numpy())

    mad = history.rolling(window=window, min_periods=min_history).apply(_mad, raw=False)
    threshold = ((mad_multiplier * mad) / 0.6745).clip(lower=min_threshold_abs)
    applicable = values.notna() & expected.notna()
    anomaly = pd.Series(False, index=series.index)
    anomaly.loc[applicable] = (
        (values.loc[applicable] - expected.loc[applicable]).abs()
        > threshold.loc[applicable].fillna(min_threshold_abs)
    )
    return anomaly, applicable, expected


def seasonal_naive_anomaly_mask(
    series: pd.Series,
    period: int,
    threshold_multiplier: float,
    min_threshold_abs: float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Detect anomalies using a seasonal-naive baseline."""
    values = pd.to_numeric(series, errors="coerce")
    expected = values.shift(period)
    residual = (values - expected).abs()
    ref_scale = median_absolute_deviation(residual.dropna())
    threshold = max(min_threshold_abs, threshold_multiplier * ref_scale)
    applicable = values.notna() & expected.notna()
    anomaly = pd.Series(False, index=series.index)
    anomaly.loc[applicable] = residual.loc[applicable] > threshold
    return anomaly, applicable, expected


def stuck_segment_mask(
    series: pd.Series,
    min_len: int,
    atol: float = 0.0,
) -> pd.Series:
    """Mark points that belong to long, nearly constant runs."""
    values = pd.to_numeric(series, errors="coerce")
    valid = values.notna()
    if not valid.any():
        return pd.Series(False, index=series.index)

    same_as_prev = values.diff().abs().le(atol)
    same_as_prev = same_as_prev.fillna(False) & valid
    group_start = (~same_as_prev) | (~valid)
    group_id = group_start.cumsum()
    group_sizes = valid.groupby(group_id).transform("sum")
    return (group_sizes >= int(min_len)) & valid


def calendar_coverage_score(
    timestamps: pd.Series,
    valid_mask: pd.Series,
    n_min: int,
) -> float:
    """Score hour-of-day x day-of-week coverage."""
    ts = pd.Series(pd.to_datetime(timestamps, errors="coerce"))
    mask = valid_mask.astype(bool) & ts.notna()
    if not mask.any():
        return 0.0
    subset = ts.loc[mask]
    counts = subset.groupby([subset.dt.dayofweek, subset.dt.hour]).size()
    covered = int((counts >= int(n_min)).sum())
    return covered / 168.0


def peak_offpeak_ratio(
    values: pd.Series,
    timestamps: pd.Series,
    peak_hours: Iterable[int],
    epsilon: float = 1e-6,
) -> float:
    """Compute peak/off-peak mean ratio."""
    vals = pd.to_numeric(values, errors="coerce")
    ts = pd.Series(pd.to_datetime(timestamps, errors="coerce"))
    mask = vals.notna() & ts.notna()
    if not mask.any():
        return 0.0

    vals = vals.loc[mask]
    ts = ts.loc[mask]
    peak_hours_set = {int(hour) for hour in peak_hours}
    peak_mask = ts.dt.hour.isin(peak_hours_set)
    if peak_mask.sum() == 0 or (~peak_mask).sum() == 0:
        return 0.0
    peak_mean = float(vals.loc[peak_mask].mean())
    off_peak_mean = float(vals.loc[~peak_mask].mean())
    return peak_mean / max(off_peak_mean, epsilon)
