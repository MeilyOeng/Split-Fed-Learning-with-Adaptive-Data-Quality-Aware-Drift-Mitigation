"""Per-client quality scoring for wide-format demand data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from common.utils import calculate_jsd

from .io import WideDemandDataset, load_wide_demand_csv, save_csv
from .schema import expected_timestamp_index
from .statistics import (
    autocorrelation_at_lags,
    calendar_coverage_score,
    make_histogram_bin_edges,
    peak_offpeak_ratio,
    rolling_median_mad_anomaly_mask,
    segment_mean_std,
    safe_mean,
    safe_std,
    seasonal_naive_anomaly_mask,
    stuck_segment_mask,
)


@dataclass(frozen=True)
class DimensionScores:
    """Scores for one client across all subchecks and dimensions."""

    client_id: str
    C1: float
    C2: float
    C3: float
    completeness: float
    A1: float
    A2: float
    A3: float
    accuracy: float
    TC1: float
    TC2: float
    TC3: float
    TC4: float
    temporal_consistency: float
    SK1: float
    SK2: float
    SK3: float
    skewness: float
    sample_count: int


def _weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = 0.0
    total = 0.0
    for key, score in scores.items():
        weight = float(weights.get(key, 1.0))
        total += weight * float(score)
        total_weight += weight
    if total_weight <= 0:
        return 0.0
    return float(total / total_weight)


class CompletenessScorer:
    """Completeness scoring for one wide-format client series."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def score_client(
        self,
        current: WideDemandDataset,
        clean: WideDemandDataset,
        client_id: str,
    ) -> dict[str, float]:
        series = current.frame[client_id] if client_id in current.frame.columns else pd.Series(np.nan, index=current.frame.index)
        valid = series.notna()
        sample_count = int(valid.sum())
        c1 = float(sample_count / max(1, len(series)))
        c2 = 1.0 if client_id in current.frame.columns and sample_count > 0 else 0.0

        expected_index = expected_timestamp_index(current.frame[current.timestamp_col], current.frequency)
        expected_intervals = max(1, len(expected_index) - 1)
        ts = pd.to_datetime(current.frame[current.timestamp_col], errors="coerce")
        deltas = ts.diff()
        matched = valid & valid.shift(1, fill_value=False) & (deltas == current.frequency)
        c3 = float(matched.sum() / expected_intervals)

        weights = dict((self.config.get("subcheck_weights", {}) or {}).get("completeness", {}) or {})
        completeness = _weighted_average({"C1": c1, "C2": c2, "C3": c3}, weights)
        return {"C1": c1, "C2": c2, "C3": c3, "completeness": completeness, "sample_count": sample_count}


class AccuracyScorer:
    """Accuracy scoring for one wide-format client series."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.accuracy_cfg = dict((self.config.get("accuracy", {}) or {}))

    def score_client(
        self,
        current: WideDemandDataset,
        clean: WideDemandDataset,
        client_id: str,
    ) -> dict[str, float]:
        current_series = current.frame[client_id] if client_id in current.frame.columns else pd.Series(np.nan, index=current.frame.index)
        clean_series = clean.frame[client_id] if client_id in clean.frame.columns else pd.Series(np.nan, index=clean.frame.index)

        lower_bound = float(self.accuracy_cfg.get("lower_bound", 0.0))
        fixed_upper = self.accuracy_cfg.get("upper_bound")
        upper_multiplier = float(self.accuracy_cfg.get("upper_multiplier", 1.5))
        clean_valid = pd.to_numeric(clean_series, errors="coerce").dropna()
        if fixed_upper is not None:
            upper_bound = float(fixed_upper)
        elif not clean_valid.empty:
            upper_bound = float(clean_valid.quantile(0.999) * upper_multiplier)
        else:
            upper_bound = float(upper_multiplier)

        current_valid = pd.to_numeric(current_series, errors="coerce")
        valid_mask = current_valid.notna()
        if valid_mask.any():
            in_range = current_valid[valid_mask].between(lower_bound, upper_bound, inclusive="both")
            a1 = float(in_range.mean())
        else:
            a1 = 0.0

        a2 = self._score_local_anomaly(current_valid)
        a3 = self._score_context_rules(current_valid, clean_valid)
        weights = dict((self.config.get("subcheck_weights", {}) or {}).get("accuracy", {}) or {})
        accuracy = _weighted_average({"A1": a1, "A2": a2, "A3": a3}, weights)
        return {"A1": a1, "A2": a2, "A3": a3, "accuracy": accuracy}

    def _score_local_anomaly(self, current_series: pd.Series) -> float:
        method = str(self.accuracy_cfg.get("deviation_method", "rolling_median")).lower()
        if method == "seasonal_naive":
            period = int(self.accuracy_cfg.get("seasonal_period", 24))
            anomaly, applicable, _ = seasonal_naive_anomaly_mask(
                current_series,
                period=period,
                threshold_multiplier=float(self.accuracy_cfg.get("seasonal_threshold_multiplier", 3.0)),
                min_threshold_abs=float(self.accuracy_cfg.get("min_threshold_abs", 0.0)),
            )
        else:
            anomaly, applicable, _ = rolling_median_mad_anomaly_mask(
                current_series,
                window=int(self.accuracy_cfg.get("rolling_window", 24)),
                mad_multiplier=float(self.accuracy_cfg.get("mad_multiplier", 4.0)),
                min_history=int(self.accuracy_cfg.get("min_history", 6)),
                min_threshold_abs=float(self.accuracy_cfg.get("min_threshold_abs", 0.0)),
            )

        if int(applicable.sum()) == 0:
            return 1.0 if current_series.notna().any() else 0.0
        anomaly_fraction = float(anomaly.loc[applicable].mean())
        return float(max(0.0, 1.0 - anomaly_fraction))

    def _score_context_rules(self, current_series: pd.Series, clean_valid: pd.Series) -> float:
        values = pd.to_numeric(current_series, errors="coerce")
        valid = values.notna()
        if not valid.any():
            return 0.0

        nonneg = float((values.loc[valid] >= 0).mean())

        diffs = values.diff().abs()
        clean_diffs = clean_valid.diff().abs().dropna()
        fixed_delta = self.accuracy_cfg.get("rate_of_change_threshold")
        if fixed_delta is not None:
            delta_max = float(fixed_delta)
        elif not clean_diffs.empty:
            delta_max = float(clean_diffs.quantile(0.99) * float(self.accuracy_cfg.get("roc_multiplier", 1.5)))
        else:
            delta_max = float(self.accuracy_cfg.get("roc_multiplier", 1.5))
        roc_applicable = diffs.notna() & valid & valid.shift(1, fill_value=False)
        if roc_applicable.any():
            roc = float((diffs.loc[roc_applicable] <= delta_max).mean())
        else:
            roc = 1.0

        freeze_mask = stuck_segment_mask(
            values,
            min_len=int(self.accuracy_cfg.get("max_stuck_len", 6)),
            atol=float(self.accuracy_cfg.get("freeze_atol", 0.0)),
        )
        freeze = float(max(0.0, 1.0 - freeze_mask.loc[valid].mean()))
        return float((nonneg + roc + freeze) / 3.0)


class TemporalConsistencyScorer:
    """Temporal consistency scoring for one client series."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.tc_cfg = dict((self.config.get("temporal_consistency", {}) or {}))

    def score_client(
        self,
        current: WideDemandDataset,
        clean: WideDemandDataset,
        client_id: str,
    ) -> dict[str, float]:
        current_series = current.frame[client_id] if client_id in current.frame.columns else pd.Series(np.nan, index=current.frame.index)
        clean_series = clean.frame[client_id] if client_id in clean.frame.columns else pd.Series(np.nan, index=clean.frame.index)
        current_valid = pd.to_numeric(current_series, errors="coerce")
        clean_valid = pd.to_numeric(clean_series, errors="coerce")
        valid_mask = current_valid.notna()

        tau = pd.to_timedelta(self.tc_cfg.get("staleness_tau", "48h"))
        if valid_mask.any():
            latest_valid_ts = pd.to_datetime(current.frame.loc[valid_mask, current.timestamp_col], errors="coerce").max()
            ref_time = pd.to_datetime(current.frame[current.timestamp_col], errors="coerce").max()
            age = ref_time - latest_valid_ts
            tc1 = float(max(0.0, 1.0 - age / tau))
        else:
            tc1 = 0.0

        epsilon = float(self.tc_cfg.get("epsilon", 1e-6))
        lags = list(self.tc_cfg.get("autocorr_lags", [1, 24, 168]))
        autocorr_threshold = float(self.tc_cfg.get("autocorr_threshold", 0.25))
        ref_autocorr = autocorrelation_at_lags(clean_valid, lags)
        cur_autocorr = autocorrelation_at_lags(current_valid, lags)
        common_lags = [lag for lag in lags if int(lag) in ref_autocorr and int(lag) in cur_autocorr]
        if common_lags:
            diffs = [
                abs(cur_autocorr[int(lag)] - ref_autocorr[int(lag)])
                for lag in common_lags
            ]
            mean_diff = float(np.mean(diffs))
            tc2 = float(max(0.0, 1.0 - min(1.0, mean_diff / max(autocorr_threshold, epsilon))))
        else:
            tc2 = 1.0 if current_valid.notna().any() and clean_valid.notna().any() else 0.0

        tc3 = calendar_coverage_score(
            current.frame[current.timestamp_col],
            valid_mask,
            n_min=int(self.tc_cfg.get("calendar_n_min", 1)),
        )

        n_segments = int(self.tc_cfg.get("stationarity_segments", 4))
        stationarity_tolerance = float(self.tc_cfg.get("stationarity_tolerance", 1.0))
        sigma_ref = segment_mean_std(clean_valid, n_segments=n_segments)
        sigma_cur = segment_mean_std(current_valid, n_segments=n_segments)
        if current_valid.notna().any() and clean_valid.notna().any():
            ref_scale = max(sigma_ref, safe_std(clean_valid), epsilon)
            excess = max(0.0, sigma_cur - sigma_ref)
            tc4 = float(max(0.0, 1.0 - min(1.0, excess / max(ref_scale * stationarity_tolerance, epsilon))))
        else:
            tc4 = 0.0

        weights = dict((self.config.get("subcheck_weights", {}) or {}).get("temporal_consistency", {}) or {})
        temporal = _weighted_average({"TC1": tc1, "TC2": tc2, "TC3": tc3, "TC4": tc4}, weights)
        return {"TC1": tc1, "TC2": tc2, "TC3": tc3, "TC4": tc4, "temporal_consistency": temporal}


class SkewnessScorer:
    """Distribution-shift scoring for one client series."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.sk_cfg = dict((self.config.get("skewness", {}) or {}))

    def score_client(
        self,
        current: WideDemandDataset,
        clean: WideDemandDataset,
        client_id: str,
    ) -> dict[str, float]:
        current_series = current.frame[client_id] if client_id in current.frame.columns else pd.Series(np.nan, index=current.frame.index)
        clean_series = clean.frame[client_id] if client_id in clean.frame.columns else pd.Series(np.nan, index=clean.frame.index)
        current_valid = pd.to_numeric(current_series, errors="coerce").dropna()
        clean_valid = pd.to_numeric(clean_series, errors="coerce").dropna()
        if current_valid.empty or clean_valid.empty:
            sk1 = 0.0
            sk2 = 0.0
            sk3 = 0.0
        else:
            bins = int(self.sk_cfg.get("jsd_bins", 20))
            strategy = str(self.sk_cfg.get("histogram_strategy", "quantile"))
            epsilon = float(self.sk_cfg.get("epsilon", 1e-6))
            edges = make_histogram_bin_edges(clean_valid.to_numpy(dtype=float), bins=bins, strategy=strategy)
            ref_probs = np.histogram(clean_valid.to_numpy(dtype=float), bins=edges)[0].astype(float) + epsilon
            cur_probs = np.histogram(current_valid.to_numpy(dtype=float), bins=edges)[0].astype(float) + epsilon
            jsd = calculate_jsd(ref_probs, cur_probs, base=np.e)
            sk1 = float(max(0.0, 1.0 - min(1.0, jsd / np.log(2.0))))

            peak_hours = list(self.sk_cfg.get("peak_hours", [17, 18, 19, 20, 21]))
            cur_ratio = peak_offpeak_ratio(current_series, current.frame[current.timestamp_col], peak_hours, epsilon=epsilon)
            ref_ratio = peak_offpeak_ratio(clean_series, clean.frame[clean.timestamp_col], peak_hours, epsilon=epsilon)
            allowed_shift = float(self.sk_cfg.get("allowed_peak_ratio_shift", 0.5))
            sk2 = float(max(0.0, 1.0 - min(1.0, abs(cur_ratio - ref_ratio) / max(allowed_shift, epsilon))))

            mean_shift = abs(safe_mean(current_valid) - safe_mean(clean_valid)) / max(safe_std(clean_valid), epsilon)
            std_shift = abs(safe_std(current_valid) - safe_std(clean_valid)) / max(safe_std(clean_valid), epsilon)
            threshold = float(self.sk_cfg.get("moment_shift_threshold", 3.0))
            sk3 = float(max(0.0, 1.0 - min(1.0, (mean_shift + std_shift) / max(threshold, epsilon))))

        weights = dict((self.config.get("subcheck_weights", {}) or {}).get("skewness", {}) or {})
        skewness = _weighted_average({"SK1": sk1, "SK2": sk2, "SK3": sk3}, weights)
        return {"SK1": sk1, "SK2": sk2, "SK3": sk3, "skewness": skewness}


class WideDemandQualityScorer:
    """Compute per-client quality scores for wide-format demand data."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.completeness = CompletenessScorer(config=self.config)
        self.accuracy = AccuracyScorer(config=self.config)
        self.temporal = TemporalConsistencyScorer(config=self.config)
        self.skewness = SkewnessScorer(config=self.config)

    def score(
        self,
        current: WideDemandDataset,
        clean: WideDemandDataset,
    ) -> pd.DataFrame:
        """Score all selected clients."""
        selected = self._select_clients(current, clean)
        rows: list[DimensionScores] = []
        for client_id in selected:
            comp = self.completeness.score_client(current, clean, client_id)
            acc = self.accuracy.score_client(current, clean, client_id)
            temp = self.temporal.score_client(current, clean, client_id)
            skew = self.skewness.score_client(current, clean, client_id)
            rows.append(
                DimensionScores(
                    client_id=client_id,
                    C1=comp["C1"],
                    C2=comp["C2"],
                    C3=comp["C3"],
                    completeness=comp["completeness"],
                    A1=acc["A1"],
                    A2=acc["A2"],
                    A3=acc["A3"],
                    accuracy=acc["accuracy"],
                    TC1=temp["TC1"],
                    TC2=temp["TC2"],
                    TC3=temp["TC3"],
                    TC4=temp["TC4"],
                    temporal_consistency=temp["temporal_consistency"],
                    SK1=skew["SK1"],
                    SK2=skew["SK2"],
                    SK3=skew["SK3"],
                    skewness=skew["skewness"],
                    sample_count=int(comp["sample_count"]),
                )
            )

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(
                columns=[
                    "client_id", "C1", "C2", "C3", "completeness",
                    "A1", "A2", "A3", "accuracy",
                    "TC1", "TC2", "TC3", "TC4", "temporal_consistency",
                    "SK1", "SK2", "SK3", "skewness", "sample_count",
                ]
            )
        return out.sort_values("client_id", kind="mergesort").reset_index(drop=True)

    def _select_clients(self, current: WideDemandDataset, clean: WideDemandDataset) -> list[str]:
        client_cfg = dict(self.config.get("clients", {}) or {})
        selected = list(client_cfg.get("selected", []) or [])
        if selected:
            return selected
        return sorted(set(current.client_cols) | set(clean.client_cols))


def score_wide_demand_quality(
    current_path: Path | str,
    clean_reference_path: Path | str,
    output_path: Path | str,
    config: Optional[dict] = None,
    timestamp_col: Optional[str] = None,
) -> pd.DataFrame:
    """Score wide-format demand quality from CSV inputs and save results."""
    config = config or {}
    scoring_cfg = dict(config)
    configured_freq = scoring_cfg.get("frequency")
    current = load_wide_demand_csv(current_path, timestamp_col=timestamp_col, configured_freq=configured_freq)
    clean = load_wide_demand_csv(clean_reference_path, timestamp_col=timestamp_col, configured_freq=configured_freq)
    scorer = WideDemandQualityScorer(config=scoring_cfg)
    scores_df = scorer.score(current, clean)
    save_csv(scores_df, output_path)
    return scores_df
