"""Per-client quality scoring for wide-format demand data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .io import WideDemandDataset, load_wide_demand_csv, save_csv
from .schema import expected_timestamp_index
from .statistics import (
    median_absolute_deviation,
    rolling_median_mad_anomaly_mask,
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
    V1: float
    V2: float
    V3: float
    V4: float
    validity: float
    CS1: float
    CS2: float
    CS3: float
    CS4: float
    CS5: float
    consistency: float
    A1: float
    A2: float
    A3: float
    accuracy: float
    F1: float
    F2: float
    F3: float
    freshness: float
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


def _get_numeric_series(dataset: WideDemandDataset, client_id: str) -> pd.Series:
    if client_id in dataset.frame.columns:
        return pd.to_numeric(dataset.frame[client_id], errors="coerce")
    return pd.Series(np.nan, index=dataset.frame.index, dtype=float)


def _get_timestamp_series(dataset: WideDemandDataset) -> pd.Series:
    return pd.to_datetime(dataset.frame[dataset.timestamp_col], errors="coerce")


def _series_upper_bound(clean_series: pd.Series, cfg: dict) -> float:
    lower_bound = float(cfg.get("lower_bound", 0.0))
    fixed_upper = cfg.get("upper_bound")
    upper_multiplier = float(cfg.get("upper_multiplier", 1.5))
    clean_valid = pd.to_numeric(clean_series, errors="coerce").dropna()
    if fixed_upper is not None:
        return float(fixed_upper)
    if not clean_valid.empty:
        return float(max(lower_bound, clean_valid.quantile(0.999) * upper_multiplier))
    return float(max(lower_bound, upper_multiplier))


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
        diag = current.diagnostics
        column_present = bool(diag.client_column_present.get(client_id, False))
        parseable_count = int(diag.client_numeric_parseable_count.get(client_id, 0))

        c1 = float(diag.client_raw_non_null_ratio.get(client_id, 0.0)) if column_present else 0.0
        c2 = 1.0 if column_present and parseable_count > 0 else 0.0

        expected_count = len(expected_timestamp_index(clean.frame[clean.timestamp_col], clean.frequency))
        coverage = float(diag.unique_timestamp_count / max(1, expected_count))
        c3 = float(min(1.0, coverage)) if column_present and diag.timestamp_col_present else 0.0

        weights = dict((self.config.get("subcheck_weights", {}) or {}).get("completeness", {}) or {})
        completeness = _weighted_average({"C1": c1, "C2": c2, "C3": c3}, weights)
        return {
            "C1": c1,
            "C2": c2,
            "C3": c3,
            "completeness": completeness,
        }


class ValidityScorer:
    """Validity scoring for wide-format client data."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.validity_cfg = dict((self.config.get("validity", {}) or {}))

    def score_client(
        self,
        current: WideDemandDataset,
        clean: WideDemandDataset,
        client_id: str,
    ) -> dict[str, float]:
        diag = current.diagnostics
        current_series = _get_numeric_series(current, client_id)
        clean_series = _get_numeric_series(clean, client_id)
        column_present = bool(diag.client_column_present.get(client_id, False))

        if column_present:
            v1 = float(diag.client_numeric_parseable_ratio.get(client_id, 1.0))
        else:
            v1 = 0.0

        v2 = float(diag.timestamp_valid_ratio) if diag.timestamp_col_present else 0.0

        lower_bound = float(self.validity_cfg.get("lower_bound", 0.0))
        upper_bound = _series_upper_bound(clean_series, self.validity_cfg)
        valid_mask = current_series.notna()
        if valid_mask.any():
            in_range = current_series.loc[valid_mask].between(lower_bound, upper_bound, inclusive="both")
            v3 = float(in_range.mean())
        else:
            v3 = 0.0

        v4 = 1.0 if (diag.timestamp_col_present and diag.schema_required_columns_present and column_present) else 0.0

        weights = dict((self.config.get("subcheck_weights", {}) or {}).get("validity", {}) or {})
        validity = _weighted_average({"V1": v1, "V2": v2, "V3": v3, "V4": v4}, weights)
        return {
            "V1": v1,
            "V2": v2,
            "V3": v3,
            "V4": v4,
            "validity": validity,
        }


class ConsistencyScorer:
    """Consistency scoring for time ordering and dynamic behavior."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.consistency_cfg = dict((self.config.get("consistency", {}) or {}))

    def score_client(
        self,
        current: WideDemandDataset,
        clean: WideDemandDataset,
        client_id: str,
    ) -> dict[str, float]:
        diag = current.diagnostics
        current_series = _get_numeric_series(current, client_id)
        clean_series = _get_numeric_series(clean, client_id)
        column_present = bool(diag.client_column_present.get(client_id, False))

        cs1 = float(diag.interval_regularity_score) if column_present and diag.timestamp_col_present else 0.0
        cs2 = float(diag.monotonic_non_decreasing_score) if column_present and diag.timestamp_col_present else 0.0
        cs3 = float(diag.duplicate_timestamp_score) if column_present and diag.timestamp_col_present else 0.0

        valid = current_series.notna()
        if valid.any():
            freeze_mask = stuck_segment_mask(
                current_series,
                min_len=int(self.consistency_cfg.get("freeze_min_len", 6)),
                atol=float(self.consistency_cfg.get("freeze_atol", 0.0)),
            )
            cs4 = float(max(0.0, 1.0 - freeze_mask.loc[valid].mean()))
        else:
            cs4 = 0.0

        diffs = current_series.diff().abs()
        clean_diffs = clean_series.diff().abs().dropna()
        fixed_delta = self.consistency_cfg.get("rate_of_change_threshold")
        if fixed_delta is not None:
            delta_max = float(fixed_delta)
        elif not clean_diffs.empty:
            delta_max = float(clean_diffs.quantile(0.99) * float(self.consistency_cfg.get("roc_multiplier", 1.5)))
        else:
            delta_max = float(self.consistency_cfg.get("roc_multiplier", 1.5))
        applicable = diffs.notna() & valid & valid.shift(1, fill_value=False)
        if applicable.any():
            cs5 = float((diffs.loc[applicable] <= delta_max).mean())
        else:
            cs5 = 1.0 if valid.any() else 0.0

        weights = dict((self.config.get("subcheck_weights", {}) or {}).get("consistency", {}) or {})
        consistency = _weighted_average(
            {"CS1": cs1, "CS2": cs2, "CS3": cs3, "CS4": cs4, "CS5": cs5},
            weights,
        )
        return {
            "CS1": cs1,
            "CS2": cs2,
            "CS3": cs3,
            "CS4": cs4,
            "CS5": cs5,
            "consistency": consistency,
        }


class AccuracyScorer:
    """Accuracy and plausibility scoring for local abnormal behavior."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.accuracy_cfg = dict((self.config.get("accuracy", {}) or {}))

    def score_client(
        self,
        current: WideDemandDataset,
        clean: WideDemandDataset,
        client_id: str,
    ) -> dict[str, float]:
        current_series = _get_numeric_series(current, client_id)
        clean_series = _get_numeric_series(clean, client_id)

        a1 = self._score_local_anomaly(current_series)
        a2 = self._score_robust_deviation(current_series)
        a3 = self._score_noise_instability(current_series, clean_series)

        weights = dict((self.config.get("subcheck_weights", {}) or {}).get("accuracy", {}) or {})
        accuracy = _weighted_average({"A1": a1, "A2": a2, "A3": a3}, weights)
        return {"A1": a1, "A2": a2, "A3": a3, "accuracy": accuracy}

    def _score_local_anomaly(self, current_series: pd.Series) -> float:
        method = str(self.accuracy_cfg.get("local_anomaly_method", "seasonal_naive")).lower()
        if method == "rolling_median":
            anomaly, applicable, _ = rolling_median_mad_anomaly_mask(
                current_series,
                window=int(self.accuracy_cfg.get("rolling_window", 24)),
                mad_multiplier=float(self.accuracy_cfg.get("mad_multiplier", 4.0)),
                min_history=int(self.accuracy_cfg.get("min_history", 6)),
                min_threshold_abs=float(self.accuracy_cfg.get("min_threshold_abs", 0.0)),
            )
        else:
            anomaly, applicable, _ = seasonal_naive_anomaly_mask(
                current_series,
                period=int(self.accuracy_cfg.get("seasonal_period", 24)),
                threshold_multiplier=float(self.accuracy_cfg.get("seasonal_threshold_multiplier", 3.0)),
                min_threshold_abs=float(self.accuracy_cfg.get("min_threshold_abs", 0.0)),
            )

        if int(applicable.sum()) == 0:
            return 1.0 if current_series.notna().any() else 0.0
        return float(max(0.0, 1.0 - anomaly.loc[applicable].mean()))

    def _score_robust_deviation(self, current_series: pd.Series) -> float:
        anomaly, applicable, _ = rolling_median_mad_anomaly_mask(
            current_series,
            window=int(self.accuracy_cfg.get("rolling_window", 24)),
            mad_multiplier=float(self.accuracy_cfg.get("mad_multiplier", 4.0)),
            min_history=int(self.accuracy_cfg.get("min_history", 6)),
            min_threshold_abs=float(self.accuracy_cfg.get("min_threshold_abs", 0.0)),
        )
        if int(applicable.sum()) == 0:
            return 1.0 if current_series.notna().any() else 0.0
        return float(max(0.0, 1.0 - anomaly.loc[applicable].mean()))

    def _score_noise_instability(self, current_series: pd.Series, clean_series: pd.Series) -> float:
        values = pd.to_numeric(current_series, errors="coerce")
        clean_values = pd.to_numeric(clean_series, errors="coerce")
        cur_diffs = values.diff().dropna().to_numpy(dtype=float)
        clean_diffs = clean_values.diff().dropna().to_numpy(dtype=float)
        cur_diffs = cur_diffs[np.isfinite(cur_diffs)]
        clean_diffs = clean_diffs[np.isfinite(clean_diffs)]
        if cur_diffs.size == 0:
            return 0.0 if values.notna().sum() == 0 else 1.0

        cur_noise = median_absolute_deviation(cur_diffs)
        if cur_noise <= 0:
            cur_noise = safe_std(cur_diffs)
        ref_noise = median_absolute_deviation(clean_diffs)
        if ref_noise <= 0:
            ref_noise = safe_std(clean_diffs)
        epsilon = float(self.accuracy_cfg.get("epsilon", 1e-6))
        ref_noise = max(ref_noise, epsilon)
        ratio = cur_noise / ref_noise
        allowed_ratio = float(self.accuracy_cfg.get("allowed_noise_ratio", 2.0))
        excess = max(0.0, ratio - 1.0)
        return float(max(0.0, 1.0 - min(1.0, excess / max(allowed_ratio, epsilon))))


class FreshnessScorer:
    """Freshness scoring for recency-based quality checks."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.freshness_cfg = dict((self.config.get("freshness", {}) or {}))

    def score_client(
        self,
        current: WideDemandDataset,
        clean: WideDemandDataset,
        client_id: str,
    ) -> dict[str, float]:
        current_series = _get_numeric_series(current, client_id)
        current_ts = _get_timestamp_series(current)
        clean_ts = _get_timestamp_series(clean)
        valid = current_series.notna() & current_ts.notna()

        ref_time = clean_ts.max()
        if valid.any() and pd.notna(ref_time):
            latest_valid_ts = current_ts.loc[valid].max()
            age = ref_time - latest_valid_ts
            tau = pd.to_timedelta(self.freshness_cfg.get("staleness_tau", "48h"))
            f1 = float(max(0.0, 1.0 - (age / tau))) if tau > pd.Timedelta(0) else 1.0
        else:
            f1 = 0.0

        recency_window = pd.to_timedelta(self.freshness_cfg.get("window_recency_span", "168h"))
        if pd.isna(ref_time) or recency_window <= pd.Timedelta(0):
            f2 = 0.0
        else:
            recent_start = ref_time - recency_window
            expected_recent = expected_timestamp_index(clean.frame[clean.timestamp_col], clean.frequency)
            expected_recent_count = int(((expected_recent >= recent_start) & (expected_recent <= ref_time)).sum())
            current_recent_valid = valid & (current_ts >= recent_start) & (current_ts <= ref_time)
            f2 = float(current_recent_valid.sum() / max(1, expected_recent_count))
            f2 = float(min(1.0, max(0.0, f2)))

        reporting_delay_enabled = bool(self.freshness_cfg.get("reporting_delay_enabled", False))
        if not reporting_delay_enabled:
            f3 = 1.0
        else:
            reporting_ref = self.freshness_cfg.get("reporting_reference_time")
            delay_tau = pd.to_timedelta(self.freshness_cfg.get("reporting_delay_tau", "24h"))
            if reporting_ref is None or delay_tau <= pd.Timedelta(0) or not valid.any():
                f3 = 1.0
            else:
                reference_time = pd.to_datetime(reporting_ref, errors="coerce")
                if pd.isna(reference_time):
                    f3 = 1.0
                else:
                    latest_valid_ts = current_ts.loc[valid].max()
                    delay = reference_time - latest_valid_ts
                    f3 = float(max(0.0, 1.0 - (delay / delay_tau)))

        weights = dict((self.config.get("subcheck_weights", {}) or {}).get("freshness", {}) or {})
        freshness = _weighted_average({"F1": f1, "F2": f2, "F3": f3}, weights)
        return {"F1": f1, "F2": f2, "F3": f3, "freshness": freshness}


class WideDemandQualityScorer:
    """Compute per-client quality scores for wide-format demand data."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.completeness = CompletenessScorer(config=self.config)
        self.validity = ValidityScorer(config=self.config)
        self.consistency = ConsistencyScorer(config=self.config)
        self.accuracy = AccuracyScorer(config=self.config)
        self.freshness = FreshnessScorer(config=self.config)

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
            val = self.validity.score_client(current, clean, client_id)
            cons = self.consistency.score_client(current, clean, client_id)
            acc = self.accuracy.score_client(current, clean, client_id)
            fresh = self.freshness.score_client(current, clean, client_id)
            sample_count = int(_get_numeric_series(current, client_id).notna().sum())
            rows.append(
                DimensionScores(
                    client_id=client_id,
                    C1=comp["C1"],
                    C2=comp["C2"],
                    C3=comp["C3"],
                    completeness=comp["completeness"],
                    V1=val["V1"],
                    V2=val["V2"],
                    V3=val["V3"],
                    V4=val["V4"],
                    validity=val["validity"],
                    CS1=cons["CS1"],
                    CS2=cons["CS2"],
                    CS3=cons["CS3"],
                    CS4=cons["CS4"],
                    CS5=cons["CS5"],
                    consistency=cons["consistency"],
                    A1=acc["A1"],
                    A2=acc["A2"],
                    A3=acc["A3"],
                    accuracy=acc["accuracy"],
                    F1=fresh["F1"],
                    F2=fresh["F2"],
                    F3=fresh["F3"],
                    freshness=fresh["freshness"],
                    sample_count=sample_count,
                )
            )

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(
                columns=[
                    "client_id",
                    "C1",
                    "C2",
                    "C3",
                    "completeness",
                    "V1",
                    "V2",
                    "V3",
                    "V4",
                    "validity",
                    "CS1",
                    "CS2",
                    "CS3",
                    "CS4",
                    "CS5",
                    "consistency",
                    "A1",
                    "A2",
                    "A3",
                    "accuracy",
                    "F1",
                    "F2",
                    "F3",
                    "freshness",
                    "sample_count",
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
