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

MISSING_TIMESTAMP_COL = "__dq_missing_timestamp__"
DUPLICATE_TIMESTAMP_COUNT_COL = "__dq_duplicate_timestamp_count__"


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


def _get_optional_numeric_column(
    dataset: WideDemandDataset,
    column_name: str,
    *,
    default: float = 0.0,
) -> pd.Series:
    if column_name not in dataset.frame.columns:
        return pd.Series(default, index=dataset.frame.index, dtype=float)
    return pd.to_numeric(dataset.frame[column_name], errors="coerce").fillna(default)


def _resolve_reference_time(clean_ts: pd.Series, config: dict) -> pd.Timestamp | pd.NaT:
    freshness_cfg = dict((config.get("freshness", {}) or {}))
    configured_ref = freshness_cfg.get("reference_time")
    if configured_ref is not None:
        ref = pd.to_datetime(configured_ref, errors="coerce")
        if pd.notna(ref):
            return ref
    return clean_ts.max()


def _normalize_metadata_label(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_client_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _load_client_metadata(config: dict) -> dict[str, dict[str, str]]:
    metadata_cfg = dict((config.get("metadata", {}) or {}))
    path = metadata_cfg.get("path")
    if not path:
        return {}
    meta_path = Path(path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Configured metadata.path does not exist: {meta_path}")
    frame = pd.read_csv(meta_path)
    client_col = metadata_cfg.get("client_id_column", "ID")
    category_col = metadata_cfg.get("category_column", "Categoria")
    subcategory_col = metadata_cfg.get("subcategory_column", "Subcategoria")
    missing = [col for col in (client_col, category_col, subcategory_col) if col not in frame.columns]
    if missing:
        raise ValueError(f"Metadata file {meta_path} is missing required columns: {missing}")
    out: dict[str, dict[str, str]] = {}
    for _, row in frame.iterrows():
        client_id = _normalize_client_id(row[client_col])
        if not client_id:
            continue
        out[client_id] = {
            "category": _normalize_metadata_label(row[category_col]),
            "subcategory": _normalize_metadata_label(row[subcategory_col]),
        }
    return out


def _mapping_lookup(mapping: object, key: str) -> Optional[float]:
    if not isinstance(mapping, dict) or not key:
        return None
    if key in mapping and mapping[key] is not None:
        return float(mapping[key])
    upper_key = key.upper()
    for raw_key, value in mapping.items():
        if str(raw_key).strip().upper() == upper_key and value is not None:
            return float(value)
    return None


def _resolve_metadata_threshold(
    *,
    config: dict,
    client_metadata: dict[str, dict[str, str]],
    client_id: str,
    global_key: str,
    category_key: str,
    subcategory_key: str,
) -> Optional[float]:
    meta = client_metadata.get(str(client_id), {})
    subcategory = meta.get("subcategory", "")
    category = meta.get("category", "")
    sub_value = _mapping_lookup(config.get(subcategory_key), subcategory)
    if sub_value is not None:
        return sub_value
    cat_value = _mapping_lookup(config.get(category_key), category)
    if cat_value is not None:
        return cat_value
    global_value = config.get(global_key)
    if global_value is not None:
        return float(global_value)
    return None


def _allow_clean_reference_fallback(config: dict) -> bool:
    thresholds_cfg = dict((config.get("thresholds", {}) or {}))
    return bool(thresholds_cfg.get("allow_clean_reference_fallback", False))


def _series_upper_bound(clean_series: pd.Series, cfg: dict) -> float:
    lower_bound = float(cfg.get("lower_bound", 0.0))
    upper_multiplier = float(cfg.get("upper_multiplier", 1.5))
    clean_valid = pd.to_numeric(clean_series, errors="coerce").dropna()
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
        current_series = _get_numeric_series(current, client_id)
        current_ts = _get_timestamp_series(current)
        expected_ts = expected_timestamp_index(clean.frame[clean.timestamp_col], clean.frequency)
        expected_count = int(len(expected_ts))
        valid_non_null_count = int((current_series.notna() & current_ts.notna()).sum())
        observed_ts = pd.DatetimeIndex(current_ts.dropna().unique())
        observed_covered = int(observed_ts.intersection(expected_ts).shape[0])
        missing_timestamp_flags = _get_optional_numeric_column(current, MISSING_TIMESTAMP_COL, default=0.0)
        missing_timestamp_count = int((missing_timestamp_flags > 0).sum())

        c1 = float(valid_non_null_count / max(1, expected_count)) if column_present else 0.0
        c2 = 1.0 if column_present and parseable_count > 0 else 0.0
        if column_present and diag.timestamp_col_present:
            effective_observed = max(0, observed_covered - missing_timestamp_count)
            c3 = float(effective_observed / max(1, expected_count))
        else:
            c3 = 0.0

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

    def __init__(self, config: Optional[dict] = None, client_metadata: Optional[dict[str, dict[str, str]]] = None):
        self.config = config or {}
        self.validity_cfg = dict((self.config.get("validity", {}) or {}))
        self.client_metadata = client_metadata or {}

    def score_client(
        self,
        current: WideDemandDataset,
        clean: WideDemandDataset,
        client_id: str,
    ) -> dict[str, float]:
        diag = current.diagnostics
        current_series = _get_numeric_series(current, client_id)
        column_present = bool(diag.client_column_present.get(client_id, False))
        observed_record_count = int(len(current.frame))

        if column_present:
            finite_values = int(np.isfinite(current_series.to_numpy(dtype=float, copy=False)).sum())
            v1 = float(finite_values / max(1, observed_record_count))
        else:
            v1 = 0.0

        v2 = float(diag.timestamp_valid_ratio) if diag.timestamp_col_present else 0.0

        lower_bound = float(self.validity_cfg.get("lower_bound", 0.0))
        upper_bound = _resolve_metadata_threshold(
            config=self.validity_cfg,
            client_metadata=self.client_metadata,
            client_id=client_id,
            global_key="upper_bound",
            category_key="upper_bound_by_category",
            subcategory_key="upper_bound_by_subcategory",
        )
        if upper_bound is None:
            if _allow_clean_reference_fallback(self.config):
                upper_bound = _series_upper_bound(_get_numeric_series(clean, client_id), self.validity_cfg)
            else:
                raise ValueError(
                    "Validity V3 requires a configured upper bound. Set validity.upper_bound, "
                    "validity.upper_bound_by_category, or validity.upper_bound_by_subcategory "
                    f"for client {client_id}; or explicitly set thresholds.allow_clean_reference_fallback=true."
                )
        valid_mask = current_series.notna()
        if valid_mask.any():
            in_range = current_series.loc[valid_mask].between(lower_bound, upper_bound, inclusive="both")
            v3 = float(in_range.mean())
        else:
            v3 = 0.0

        required_present = int(bool(diag.timestamp_col_present)) + int(bool(column_present))
        v4 = float(required_present / 2.0)

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

    def __init__(self, config: Optional[dict] = None, client_metadata: Optional[dict[str, dict[str, str]]] = None):
        self.config = config or {}
        self.consistency_cfg = dict((self.config.get("consistency", {}) or {}))
        self.client_metadata = client_metadata or {}

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

        if column_present and diag.timestamp_col_present:
            timestamps = _get_timestamp_series(current)
            valid_ts = timestamps.dropna().reset_index(drop=True)
            duplicate_counts = _get_optional_numeric_column(current, DUPLICATE_TIMESTAMP_COUNT_COL, default=0.0)
            duplicate_count = int(duplicate_counts.clip(lower=0).sum())
            if valid_ts.shape[0] <= 1:
                cs1 = 1.0 if valid_ts.shape[0] == 1 else 0.0
                cs2 = cs1
                if duplicate_count > 0:
                    cs1 = 0.0
                    cs2 = 0.0
            else:
                deltas = valid_ts.diff().dropna()
                tolerance = max(pd.Timedelta(seconds=1), current.frequency * 0.1)
                regular_count = int((deltas.sub(current.frequency).abs() <= tolerance).sum())
                positive_count = int((deltas > pd.Timedelta(0)).sum())
                total_deltas = int(len(deltas) + duplicate_count)
                cs1 = float(regular_count / max(1, total_deltas))
                cs2 = float(positive_count / max(1, total_deltas))
            observed_count = int(valid_ts.shape[0] + duplicate_count)
            cs3 = float(valid_ts.nunique() / max(1, observed_count))
        else:
            cs1 = 0.0
            cs2 = 0.0
            cs3 = 0.0

        valid = current_series.notna()
        if valid.any():
            freeze_mask = stuck_segment_mask(
                current_series,
                min_len=int(self.consistency_cfg.get("freeze_min_len", 6)),
                atol=float(self.consistency_cfg.get("freeze_atol", 0.0)),
            )
            frozen_ratio = float(freeze_mask.loc[valid].mean())
            allow_constant_series = bool(self.consistency_cfg.get("allow_constant_series", False))
            if allow_constant_series and frozen_ratio >= 0.999:
                cs4 = 1.0
            else:
                cs4 = float(max(0.0, 1.0 - frozen_ratio))
        else:
            cs4 = 0.0

        diffs = current_series.diff().abs()
        delta_max = _resolve_metadata_threshold(
            config=self.consistency_cfg,
            client_metadata=self.client_metadata,
            client_id=client_id,
            global_key="rate_of_change_threshold",
            category_key="rate_of_change_threshold_by_category",
            subcategory_key="rate_of_change_threshold_by_subcategory",
        )
        if delta_max is None:
            if _allow_clean_reference_fallback(self.config):
                clean_diffs = clean_series.diff().abs().dropna()
                if not clean_diffs.empty:
                    delta_max = float(clean_diffs.quantile(0.99) * float(self.consistency_cfg.get("roc_multiplier", 1.5)))
                else:
                    delta_max = float(self.consistency_cfg.get("roc_multiplier", 1.5))
            else:
                raise ValueError(
                    "Consistency CS5 requires a configured rate-of-change threshold. Set "
                    "consistency.rate_of_change_threshold, consistency.rate_of_change_threshold_by_category, "
                    "or consistency.rate_of_change_threshold_by_subcategory "
                    f"for client {client_id}; or explicitly set thresholds.allow_clean_reference_fallback=true."
                )
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

        a1 = self._score_local_anomaly(current_series)
        a2 = self._score_robust_deviation(current_series)
        a3 = self._score_historical_noise_instability(current_series)

        weights = dict((self.config.get("subcheck_weights", {}) or {}).get("accuracy", {}) or {})
        accuracy = _weighted_average({"A1": a1, "A2": a2, "A3": a3}, weights)
        return {"A1": a1, "A2": a2, "A3": a3, "accuracy": accuracy}

    def _score_local_anomaly(self, current_series: pd.Series) -> float:
        values = pd.to_numeric(current_series, errors="coerce")
        applicable = values.notna() & values.shift(1).notna() & values.shift(-1).notna()
        if int(applicable.sum()) == 0:
            return 1.0 if current_series.notna().any() else 0.0
        x_hat = (values.shift(1) + values.shift(-1)) / 2.0
        sigma_ref = max(
            safe_std(values.dropna().to_numpy(dtype=float)),
            float(self.accuracy_cfg.get("min_threshold_abs", 0.0)),
            1e-6,
        )
        kappa = float(
            self.accuracy_cfg.get(
                "local_spike_kappa",
                self.accuracy_cfg.get("seasonal_threshold_multiplier", 3.0),
            )
        )
        anomaly = (values - x_hat).abs() > (kappa * sigma_ref)
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

    def _score_historical_noise_instability(self, current_series: pd.Series) -> float:
        values = pd.to_numeric(current_series, errors="coerce")
        diffs = values.diff().abs().to_numpy(dtype=float)
        finite = np.isfinite(diffs)
        if int(finite.sum()) == 0:
            return 0.0 if values.notna().sum() == 0 else 1.0

        min_diffs = int(self.accuracy_cfg.get("historical_min_diffs", self.accuracy_cfg.get("min_history", 24)))
        lookback_steps = self.accuracy_cfg.get("historical_lookback_steps")
        quantile = float(self.accuracy_cfg.get("historical_diff_quantile", 0.95))
        quantile = min(1.0, max(0.0, quantile))
        epsilon = float(self.accuracy_cfg.get("epsilon", 1e-6))

        outcomes: list[bool] = []
        for idx in range(len(diffs)):
            current_diff = diffs[idx]
            if not np.isfinite(current_diff):
                continue
            start = 0 if lookback_steps is None else max(0, idx - int(lookback_steps))
            history = diffs[start:idx]
            history = history[np.isfinite(history)]
            if history.size < min_diffs:
                continue
            threshold = float(np.quantile(history, quantile))
            if threshold <= epsilon:
                threshold = max(float(np.mean(history)), epsilon)
            outcomes.append(bool(current_diff <= threshold))

        if not outcomes:
            return 1.0
        return float(np.mean(outcomes))


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

        ref_time = _resolve_reference_time(clean_ts, self.config)
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
            delay_tau = pd.to_timedelta(self.freshness_cfg.get("reporting_delay_tau", "24h"))
            deadline_raw = self.freshness_cfg.get("reporting_deadline_time")
            arrival_raw = self.freshness_cfg.get("reporting_arrival_time")
            deadline_time = pd.to_datetime(deadline_raw, errors="coerce") if deadline_raw is not None else pd.NaT
            arrival_time = pd.to_datetime(arrival_raw, errors="coerce") if arrival_raw is not None else pd.NaT
            if delay_tau <= pd.Timedelta(0) or pd.isna(deadline_time) or pd.isna(arrival_time):
                f3 = 1.0
            else:
                delay = max(pd.Timedelta(0), arrival_time - deadline_time)
                f3 = float(max(0.0, 1.0 - (delay / delay_tau)))

        weights = dict((self.config.get("subcheck_weights", {}) or {}).get("freshness", {}) or {})
        freshness = _weighted_average({"F1": f1, "F2": f2, "F3": f3}, weights)
        return {"F1": f1, "F2": f2, "F3": f3, "freshness": freshness}


class WideDemandQualityScorer:
    """Compute per-client quality scores for wide-format demand data."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.client_metadata = _load_client_metadata(self.config)
        self.completeness = CompletenessScorer(config=self.config)
        self.validity = ValidityScorer(config=self.config, client_metadata=self.client_metadata)
        self.consistency = ConsistencyScorer(config=self.config, client_metadata=self.client_metadata)
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
