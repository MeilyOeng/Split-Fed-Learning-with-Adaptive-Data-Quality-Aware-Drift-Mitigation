"""I/O helpers for the demand quality pipeline."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .schema import (
    WideDemandDataset,
    WideDemandDiagnostics,
    candidate_value_columns,
    coerce_timestamp_series,
    detect_client_columns,
    detect_timestamp_column,
    infer_frequency,
)


def normalize_consumption_sign(
    frame: pd.DataFrame,
    client_cols: Iterable[str],
) -> pd.DataFrame:
    """Normalize predominantly negative demand columns to positive consumption.

    The raw smart-meter dataset stores demand as negative values. This helper
    applies ``y = -y`` to client columns that are predominantly negative so the
    rest of the pipeline operates on positive consumption values.

    The normalization is intentionally idempotent at the dataset level: columns
    that are already mostly positive are left unchanged. This prevents double
    flipping when fault-injected CSVs are reloaded later for scoring.
    """
    normalized = frame.copy(deep=False)
    for col in client_cols:
        values = pd.to_numeric(normalized[col], errors="coerce")
        finite = values.dropna()
        if finite.empty:
            normalized[col] = values
            continue
        negative_fraction = float((finite < 0).mean())
        if negative_fraction > 0.5:
            normalized[col] = -values
        else:
            normalized[col] = values
    return normalized


def _pairwise_score(mask: pd.Series, success_mask: pd.Series) -> float:
    """Return a score over valid adjacent pairs."""
    valid_pairs = int(mask.sum())
    if valid_pairs == 0:
        return 1.0
    return float(success_mask.loc[mask].mean())


def build_wide_demand_diagnostics(
    raw_frame: pd.DataFrame,
    *,
    timestamp_col: str,
    frequency: pd.Timedelta,
    ignore_cols: Optional[Iterable[str]] = None,
) -> WideDemandDiagnostics:
    """Compute raw-load diagnostics needed for later quality scoring."""
    raw_row_count = int(len(raw_frame))
    raw_ts = pd.to_datetime(raw_frame[timestamp_col], errors="coerce")
    timestamp_valid = raw_ts.notna()
    timestamp_valid_count = int(timestamp_valid.sum())
    timestamp_invalid_count = int(raw_row_count - timestamp_valid_count)
    timestamp_valid_ratio = float(timestamp_valid_count / max(1, raw_row_count))

    valid_ts = raw_ts.loc[timestamp_valid].reset_index(drop=True)
    unique_timestamp_count = int(valid_ts.nunique())
    duplicate_timestamp_count = int(valid_ts.duplicated(keep=False).sum())
    duplicate_timestamp_score = (
        1.0
        if timestamp_valid_count == 0
        else float(max(0.0, 1.0 - (duplicate_timestamp_count / max(1, timestamp_valid_count))))
    )

    deltas = valid_ts.diff()
    comparable = deltas.notna()
    monotonic_non_decreasing_score = _pairwise_score(
        comparable,
        deltas.fillna(pd.Timedelta(0)) >= pd.Timedelta(0),
    )
    interval_regularity_score = _pairwise_score(
        comparable,
        deltas.fillna(pd.Timedelta(0)) == frequency,
    )

    value_columns = candidate_value_columns(
        raw_frame,
        timestamp_col=timestamp_col,
        ignore_cols=ignore_cols,
    )
    client_column_present = {str(col): True for col in value_columns}
    client_raw_non_null_count: dict[str, int] = {}
    client_raw_non_null_ratio: dict[str, float] = {}
    client_numeric_parseable_count: dict[str, int] = {}
    client_numeric_parseable_ratio: dict[str, float] = {}

    for col in value_columns:
        raw_series = raw_frame[col]
        non_null_mask = raw_series.notna()
        non_null_count = int(non_null_mask.sum())
        client_raw_non_null_count[str(col)] = non_null_count
        client_raw_non_null_ratio[str(col)] = float(non_null_count / max(1, raw_row_count))

        if non_null_count == 0:
            parseable_count = 0
            parseable_ratio = 1.0
        else:
            parseable = pd.to_numeric(raw_series.loc[non_null_mask], errors="coerce").notna()
            parseable_count = int(parseable.sum())
            parseable_ratio = float(parseable_count / non_null_count)
        client_numeric_parseable_count[str(col)] = parseable_count
        client_numeric_parseable_ratio[str(col)] = parseable_ratio

    return WideDemandDiagnostics(
        raw_row_count=raw_row_count,
        timestamp_col_present=timestamp_col in raw_frame.columns,
        schema_required_columns_present=(timestamp_col in raw_frame.columns and len(value_columns) > 0),
        timestamp_valid_count=timestamp_valid_count,
        timestamp_invalid_count=timestamp_invalid_count,
        timestamp_valid_ratio=timestamp_valid_ratio,
        unique_timestamp_count=unique_timestamp_count,
        duplicate_timestamp_count=duplicate_timestamp_count,
        duplicate_timestamp_score=duplicate_timestamp_score,
        monotonic_non_decreasing_score=monotonic_non_decreasing_score,
        interval_regularity_score=interval_regularity_score,
        client_column_present=client_column_present,
        client_raw_non_null_count=client_raw_non_null_count,
        client_raw_non_null_ratio=client_raw_non_null_ratio,
        client_numeric_parseable_count=client_numeric_parseable_count,
        client_numeric_parseable_ratio=client_numeric_parseable_ratio,
    )


def load_yaml_config(path: Path | str) -> dict:
    """Load a YAML config file."""
    config_path = Path(path)
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except ModuleNotFoundError:
        # Fall back to the system Python when the active project venv does not
        # include PyYAML but a machine-wide install is available.
        script = (
            "import json, sys, yaml; "
            "print(json.dumps(yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8')) or {}))"
        )
        proc = subprocess.run(
            ["python3", "-c", script, str(config_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(proc.stdout or "{}")
    if not isinstance(data, dict):
        raise ValueError(f"Config at {config_path} must contain a YAML mapping.")
    return data


def load_wide_demand_dataframe(
    df: pd.DataFrame,
    timestamp_col: Optional[str] = None,
    ignore_cols: Optional[Iterable[str]] = None,
    configured_freq: Optional[str] = None,
    min_numeric_fraction: float = 0.5,
) -> WideDemandDataset:
    """Standardize a wide demand DataFrame.

    After timestamp parsing and client detection, demand columns are normalized
    to positive consumption values. Raw smart-meter inputs that are
    predominantly negative are converted via ``y = -y``.
    """
    if df.empty:
        raise ValueError("Input demand DataFrame is empty.")

    raw_frame = df.copy(deep=True)
    frame = df.copy(deep=True)
    ts_col = detect_timestamp_column(frame, preferred=timestamp_col)
    frame[ts_col] = coerce_timestamp_series(frame[ts_col])
    frame = frame.dropna(subset=[ts_col]).sort_values(ts_col, kind="mergesort").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No rows remain after dropping invalid timestamps.")

    client_cols = detect_client_columns(
        frame,
        timestamp_col=ts_col,
        ignore_cols=ignore_cols,
        min_numeric_fraction=min_numeric_fraction,
    )
    for col in client_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = normalize_consumption_sign(frame, client_cols)

    frequency = infer_frequency(frame[ts_col], configured_freq=configured_freq)
    diagnostics = build_wide_demand_diagnostics(
        raw_frame,
        timestamp_col=ts_col,
        frequency=frequency,
        ignore_cols=ignore_cols,
    )
    return WideDemandDataset(
        frame=frame,
        timestamp_col=ts_col,
        client_cols=client_cols,
        frequency=frequency,
        diagnostics=diagnostics,
    )


def load_wide_demand_csv(
    path: Path | str,
    timestamp_col: Optional[str] = None,
    ignore_cols: Optional[Iterable[str]] = None,
    configured_freq: Optional[str] = None,
    min_numeric_fraction: float = 0.5,
) -> WideDemandDataset:
    """Load and standardize a wide-format demand CSV."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    return load_wide_demand_dataframe(
        df,
        timestamp_col=timestamp_col,
        ignore_cols=ignore_cols,
        configured_freq=configured_freq,
        min_numeric_fraction=min_numeric_fraction,
    )


def save_csv(df: pd.DataFrame, path: Path | str) -> None:
    """Persist a DataFrame to CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def save_json(data: dict, path: Path | str) -> None:
    """Persist a JSON artifact."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
