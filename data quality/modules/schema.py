"""Schema detection helpers for wide-format demand data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import pandas as pd
from pandas.tseries.frequencies import to_offset


DEFAULT_TIMESTAMP_CANDIDATES: Sequence[str] = (
    "timestamp",
    "date",
    "datetime",
    "time",
    "Fecha_Hora",
    "fecha_hora",
    "event_time",
    "ts",
)

DATA_QUALITY_METADATA_PREFIX = "__dq_"


@dataclass(frozen=True)
class WideDemandDataset:
    """In-memory representation of a wide demand table."""

    frame: pd.DataFrame
    timestamp_col: str
    client_cols: list[str]
    frequency: pd.Timedelta
    diagnostics: "WideDemandDiagnostics"


@dataclass(frozen=True)
class WideDemandDiagnostics:
    """Raw-load diagnostics preserved for later data-quality scoring."""

    raw_row_count: int
    timestamp_col_present: bool
    schema_required_columns_present: bool
    timestamp_valid_count: int
    timestamp_invalid_count: int
    timestamp_valid_ratio: float
    unique_timestamp_count: int
    duplicate_timestamp_count: int
    duplicate_timestamp_score: float
    monotonic_non_decreasing_score: float
    interval_regularity_score: float
    client_column_present: dict[str, bool]
    client_raw_non_null_count: dict[str, int]
    client_raw_non_null_ratio: dict[str, float]
    client_numeric_parseable_count: dict[str, int]
    client_numeric_parseable_ratio: dict[str, float]


def detect_timestamp_column(
    df: pd.DataFrame,
    preferred: Optional[str] = None,
    candidates: Sequence[str] = DEFAULT_TIMESTAMP_CANDIDATES,
) -> str:
    """Detect the timestamp column in a wide demand table."""
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"Timestamp column '{preferred}' not found.")
        return preferred

    lower_map = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        col = lower_map.get(candidate.lower())
        if col is not None:
            return col

    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return str(col)

    raise ValueError(
        "Could not detect a timestamp column. Pass --timestamp-col explicitly."
    )


def candidate_value_columns(
    df: pd.DataFrame,
    timestamp_col: str,
    ignore_cols: Optional[Iterable[str]] = None,
) -> list[str]:
    """Return non-timestamp, non-ignored columns that may contain client values."""
    ignore = {timestamp_col}
    if ignore_cols:
        ignore.update(ignore_cols)

    out: list[str] = []
    for col in df.columns:
        if col in ignore:
            continue
        if str(col).startswith("Unnamed:"):
            continue
        if str(col).startswith(DATA_QUALITY_METADATA_PREFIX):
            continue
        out.append(str(col))
    return out


def coerce_timestamp_series(series: pd.Series) -> pd.Series:
    """Parse a timestamp series while preserving local calendar semantics."""
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() == 0:
        raise ValueError("Timestamp column could not be parsed into datetimes.")
    return parsed


def detect_client_columns(
    df: pd.DataFrame,
    timestamp_col: str,
    ignore_cols: Optional[Iterable[str]] = None,
    min_numeric_fraction: float = 0.5,
) -> list[str]:
    """Detect numeric client columns in a wide demand table."""
    client_cols: list[str] = []
    for col in candidate_value_columns(df, timestamp_col=timestamp_col, ignore_cols=ignore_cols):
        if pd.api.types.is_numeric_dtype(df[col]):
            client_cols.append(str(col))
            continue

        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().mean() >= min_numeric_fraction:
            client_cols.append(str(col))

    if not client_cols:
        raise ValueError("No numeric client columns detected in demand dataset.")

    return client_cols


def infer_frequency(
    timestamps: pd.Series,
    configured_freq: Optional[str] = None,
    default: str = "1h",
) -> pd.Timedelta:
    """Infer or parse the expected sampling frequency."""
    if configured_freq:
        offset = to_offset(configured_freq)
        if getattr(offset, "nanos", 0) <= 0:
            raise ValueError(f"Configured frequency '{configured_freq}' is not fixed.")
        return pd.Timedelta(offset.nanos, unit="ns")

    ts = pd.Series(pd.to_datetime(timestamps, errors="coerce")).dropna().sort_values()
    ts = ts.drop_duplicates()
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return pd.to_timedelta(default)
    return diffs.mode().iloc[0]


def expected_timestamp_index(
    timestamps: pd.Series,
    frequency: pd.Timedelta,
) -> pd.DatetimeIndex:
    """Build the expected timestamp grid over the observed span."""
    ts = pd.Series(pd.to_datetime(timestamps, errors="coerce")).dropna().sort_values()
    if ts.empty:
        return pd.DatetimeIndex([])
    return pd.date_range(start=ts.iloc[0], end=ts.iloc[-1], freq=frequency)
