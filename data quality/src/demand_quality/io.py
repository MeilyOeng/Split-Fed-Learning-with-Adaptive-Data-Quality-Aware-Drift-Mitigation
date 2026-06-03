"""I/O helpers for the demand quality pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml

from .schema import (
    WideDemandDataset,
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


def load_yaml_config(path: Path | str) -> dict:
    """Load a YAML config file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
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
    return WideDemandDataset(
        frame=frame,
        timestamp_col=ts_col,
        client_cols=client_cols,
        frequency=frequency,
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
