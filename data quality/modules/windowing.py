"""Demand-specific client-window construction for surrogate-model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from .io import WideDemandDataset, load_wide_demand_csv, load_wide_demand_dataframe
from .schema import expected_timestamp_index


DQ_METADATA_PREFIX = "__dq_"


@dataclass(frozen=True)
class DemandWindowSource:
    """Shared source state for per-client window generation."""

    clean_raw: pd.DataFrame
    current_raw: pd.DataFrame
    clean_dataset: WideDemandDataset
    current_dataset: WideDemandDataset
    timestamp_col: str
    frequency: pd.Timedelta
    expected_grid: pd.DatetimeIndex
    parsed_clean_ts: pd.Series
    parsed_current_ts: pd.Series
    clients: tuple[str, ...]


@dataclass(frozen=True)
class ClientWindowSlice:
    """One client-window with raw and standardized views."""

    client_id: str
    window_id: str
    window_index: int
    window_start_index: int
    window_end_index: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    timestamp_col: str
    frequency: pd.Timedelta
    expected_timestamps: pd.DatetimeIndex
    clean_raw_window: pd.DataFrame
    current_raw_window: pd.DataFrame
    clean_raw_history: pd.DataFrame
    current_raw_history: pd.DataFrame
    clean_window: WideDemandDataset
    current_window: WideDemandDataset


def _selected_clients(
    clean_dataset: WideDemandDataset,
    current_dataset: WideDemandDataset,
    selected_clients: Optional[Sequence[str]] = None,
) -> tuple[str, ...]:
    if selected_clients:
        requested = {str(client_id) for client_id in selected_clients}
        clean_clients = [client_id for client_id in clean_dataset.client_cols if str(client_id) in requested]
        return tuple(clean_clients)
    clean_clients = set(map(str, clean_dataset.client_cols))
    current_clients = set(map(str, current_dataset.client_cols))
    return tuple(sorted(clean_clients & current_clients if current_clients else clean_clients))


def load_window_source(
    clean_path: Path | str,
    current_path: Path | str,
    *,
    timestamp_col: Optional[str] = None,
    configured_freq: Optional[str] = None,
    selected_clients: Optional[Sequence[str]] = None,
) -> DemandWindowSource:
    """Load clean/current wide demand inputs once for downstream window iteration."""
    normalized_freq = _frequency_string(configured_freq) if configured_freq is not None else None
    clean_dataset = load_wide_demand_csv(
        clean_path,
        timestamp_col=timestamp_col,
        configured_freq=normalized_freq,
    )
    current_dataset = load_wide_demand_csv(
        current_path,
        timestamp_col=timestamp_col or clean_dataset.timestamp_col,
        configured_freq=normalized_freq or _frequency_string(clean_dataset.frequency),
    )
    timestamp_name = clean_dataset.timestamp_col
    clean_raw = pd.read_csv(Path(clean_path))
    current_raw = pd.read_csv(Path(current_path))
    parsed_clean_ts = pd.to_datetime(clean_raw[timestamp_name], errors="coerce")
    parsed_current_ts = pd.to_datetime(current_raw[timestamp_name], errors="coerce")
    expected_grid = expected_timestamp_index(clean_dataset.frame[timestamp_name], clean_dataset.frequency)
    clients = _selected_clients(clean_dataset, current_dataset, selected_clients=selected_clients)
    return DemandWindowSource(
        clean_raw=clean_raw,
        current_raw=current_raw,
        clean_dataset=clean_dataset,
        current_dataset=current_dataset,
        timestamp_col=timestamp_name,
        frequency=clean_dataset.frequency,
        expected_grid=expected_grid,
        parsed_clean_ts=parsed_clean_ts,
        parsed_current_ts=parsed_current_ts,
        clients=clients,
    )


def _slice_raw_client_frame(
    raw_frame: pd.DataFrame,
    parsed_ts: pd.Series,
    *,
    timestamp_col: str,
    client_id: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    fallback_grid: pd.DatetimeIndex,
) -> pd.DataFrame:
    cols = [timestamp_col]
    if client_id in raw_frame.columns:
        cols.append(client_id)
    cols.extend([col for col in raw_frame.columns if str(col).startswith(DQ_METADATA_PREFIX)])
    subset = raw_frame.loc[(parsed_ts >= window_start) & (parsed_ts <= window_end), cols].copy()
    if client_id not in subset.columns:
        subset[client_id] = np.nan
    if subset.empty:
        data = {
            timestamp_col: pd.Series(fallback_grid, dtype="datetime64[ns]"),
            client_id: np.nan,
        }
        for col in raw_frame.columns:
            if str(col).startswith(DQ_METADATA_PREFIX):
                data[str(col)] = np.nan
        subset = pd.DataFrame(data)
    return subset


def _build_standardized_client_window(
    raw_window: pd.DataFrame,
    *,
    timestamp_col: str,
    configured_freq: str,
) -> WideDemandDataset:
    return load_wide_demand_dataframe(
        raw_window,
        timestamp_col=timestamp_col,
        configured_freq=configured_freq,
    )


def _frequency_string(freq: pd.Timedelta | str) -> str:
    if isinstance(freq, pd.Timedelta):
        return to_offset(freq).freqstr
    return str(freq)


def iter_client_windows(
    source: DemandWindowSource,
    *,
    window_size_steps: int,
    stride_steps: int = 1,
) -> Iterator[ClientWindowSlice]:
    """Yield one standardized client-window at a time."""
    if window_size_steps <= 0:
        raise ValueError("window_size_steps must be positive.")
    if stride_steps <= 0:
        raise ValueError("stride_steps must be positive.")
    if len(source.expected_grid) < window_size_steps:
        return

    configured_freq = _frequency_string(source.frequency)
    for client_id in source.clients:
        for end_idx in range(window_size_steps - 1, len(source.expected_grid), stride_steps):
            start_idx = end_idx - window_size_steps + 1
            window_ts = source.expected_grid[start_idx : end_idx + 1]
            window_start = pd.Timestamp(window_ts[0])
            window_end = pd.Timestamp(window_ts[-1])
            clean_raw_window = _slice_raw_client_frame(
                source.clean_raw,
                source.parsed_clean_ts,
                timestamp_col=source.timestamp_col,
                client_id=client_id,
                window_start=window_start,
                window_end=window_end,
                fallback_grid=window_ts,
            )
            current_raw_window = _slice_raw_client_frame(
                source.current_raw,
                source.parsed_current_ts,
                timestamp_col=source.timestamp_col,
                client_id=client_id,
                window_start=window_start,
                window_end=window_end,
                fallback_grid=window_ts,
            )
            clean_history = _slice_raw_client_frame(
                source.clean_raw,
                source.parsed_clean_ts,
                timestamp_col=source.timestamp_col,
                client_id=client_id,
                window_start=pd.Timestamp(source.expected_grid[0]),
                window_end=window_end,
                fallback_grid=source.expected_grid[: end_idx + 1],
            )
            current_history = _slice_raw_client_frame(
                source.current_raw,
                source.parsed_current_ts,
                timestamp_col=source.timestamp_col,
                client_id=client_id,
                window_start=pd.Timestamp(source.expected_grid[0]),
                window_end=window_end,
                fallback_grid=source.expected_grid[: end_idx + 1],
            )
            clean_window = _build_standardized_client_window(
                clean_raw_window,
                timestamp_col=source.timestamp_col,
                configured_freq=configured_freq,
            )
            current_window = _build_standardized_client_window(
                current_raw_window,
                timestamp_col=source.timestamp_col,
                configured_freq=configured_freq,
            )
            window_id = (
                f"{client_id}__"
                f"{window_start.isoformat().replace(':', '-')}"
                f"__{window_end.isoformat().replace(':', '-')}"
            )
            yield ClientWindowSlice(
                client_id=str(client_id),
                window_id=window_id,
                window_index=int(end_idx),
                window_start_index=int(start_idx),
                window_end_index=int(end_idx),
                window_start=window_start,
                window_end=window_end,
                timestamp_col=source.timestamp_col,
                frequency=source.frequency,
                expected_timestamps=window_ts,
                clean_raw_window=clean_raw_window,
                current_raw_window=current_raw_window,
                clean_raw_history=clean_history,
                current_raw_history=current_history,
                clean_window=clean_window,
                current_window=current_window,
            )


def collect_client_windows(
    source: DemandWindowSource,
    *,
    window_size_steps: int,
    stride_steps: int = 1,
) -> list[ClientWindowSlice]:
    """Materialize all client-windows into a list."""
    return list(
        iter_client_windows(
            source,
            window_size_steps=window_size_steps,
            stride_steps=stride_steps,
        )
    )
