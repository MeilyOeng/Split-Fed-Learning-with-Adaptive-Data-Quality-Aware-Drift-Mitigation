"""Fault injection for wide-format demand data."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .io import load_wide_demand_csv, save_csv
from .schema import WideDemandDataset
from .statistics import safe_std

SIMPLE_FAULT_TYPES = (
    "missing",
    "invalid_high",
    "negative_value",
    "replayed_value",
)

SIMPLE_FAULT_META = {
    "missing": ("completeness", "C1"),
    "invalid_high": ("validity", "V3"),
    "negative_value": ("validity", "V3"),
    "replayed_value": ("consistency", "CS4"),
    "noise": ("accuracy", "A3"),
    "spike": ("accuracy", "A1"),
    # Legacy aliases kept for backward-compatible parsing.
    "out_of_range": ("validity", "V3"),
    "inconsistent": ("validity", "V3"),
    "timeliness": ("consistency", "CS4"),
}

_ORIG_ROW_ID_COL = "__dq_orig_row_id__"
_IS_DUPLICATE_COL = "__dq_is_duplicate__"
_MISSING_TIMESTAMP_COL = "__dq_missing_timestamp__"
_DUPLICATE_TIMESTAMP_COUNT_COL = "__dq_duplicate_timestamp_count__"


@dataclass
class FaultEvent:
    """A single fault-injection event."""

    client_id: str
    fault_type: str
    dimension: str
    subcheck: str
    start_time: Any
    end_time: Any
    affected_rows: int
    affected_fraction: float
    parameters_used: str
    cluster_id: str | None = None
    scope: str = "client"
    shared_timestamp_event: bool = False
    window_start: Any = None
    window_end: Any = None
    window_start_index: int | None = None
    window_end_index: int | None = None
    affected_positions: str | None = None
    affected_timestamps: str | None = None
    seed: int | None = None


def _sample_positions(
    valid_positions: np.ndarray,
    rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if rate <= 0 or valid_positions.size == 0:
        return np.array([], dtype=int)
    count = max(1, int(round(valid_positions.size * rate)))
    count = min(count, valid_positions.size)
    return np.sort(rng.choice(valid_positions, size=count, replace=False))


def _sample_blocks(
    n_rows: int,
    count: int,
    min_length: int,
    max_length: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    blocks: list[np.ndarray] = []
    if n_rows <= 0 or count <= 0:
        return blocks
    for _ in range(count):
        length = int(rng.integers(max(1, min_length), max(1, max_length) + 1))
        length = min(length, n_rows)
        start = int(rng.integers(0, max(1, n_rows - length + 1)))
        blocks.append(np.arange(start, start + length, dtype=int))
    return blocks


def _sample_single_block(
    n_rows: int,
    min_length: int,
    max_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks = _sample_blocks(
        n_rows=n_rows,
        count=1,
        min_length=min_length,
        max_length=max_length,
        rng=rng,
    )
    if not blocks:
        return np.array([], dtype=int)
    return blocks[0]


def _series_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    std = safe_std(finite)
    if std > 0:
        return std
    q05 = np.quantile(finite, 0.05)
    q95 = np.quantile(finite, 0.95)
    return float(max(abs(q95 - q05), abs(np.nanmean(finite)), 1.0))


def _json_params(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def _values_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return (np.isnan(a_arr) & np.isnan(b_arr)) | np.isclose(a_arr, b_arr, equal_nan=False)


def _resolve_fault_config(config: dict, key: str) -> dict:
    return dict((config.get("faults", {}) or {}).get(key, {}) or {})


def _resolve_drift_config(config: dict, key: str) -> dict:
    return dict((config.get("drift", {}) or {}).get(key, {}) or {})


def _normalize_identifier(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _normalize_identifier_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (str, int, float)):
        text = _normalize_identifier(raw)
        return [text] if text else []
    if isinstance(raw, Sequence):
        out: list[str] = []
        for item in raw:
            text = _normalize_identifier(item)
            if text:
                out.append(text)
        return out
    text = _normalize_identifier(raw)
    return [text] if text else []


def _load_cluster_assignment_map(
    path: Path | str,
    *,
    client_id_column: str = "ID",
    cluster_id_column: str = "Cluster",
) -> dict[str, str]:
    assignment_path = Path(path)
    if not assignment_path.exists():
        raise FileNotFoundError(f"Cluster assignment CSV not found: {assignment_path}")

    assignments = pd.read_csv(assignment_path).copy()
    required_cols = {client_id_column, cluster_id_column}
    if not required_cols.issubset(assignments.columns):
        missing = sorted(required_cols - set(assignments.columns))
        raise ValueError(
            f"Cluster assignment CSV at {assignment_path} is missing required columns: {missing}"
        )

    assignments[client_id_column] = assignments[client_id_column].map(_normalize_identifier)
    assignments[cluster_id_column] = assignments[cluster_id_column].map(_normalize_identifier)
    assignments = assignments.drop_duplicates(subset=[client_id_column], keep="first").reset_index(drop=True)
    return {
        str(client_id): str(cluster_id)
        for client_id, cluster_id in zip(assignments[client_id_column], assignments[cluster_id_column])
        if str(client_id) and str(cluster_id)
    }


class WideDemandFaultInjector:
    """Injects structured faults into a wide-format demand table."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.last_controlled_window_summary_df = pd.DataFrame()

    def inject(
        self,
        dataset: WideDemandDataset,
        seed: int = 42,
        cluster_assignment_path: Optional[Path | str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Inject configured faults into a wide demand dataset."""
        rng = np.random.default_rng(seed)
        frame = dataset.frame.copy(deep=True)
        frame[_ORIG_ROW_ID_COL] = np.arange(len(frame), dtype=int)
        frame[_IS_DUPLICATE_COL] = False
        ts_col = dataset.timestamp_col
        all_dataset_clients = list(map(str, dataset.client_cols))
        client_cols = all_dataset_clients if self.config.get("fault_mode") == "controlled_window" else self._select_clients(dataset.client_cols, rng)
        for client_id in dataset.client_cols:
            frame[client_id] = pd.to_numeric(frame[client_id], errors="coerce").astype(float)
        original_frame = frame.copy(deep=True)
        n_rows = len(frame)
        timestamps = frame[ts_col].reset_index(drop=True)

        if self.config.get("fault_mode") == "controlled_window":
            return self._inject_controlled_window_mode(
                frame=frame,
                original_frame=original_frame,
                client_cols=client_cols,
                all_client_cols=all_dataset_clients,
                ts_col=ts_col,
                timestamps=timestamps,
                seed=seed,
                rng=rng,
                cluster_assignment_path=cluster_assignment_path,
            )

        if self.config.get("fault_fraction") is not None:
            return self._inject_simple_fraction_mode(
                frame=frame,
                original_frame=original_frame,
                client_cols=client_cols,
                ts_col=ts_col,
                timestamps=timestamps,
                seed=seed,
                rng=rng,
            )

        events: list[FaultEvent] = []
        shared_drop_rows: set[int] = set()
        shared_duplicate_rows: list[int] = []

        completeness_cfg = _resolve_fault_config(self.config, "completeness")
        validity_cfg = _resolve_fault_config(self.config, "validity")
        consistency_cfg = _resolve_fault_config(self.config, "consistency")
        accuracy_cfg = _resolve_fault_config(self.config, "accuracy")
        freshness_cfg = _resolve_fault_config(self.config, "freshness")
        drift_distribution_cfg = _resolve_drift_config(self.config, "distribution_shift")
        drift_temporal_cfg = _resolve_drift_config(self.config, "temporal_pattern_shift")
        drift_regime_cfg = _resolve_drift_config(self.config, "regime_shift")
        cluster_assignment_cfg = dict(self.config.get("cluster_assignment", {}) or {})
        cluster_shared_noise_cfg = dict((accuracy_cfg.get("cluster_shared_noise", {}) or {}))
        cluster_noise_clusters = _normalize_identifier_list(
            cluster_shared_noise_cfg.get("clusters", cluster_shared_noise_cfg.get("cluster"))
        )
        cluster_noise_enabled = bool(cluster_shared_noise_cfg.get("enabled", False) or cluster_noise_clusters)
        if cluster_shared_noise_cfg.get("enabled", False) and not cluster_noise_clusters:
            raise ValueError(
                "accuracy.cluster_shared_noise.enabled is true but no clusters were configured."
            )
        cluster_assignment_map: dict[str, str] = {}
        if cluster_noise_enabled:
            resolved_assignment_path = cluster_assignment_path or cluster_assignment_cfg.get("path")
            if resolved_assignment_path is None:
                raise ValueError(
                    "Cluster-shared noise requires a cluster assignment CSV. "
                    "Pass --cluster-assignment or set cluster_assignment.path in the fault config."
                )
            cluster_assignment_map = _load_cluster_assignment_map(
                resolved_assignment_path,
                client_id_column=str(cluster_assignment_cfg.get("client_id_column", "ID")),
                cluster_id_column=str(cluster_assignment_cfg.get("cluster_id_column", "Cluster")),
            )
            cluster_shared_noise_cfg["clusters"] = cluster_noise_clusters

        shared_drop_rate = float(completeness_cfg.get("dropped_timestamp_rows_fraction", 0.0))
        if shared_drop_rate > 0:
            drop_count = min(n_rows, max(1, int(round(n_rows * shared_drop_rate))))
            shared_drop_rows.update(map(int, rng.choice(n_rows, size=drop_count, replace=False)))
            idxs = np.array(sorted(shared_drop_rows), dtype=int)
            events.append(
                self._make_event(
                    client_id="__all_clients__",
                    fault_type="dropped_timestamp_rows",
                    dimension="completeness",
                    subcheck="C3",
                    idxs=idxs,
                    timestamps=timestamps,
                    n_rows=n_rows,
                    parameters={"fraction": shared_drop_rate},
                    shared=True,
                )
            )

        duplicate_rate = float(consistency_cfg.get("duplicated_timestamp_fraction", 0.0))
        if duplicate_rate > 0:
            dup_count = min(n_rows, max(1, int(round(n_rows * duplicate_rate))))
            shared_duplicate_rows = list(map(int, rng.choice(n_rows, size=dup_count, replace=False)))
            idxs = np.array(sorted(shared_duplicate_rows), dtype=int)
            events.append(
                self._make_event(
                    client_id="__all_clients__",
                    fault_type="duplicated_timestamps",
                    dimension="consistency",
                    subcheck="CS3",
                    idxs=idxs,
                    timestamps=timestamps,
                    n_rows=n_rows,
                    parameters={"fraction": duplicate_rate},
                    shared=True,
                )
            )

        global_rate = float(self.config.get("global_fault_rate", 1.0))
        per_client_map = dict(self.config.get("per_client_fault_rate", {}) or {})

        for client_id in client_cols:
            client_factor = global_rate * float(per_client_map.get(client_id, 1.0))
            values = pd.to_numeric(frame[client_id], errors="coerce").to_numpy(dtype=float, copy=True)
            valid_positions = np.flatnonzero(np.isfinite(values))
            scale = _series_scale(values)
            abs_q99 = float(np.nanquantile(np.abs(values[np.isfinite(values)]), 0.99)) if valid_positions.size else 1.0
            upper_anchor = max(abs_q99, scale, 1.0)

            self._inject_validity_faults(
                frame,
                client_id,
                values,
                valid_positions,
                scale,
                upper_anchor,
                validity_cfg,
                client_factor,
                rng,
                timestamps,
                events,
            )
            values = pd.to_numeric(frame[client_id], errors="coerce").to_numpy(dtype=float, copy=True)
            valid_positions = np.flatnonzero(np.isfinite(values))

            self._inject_consistency_faults(
                frame,
                client_id,
                values,
                valid_positions,
                consistency_cfg,
                client_factor,
                rng,
                timestamps,
                events,
            )
            values = pd.to_numeric(frame[client_id], errors="coerce").to_numpy(dtype=float, copy=True)
            valid_positions = np.flatnonzero(np.isfinite(values))

            self._inject_accuracy_faults(
                frame,
                client_id,
                values,
                valid_positions,
                scale,
                accuracy_cfg,
                client_factor,
                rng,
                timestamps,
                events,
            )
            values = pd.to_numeric(frame[client_id], errors="coerce").to_numpy(dtype=float, copy=True)
            valid_positions = np.flatnonzero(np.isfinite(values))

            self._inject_completeness_faults(
                frame,
                client_id,
                values,
                valid_positions,
                completeness_cfg,
                client_factor,
                rng,
                timestamps,
                events,
            )

            values = pd.to_numeric(frame[client_id], errors="coerce").to_numpy(dtype=float, copy=True)
            valid_positions = np.flatnonzero(np.isfinite(values))

            self._inject_freshness_faults(
                frame,
                client_id,
                values,
                valid_positions,
                freshness_cfg,
                client_factor,
                rng,
                timestamps,
                events,
            )

            values = pd.to_numeric(frame[client_id], errors="coerce").to_numpy(dtype=float, copy=True)
            valid_positions = np.flatnonzero(np.isfinite(values))

            self._inject_drift_faults(
                frame,
                client_id,
                values,
                valid_positions,
                scale,
                drift_distribution_cfg,
                drift_temporal_cfg,
                drift_regime_cfg,
                client_factor,
                rng,
                timestamps,
                events,
            )

        if cluster_noise_enabled:
            self._inject_cluster_shared_noise(
                frame=frame,
                all_client_cols=dataset.client_cols,
                cluster_assignment_map=cluster_assignment_map,
                config=cluster_shared_noise_cfg,
                rng=rng,
                timestamps=timestamps,
                events=events,
            )

        if shared_duplicate_rows:
            dup_df = frame.iloc[shared_duplicate_rows].copy(deep=True)
            dup_df[_IS_DUPLICATE_COL] = True
            frame = pd.concat([frame, dup_df], ignore_index=True)
            frame = frame.sort_values(ts_col, kind="mergesort").reset_index(drop=True)

        if shared_drop_rows:
            keep_mask = ~(
                frame[_ORIG_ROW_ID_COL].isin(shared_drop_rows)
                & ~frame[_IS_DUPLICATE_COL].astype(bool)
            )
            frame = frame.loc[keep_mask].reset_index(drop=True)

        return self._finalize_output(
            frame=frame,
            original_frame=original_frame,
            client_cols=client_cols,
            ts_col=ts_col,
            events=events,
        )


    def _inject_controlled_window_mode(
        self,
        frame: pd.DataFrame,
        original_frame: pd.DataFrame,
        client_cols: Sequence[str],
        all_client_cols: Sequence[str],
        ts_col: str,
        timestamps: pd.Series,
        seed: int,
        rng: np.random.Generator,
        cluster_assignment_path: Optional[Path | str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        _ = rng
        cfg = self._validate_controlled_window_config()
        self.last_controlled_window_summary_df = pd.DataFrame()
        self.last_controlled_selected_clients_df = pd.DataFrame()
        self._controlled_window_warnings: list[str] = []

        cluster_assignment_cfg = dict(self.config.get("cluster_assignment", {}) or {})
        cluster_assignment_map: dict[str, str] = {}
        resolved_assignment_path = cluster_assignment_path or cluster_assignment_cfg.get("path")
        if resolved_assignment_path is not None:
            if Path(resolved_assignment_path).exists():
                cluster_assignment_map = _load_cluster_assignment_map(
                    resolved_assignment_path,
                    client_id_column=str(cluster_assignment_cfg.get("client_id_column", "ID")),
                    cluster_id_column=str(cluster_assignment_cfg.get("cluster_id_column", "Cluster")),
                )
            else:
                warnings.warn(
                    "Controlled-window mode did not find a cluster assignment CSV at "
                    f"{resolved_assignment_path}; continuing without cluster_id metadata.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        client_selection_cfg = dict(self.config.get("clients", {}) or {})
        require_assignment = str(client_selection_cfg.get("selection_mode", "explicit" if client_selection_cfg.get("selected") else "all")).strip().lower() == "per_cluster_random"
        client_cols, selected_clients_df, selection_metadata = self._build_controlled_client_selection(
            all_client_cols,
            cluster_assignment_map,
            seed,
            require_cluster_assignment=require_assignment,
        )
        self.last_controlled_selected_clients_df = selected_clients_df

        events: list[FaultEvent] = []
        summary_rows: list[dict[str, Any]] = []
        n_rows = len(frame)
        dimensions = list(cfg["dimension_window_fractions"].keys())

        for client_id in client_cols:
            cluster_id = cluster_assignment_map.get(str(client_id))
            windows = self._build_controlled_windows(n_rows, cfg["window_size"], cfg["stride"])
            allocations = self._allocate_controlled_windows(
                n_windows=len(windows),
                dimension_fractions=cfg["dimension_window_fractions"],
                total_fraction=cfg["total_faulty_window_fraction"],
                client_id=str(client_id),
                cluster_id=cluster_id,
            )
            client_rng = np.random.default_rng(self._controlled_client_seed(seed, cluster_id, str(client_id)))
            window_order = client_rng.permutation(len(windows)) if windows else np.array([], dtype=int)
            initial_values = pd.to_numeric(frame[str(client_id)], errors="coerce").to_numpy(dtype=float, copy=True)
            finite_values = initial_values[np.isfinite(initial_values)]
            client_scale = _series_scale(initial_values)
            client_abs_q99 = float(np.nanquantile(np.abs(finite_values), 0.99)) if finite_values.size else 1.0
            client_stats = {
                "scale": float(client_scale),
                "upper_anchor": float(max(client_abs_q99, client_scale, 1.0)),
            }
            cursor = 0
            dimension_event_counts = {dimension: 0 for dimension in dimensions}
            selected_window_count = 0

            for dimension in dimensions:
                count = int(allocations.get(dimension, 0))
                if count <= 0:
                    continue
                chosen_window_ids = window_order[cursor: cursor + count]
                cursor += count
                for window_id in chosen_window_ids:
                    start_idx, end_idx = windows[int(window_id)]
                    applied = self._apply_controlled_window_dimension(
                        frame=frame,
                        client_id=str(client_id),
                        dimension=dimension,
                        window_start_idx=start_idx,
                        window_end_idx=end_idx,
                        timestamps=timestamps,
                        seed=seed,
                        rng=client_rng,
                        cluster_id=cluster_id,
                        cfg=cfg,
                        client_stats=client_stats,
                        events=events,
                    )
                    if applied <= 0:
                        raise RuntimeError(
                            "Controlled-window injection selected a window but did not inject any fault: "
                            f"client={client_id}, cluster={cluster_id}, dimension={dimension}, "
                            f"window=({start_idx}, {end_idx})."
                        )
                    selected_window_count += 1
                    dimension_event_counts[dimension] += applied

            row = {
                **selection_metadata,
                "client_id": str(client_id),
                "cluster_id": cluster_id,
                "number_of_windows": int(len(windows)),
                "selected_faulty_windows": int(selected_window_count),
                "achieved_faulty_fraction": float(selected_window_count / max(1, len(windows))),
            }
            for dimension in dimensions:
                row[f"windows_allocated_{dimension}"] = int(allocations.get(dimension, 0))
                row[f"fault_count_{dimension}"] = int(dimension_event_counts.get(dimension, 0))
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            if summary_df["cluster_id"].notna().any():
                cluster_counts = (
                    summary_df.groupby("cluster_id", dropna=False)["selected_faulty_windows"]
                    .sum()
                    .rename("fault_count_cluster")
                    .reset_index()
                )
                summary_df = summary_df.merge(cluster_counts, on="cluster_id", how="left")
                selected_counts = (
                    summary_df.groupby("cluster_id", dropna=False)["client_id"]
                    .nunique()
                    .rename("cluster_selected_client_count")
                    .reset_index()
                )
                summary_df = summary_df.merge(selected_counts, on="cluster_id", how="left")
            else:
                summary_df["fault_count_cluster"] = int(summary_df["selected_faulty_windows"].sum())
                summary_df["cluster_selected_client_count"] = int(summary_df["client_id"].nunique())
        if self._controlled_window_warnings:
            warning_text = "\n".join(self._controlled_window_warnings)
            warnings.warn(warning_text, RuntimeWarning, stacklevel=2)
            if not summary_df.empty:
                summary_df["warnings"] = " | ".join(self._controlled_window_warnings)
        self.last_controlled_window_summary_df = summary_df

        return self._finalize_output(
            frame=frame,
            original_frame=original_frame,
            client_cols=all_client_cols,
            ts_col=ts_col,
            events=events,
        )

    def _validate_controlled_window_config(self) -> dict[str, Any]:
        window_cfg = dict(self.config.get("window_faulting", {}) or {})
        window_size = int(window_cfg.get("window_size", 24))
        stride = int(window_cfg.get("stride", 1))
        total_fraction = float(window_cfg.get("total_faulty_window_fraction", 0.0))
        dimension_fractions = dict(window_cfg.get("dimension_window_fractions", {}) or {})
        if window_size <= 0:
            raise ValueError("window_faulting.window_size must be a positive integer.")
        if stride <= 0:
            raise ValueError("window_faulting.stride must be a positive integer.")
        if not (0.0 <= total_fraction <= 1.0):
            raise ValueError("window_faulting.total_faulty_window_fraction must be between 0 and 1.")
        if not dimension_fractions:
            raise ValueError("window_faulting.dimension_window_fractions must configure at least one dimension.")

        allowed_dimensions = {"completeness", "consistency", "accuracy", "validity", "freshness", "mixed"}
        cleaned: dict[str, float] = {}
        for dimension, raw_fraction in dimension_fractions.items():
            dimension = str(dimension)
            if dimension not in allowed_dimensions:
                raise ValueError(
                    "window_faulting.dimension_window_fractions contains an unsupported dimension: "
                    f"{dimension}. Supported dimensions are {sorted(allowed_dimensions)}."
                )
            fraction = float(raw_fraction)
            if fraction < 0:
                raise ValueError("window_faulting.dimension_window_fractions must be non-negative.")
            if fraction > 0:
                cleaned[dimension] = fraction
        fraction_sum = float(sum(cleaned.values()))
        if abs(fraction_sum - total_fraction) > 1e-9:
            raise ValueError(
                "Sum of window_faulting.dimension_window_fractions must equal "
                "window_faulting.total_faulty_window_fraction. "
                f"Got sum={fraction_sum:.12g}, total={total_fraction:.12g}."
            )

        min_faults = int(window_cfg.get("min_faults_per_faulty_window", 1))
        max_faults = int(window_cfg.get("max_faults_per_faulty_window", 2))
        multi_fault_probability = float(window_cfg.get("multi_fault_probability", 0.15))
        if min_faults <= 0:
            raise ValueError("window_faulting.min_faults_per_faulty_window must be positive.")
        if max_faults < min_faults:
            raise ValueError("window_faulting.max_faults_per_faulty_window must be >= min_faults_per_faulty_window.")
        if not (0.0 <= multi_fault_probability <= 1.0):
            raise ValueError("window_faulting.multi_fault_probability must be between 0 and 1.")
        sampling_scope = str(window_cfg.get("sampling_scope", "per_cluster_per_client"))
        if sampling_scope != "per_cluster_per_client":
            raise ValueError("controlled_window currently supports sampling_scope=per_cluster_per_client only.")

        return {
            "window_size": window_size,
            "stride": stride,
            "total_faulty_window_fraction": total_fraction,
            "dimension_window_fractions": cleaned,
            "min_faults_per_faulty_window": min_faults,
            "max_faults_per_faulty_window": max_faults,
            "multi_fault_probability": multi_fault_probability,
        }

    def _build_controlled_windows(self, n_rows: int, window_size: int, stride: int) -> list[tuple[int, int]]:
        if n_rows < window_size:
            return []
        return [
            (int(start), int(start + window_size - 1))
            for start in range(0, n_rows - window_size + 1, stride)
        ]

    def _allocate_controlled_windows(
        self,
        n_windows: int,
        dimension_fractions: dict[str, float],
        total_fraction: float,
        client_id: str,
        cluster_id: Optional[str],
    ) -> dict[str, int]:
        dimensions = list(dimension_fractions.keys())
        allocations = {dimension: 0 for dimension in dimensions}
        if n_windows <= 0 or total_fraction <= 0:
            if n_windows <= 0:
                self._controlled_window_warnings.append(
                    f"client={client_id}, cluster={cluster_id}: no windows available for controlled injection."
                )
            return allocations

        target_total = int(round(n_windows * total_fraction))
        target_total = max(1, min(n_windows, target_total))
        exact = {dimension: n_windows * fraction for dimension, fraction in dimension_fractions.items()}
        floors = {dimension: int(np.floor(value)) for dimension, value in exact.items()}
        allocations.update(floors)
        allocated = int(sum(allocations.values()))

        remainders = sorted(
            dimensions,
            key=lambda dim: (-(exact[dim] - floors[dim]), dim),
        )
        while allocated < target_total:
            for dimension in remainders:
                if allocated >= target_total:
                    break
                allocations[dimension] += 1
                allocated += 1
        while allocated > target_total:
            removable = sorted(
                [dim for dim in dimensions if allocations[dim] > 0],
                key=lambda dim: (exact[dim] - allocations[dim], dim),
            )
            if not removable:
                break
            allocations[removable[0]] -= 1
            allocated -= 1

        positive_dimensions = [dim for dim, fraction in dimension_fractions.items() if fraction > 0]
        if target_total >= len(positive_dimensions):
            missing = [dim for dim in positive_dimensions if allocations[dim] == 0]
            for dimension in missing:
                donor_candidates = sorted(
                    [dim for dim in positive_dimensions if allocations[dim] > 1],
                    key=lambda dim: (-allocations[dim], dim),
                )
                if not donor_candidates:
                    break
                donor = donor_candidates[0]
                allocations[donor] -= 1
                allocations[dimension] += 1
        elif positive_dimensions:
            self._controlled_window_warnings.append(
                f"client={client_id}, cluster={cluster_id}: only {target_total} faulty windows can be allocated "
                f"across {len(positive_dimensions)} configured dimensions; exact per-dimension coverage is impossible."
            )

        achieved = int(sum(allocations.values()))
        if achieved != target_total:
            self._controlled_window_warnings.append(
                f"client={client_id}, cluster={cluster_id}: requested {target_total} faulty windows, "
                f"allocated {achieved} after deterministic rounding."
            )
        return allocations

    def _controlled_client_seed(self, seed: int, cluster_id: Optional[str], client_id: str) -> int:
        text = f"{seed}|{cluster_id or ''}|{client_id}"
        value = 2166136261
        for byte in text.encode("utf-8"):
            value ^= byte
            value = (value * 16777619) % (2 ** 32)
        return int(value)

    def _apply_controlled_window_dimension(
        self,
        frame: pd.DataFrame,
        client_id: str,
        dimension: str,
        window_start_idx: int,
        window_end_idx: int,
        timestamps: pd.Series,
        seed: int,
        rng: np.random.Generator,
        cluster_id: Optional[str],
        cfg: dict[str, Any],
        client_stats: dict[str, float],
        events: list[FaultEvent],
    ) -> int:
        if dimension == "mixed":
            dimensions = ["completeness", "consistency", "accuracy", "validity", "freshness"]
            chosen = list(rng.choice(np.asarray(dimensions, dtype=object), size=2, replace=False))
            applied = 0
            for sub_dimension in chosen:
                applied += self._apply_controlled_window_dimension(
                    frame=frame,
                    client_id=client_id,
                    dimension=str(sub_dimension),
                    window_start_idx=window_start_idx,
                    window_end_idx=window_end_idx,
                    timestamps=timestamps,
                    seed=seed,
                    rng=rng,
                    cluster_id=cluster_id,
                    cfg=cfg,
                    client_stats=client_stats,
                    events=events,
                )
            return applied

        extra_faults = 0
        if cfg["max_faults_per_faulty_window"] > 1 and rng.random() < cfg["multi_fault_probability"]:
            extra_faults = min(1, cfg["max_faults_per_faulty_window"] - 1)
        attempts = max(cfg["min_faults_per_faulty_window"], 1) + extra_faults
        applied = 0
        for _ in range(attempts):
            event = self._apply_single_controlled_fault(
                frame=frame,
                client_id=client_id,
                dimension=dimension,
                window_start_idx=window_start_idx,
                window_end_idx=window_end_idx,
                timestamps=timestamps,
                seed=seed,
                rng=rng,
                cluster_id=cluster_id,
                client_stats=client_stats,
            )
            if event is not None:
                events.append(event)
                applied += 1
        return applied

    def _window_valid_positions(
        self,
        frame: pd.DataFrame,
        client_id: str,
        start_idx: int,
        end_idx: int,
    ) -> np.ndarray:
        idxs = np.arange(start_idx, end_idx + 1, dtype=int)
        values = pd.to_numeric(frame.loc[idxs, client_id], errors="coerce").to_numpy(dtype=float)
        return idxs[np.isfinite(values)]

    def _choose_window_positions(
        self,
        valid_positions: np.ndarray,
        rng: np.random.Generator,
        min_count: int = 1,
        max_count: int = 1,
    ) -> np.ndarray:
        if valid_positions.size == 0:
            return np.array([], dtype=int)
        max_count = max(min_count, max_count)
        count = int(rng.integers(min_count, max_count + 1))
        count = min(count, valid_positions.size)
        return np.sort(rng.choice(valid_positions, size=count, replace=False))

    def _apply_single_controlled_fault(
        self,
        frame: pd.DataFrame,
        client_id: str,
        dimension: str,
        window_start_idx: int,
        window_end_idx: int,
        timestamps: pd.Series,
        seed: int,
        rng: np.random.Generator,
        cluster_id: Optional[str],
        client_stats: dict[str, float],
    ) -> Optional[FaultEvent]:
        n_rows = len(frame)
        valid_positions = self._window_valid_positions(frame, client_id, window_start_idx, window_end_idx)
        window_positions = np.arange(window_start_idx, window_end_idx + 1, dtype=int)
        candidate_positions = valid_positions if valid_positions.size else window_positions
        if candidate_positions.size == 0:
            return None
        scale = float(client_stats.get("scale", 1.0))
        upper_anchor = float(client_stats.get("upper_anchor", max(scale, 1.0)))
        max_cells = min(3, max(1, window_end_idx - window_start_idx + 1))
        idxs = self._choose_window_positions(candidate_positions, rng, 1, max_cells)
        if idxs.size == 0:
            return None

        if dimension == "completeness":
            completeness_cfg = _resolve_fault_config(self.config, "completeness")
            fault_type = "controlled_random_missing"
            subcheck = "C1"
            frame.loc[idxs, client_id] = np.nan
            parameters = {
                "mode": "controlled_window",
                "source_dimension": dimension,
                "severity_config": {
                    "random_missing_rate": completeness_cfg.get("random_missing_rate"),
                    "block_missing": completeness_cfg.get("block_missing"),
                },
            }
        elif dimension == "validity":
            validity_cfg = _resolve_fault_config(self.config, "validity")
            fault_type = str(rng.choice(np.asarray(["controlled_negative_values", "controlled_impossible_high"], dtype=object)))
            subcheck = "V3"
            if fault_type == "controlled_negative_values":
                current = pd.to_numeric(frame.loc[idxs, client_id], errors="coerce").to_numpy(dtype=float)
                current = np.nan_to_num(current, nan=0.0)
                base = np.maximum(np.abs(current), 1.0)
                frame.loc[idxs, client_id] = -base
            else:
                high_multiplier = float(validity_cfg.get("high_multiplier", 3.0))
                frame.loc[idxs, client_id] = upper_anchor * high_multiplier
            parameters = {
                "mode": "controlled_window",
                "source_dimension": dimension,
                "high_multiplier": float(validity_cfg.get("high_multiplier", 3.0)),
            }
        elif dimension == "consistency":
            consistency_cfg = _resolve_fault_config(self.config, "consistency")
            fault_type = str(rng.choice(np.asarray(["controlled_stale_values", "controlled_rate_jumps", "controlled_stuck_constant"], dtype=object)))
            if fault_type == "controlled_stale_values":
                subcheck = "CS4"
                lag_choices = np.asarray(consistency_cfg.get("lag_choices", [1]), dtype=int)
                lag_choices = lag_choices[lag_choices != 0]
                if lag_choices.size == 0:
                    lag_choices = np.array([1], dtype=int)
                lag = int(rng.choice(lag_choices))
                source_positions = np.clip(idxs - lag, 0, n_rows - 1)
                source_values = pd.to_numeric(frame.loc[source_positions, client_id], errors="coerce").to_numpy(dtype=float)
                frame.loc[idxs, client_id] = np.nan_to_num(source_values, nan=0.0)
                parameters = {"mode": "controlled_window", "source_dimension": dimension, "lag": lag}
            elif fault_type == "controlled_rate_jumps":
                subcheck = "CS5"
                jump_multiplier = float(consistency_cfg.get("jump_multiplier", 6.0))
                previous = pd.to_numeric(frame.loc[np.maximum(idxs - 1, 0), client_id], errors="coerce").to_numpy(dtype=float)
                previous = np.nan_to_num(previous, nan=0.0)
                signs = rng.choice(np.asarray([-1.0, 1.0]), size=idxs.size)
                frame.loc[idxs, client_id] = np.maximum(0.0, previous + signs * jump_multiplier * scale)
                parameters = {"mode": "controlled_window", "source_dimension": dimension, "jump_multiplier": jump_multiplier}
            else:
                subcheck = "CS4"
                constant = float(pd.to_numeric(frame.loc[[int(idxs[0])], client_id], errors="coerce").iloc[0])
                if not np.isfinite(constant):
                    constant = 0.0
                frame.loc[idxs, client_id] = constant
                parameters = {"mode": "controlled_window", "source_dimension": dimension}
        elif dimension == "accuracy":
            accuracy_cfg = _resolve_fault_config(self.config, "accuracy")
            fault_type = str(rng.choice(np.asarray(["controlled_spikes", "controlled_noisy_values"], dtype=object)))
            current = pd.to_numeric(frame.loc[idxs, client_id], errors="coerce").to_numpy(dtype=float)
            current = np.nan_to_num(current, nan=0.0)
            if fault_type == "controlled_spikes":
                subcheck = "A1"
                spike_multiplier = float(accuracy_cfg.get("spike_multiplier", 4.0))
                frame.loc[idxs, client_id] = current + spike_multiplier * scale
                parameters = {"mode": "controlled_window", "source_dimension": dimension, "spike_multiplier": spike_multiplier}
            else:
                subcheck = "A3"
                noise_std_fraction = float(accuracy_cfg.get("noise_std_fraction", 0.05))
                noise = rng.normal(loc=0.0, scale=max(scale * noise_std_fraction, 1e-6), size=idxs.size)
                frame.loc[idxs, client_id] = current + noise
                parameters = {"mode": "controlled_window", "source_dimension": dimension, "noise_std_fraction": noise_std_fraction}
        elif dimension == "freshness":
            freshness_cfg = _resolve_fault_config(self.config, "freshness")
            window_length = window_end_idx - window_start_idx + 1
            delay_steps = int(freshness_cfg.get("delay_steps", 5))
            delay_steps = max(1, min(delay_steps, window_length))
            idxs = np.arange(window_end_idx - delay_steps + 1, window_end_idx + 1, dtype=int)
            fault_type = "controlled_delayed_tail"
            subcheck = "F1"
            frame.loc[idxs, client_id] = np.nan
            parameters = {
                "mode": "controlled_window",
                "source_dimension": dimension,
                "delay_steps": delay_steps,
                "delay_start_offset": window_length - delay_steps,
            }
        else:
            raise ValueError(f"Unsupported controlled-window dimension: {dimension}")

        return self._make_event(
            client_id=client_id,
            fault_type=fault_type,
            dimension=dimension,
            subcheck=subcheck,
            idxs=idxs,
            timestamps=timestamps,
            n_rows=n_rows,
            parameters=parameters,
            cluster_id=cluster_id,
            scope="controlled_window",
            window_start_index=window_start_idx,
            window_end_index=window_end_idx,
            seed=seed,
        )


    def _controlled_selection_seed(self, seed: int, *parts: Any) -> int:
        text = "|".join([str(seed), *[str(part) for part in parts]])
        value = 2166136261
        for byte in text.encode("utf-8"):
            value ^= byte
            value = (value * 16777619) % (2 ** 32)
        return int(value)

    def _controlled_client_order(self, clients: Sequence[str], seed: int, *parts: Any) -> list[str]:
        ordered = sorted(map(str, clients))
        if not ordered:
            return []
        rng = np.random.default_rng(self._controlled_selection_seed(seed, *parts))
        order = rng.permutation(len(ordered))
        return [ordered[int(idx)] for idx in order]

    def _controlled_selection_count(
        self,
        *,
        available_count: int,
        client_cfg: dict,
        per_cluster: bool,
    ) -> int:
        if available_count <= 0:
            return 0
        random_count = client_cfg.get("random_count")
        random_fraction = client_cfg.get("random_fraction")
        min_key = "min_count_per_cluster" if per_cluster else "min_count"
        min_count = max(0, int(client_cfg.get(min_key, client_cfg.get("min_count_per_cluster", 0 if random_fraction in (None, 0, 0.0) else 1))))
        if random_count is not None:
            count = int(random_count)
        elif random_fraction is not None:
            fraction = float(random_fraction)
            if not (0.0 <= fraction <= 1.0):
                raise ValueError("clients.random_fraction must be within [0, 1].")
            rounding = str(client_cfg.get("rounding", "round")).strip().lower()
            exact = available_count * fraction
            if rounding == "floor":
                count = int(np.floor(exact))
            elif rounding == "ceil":
                count = int(np.ceil(exact))
            elif rounding == "round":
                count = int(round(exact))
            else:
                raise ValueError("clients.rounding must be one of: round, floor, ceil.")
            if fraction > 0:
                count = max(min_count, count)
        else:
            count = available_count
        return max(0, min(available_count, count))

    def _build_controlled_client_selection(
        self,
        all_clients: Sequence[str],
        cluster_assignment_map: dict[str, str],
        seed: int,
        *,
        require_cluster_assignment: bool,
    ) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
        client_cfg = dict(self.config.get("clients", {}) or {})
        selection_mode = str(client_cfg.get("selection_mode", "explicit" if client_cfg.get("selected") else "all")).strip().lower()
        include_unassigned = bool(client_cfg.get("include_unassigned", False))
        all_client_set = {str(client) for client in all_clients}
        selected_raw = _normalize_identifier_list(client_cfg.get("selected", self.config.get("selected_clients")))

        assigned_clients = sorted(client for client in all_client_set if client in cluster_assignment_map)
        unassigned_clients = sorted(all_client_set - set(assigned_clients))
        if require_cluster_assignment and selection_mode == "per_cluster_random" and not cluster_assignment_map:
            raise ValueError("clients.selection_mode=per_cluster_random requires a cluster assignment CSV.")

        eligible_clients = sorted(all_client_set if include_unassigned or not cluster_assignment_map else set(assigned_clients))
        if selection_mode == "all":
            selected_clients = eligible_clients
            selection_rows = []
            for rank, client_id in enumerate(selected_clients, start=1):
                selection_rows.append({
                    "cluster_id": cluster_assignment_map.get(client_id),
                    "client_id": client_id,
                    "selection_mode": selection_mode,
                    "selection_rank": rank,
                    "selected": True,
                    "selected_for_1_client": True,
                    "selected_for_20": True,
                    "selected_for_40": True,
                    "selected_for_100": True,
                    "seed": int(seed),
                })
        elif selection_mode == "explicit":
            selected_clients = [client for client in selected_raw if client in all_client_set]
            missing = sorted(set(selected_raw) - all_client_set)
            if missing:
                self._controlled_window_warnings.append(
                    "Explicit controlled clients missing from dataset columns: " + ",".join(missing[:20])
                )
            selection_rows = []
            for rank, client_id in enumerate(selected_clients, start=1):
                selection_rows.append({
                    "cluster_id": cluster_assignment_map.get(client_id),
                    "client_id": client_id,
                    "selection_mode": selection_mode,
                    "selection_rank": rank,
                    "selected": True,
                    "selected_for_1_client": rank <= 1,
                    "selected_for_20": True,
                    "selected_for_40": True,
                    "selected_for_100": True,
                    "seed": int(seed),
                })
        elif selection_mode == "global_random":
            ordered = self._controlled_client_order(eligible_clients, seed, "global")
            count = self._controlled_selection_count(available_count=len(ordered), client_cfg=client_cfg, per_cluster=False)
            selected_set = set(ordered[:count])
            selected_clients = sorted(selected_set)
            selection_rows = []
            for rank, client_id in enumerate(ordered, start=1):
                selection_rows.append({
                    "cluster_id": cluster_assignment_map.get(client_id),
                    "client_id": client_id,
                    "selection_mode": selection_mode,
                    "selection_rank": rank,
                    "selected": client_id in selected_set,
                    "selected_for_1_client": rank <= 1,
                    "selected_for_20": rank <= self._controlled_selection_count(available_count=len(ordered), client_cfg={**client_cfg, "random_fraction": 0.20, "random_count": None}, per_cluster=False),
                    "selected_for_40": rank <= self._controlled_selection_count(available_count=len(ordered), client_cfg={**client_cfg, "random_fraction": 0.40, "random_count": None}, per_cluster=False),
                    "selected_for_100": True,
                    "seed": int(seed),
                })
        elif selection_mode == "per_cluster_random":
            cluster_members: dict[str, list[str]] = {}
            for client_id in assigned_clients:
                cluster_members.setdefault(str(cluster_assignment_map[client_id]), []).append(client_id)
            if not cluster_members:
                raise ValueError("No assigned clients matched dataset columns for per_cluster_random selection.")
            selected_set: set[str] = set()
            selection_rows = []
            for cluster_id in sorted(cluster_members, key=lambda value: (int(value) if str(value).isdigit() else str(value))):
                ordered = self._controlled_client_order(cluster_members[cluster_id], seed, "cluster", cluster_id)
                count = self._controlled_selection_count(available_count=len(ordered), client_cfg=client_cfg, per_cluster=True)
                selected_set.update(ordered[:count])
                count_20 = self._controlled_selection_count(available_count=len(ordered), client_cfg={**client_cfg, "random_fraction": 0.20, "random_count": None}, per_cluster=True)
                count_40 = self._controlled_selection_count(available_count=len(ordered), client_cfg={**client_cfg, "random_fraction": 0.40, "random_count": None}, per_cluster=True)
                for rank, client_id in enumerate(ordered, start=1):
                    selection_rows.append({
                        "cluster_id": cluster_id,
                        "client_id": client_id,
                        "selection_mode": selection_mode,
                        "selection_rank": rank,
                        "selected": client_id in selected_set,
                        "selected_for_1_client": rank <= 1,
                        "selected_for_20": rank <= count_20,
                        "selected_for_40": rank <= count_40,
                        "selected_for_100": True,
                        "seed": int(seed),
                    })
            if include_unassigned and unassigned_clients:
                ordered_unassigned = self._controlled_client_order(unassigned_clients, seed, "cluster", "__unassigned__")
                for rank, client_id in enumerate(ordered_unassigned, start=1):
                    selection_rows.append({
                        "cluster_id": None,
                        "client_id": client_id,
                        "selection_mode": selection_mode,
                        "selection_rank": rank,
                        "selected": False,
                        "selected_for_1_client": False,
                        "selected_for_20": False,
                        "selected_for_40": False,
                        "selected_for_100": True,
                        "seed": int(seed),
                    })
            selected_clients = sorted(selected_set)
        else:
            raise ValueError("clients.selection_mode must be one of: all, explicit, global_random, per_cluster_random.")

        if not selected_clients:
            raise ValueError("Controlled-window client selection produced no selected clients.")
        selected_df = pd.DataFrame(selection_rows)
        metadata = {
            "client_selection_mode": selection_mode,
            "client_selection_seed": int(seed),
            "selected_client_count": int(len(selected_clients)),
            "eligible_client_count": int(len(eligible_clients)),
            "assigned_client_count": int(len(assigned_clients)),
            "unassigned_client_count": int(len(unassigned_clients)),
        }
        return selected_clients, selected_df, metadata

    def _select_clients(
        self,
        all_clients: Sequence[str],
        rng: np.random.Generator,
    ) -> list[str]:
        client_cfg = dict(self.config.get("clients", {}) or {})
        selected = list(client_cfg.get("selected", self.config.get("selected_clients", [])) or [])
        candidates = [client for client in all_clients if not selected or client in selected]
        if not candidates:
            candidates = list(all_clients)

        random_count = client_cfg.get("random_count", self.config.get("random_clients_count"))
        random_fraction = float(client_cfg.get("random_fraction", self.config.get("random_client_fraction", 1.0)))
        if random_count is not None:
            count = max(1, min(len(candidates), int(random_count)))
            return sorted(map(str, rng.choice(candidates, size=count, replace=False)))
        if random_fraction < 1.0:
            count = max(1, int(round(len(candidates) * random_fraction)))
            count = min(len(candidates), count)
            return sorted(map(str, rng.choice(candidates, size=count, replace=False)))
        return list(map(str, candidates))

    def _inject_simple_fraction_mode(
        self,
        frame: pd.DataFrame,
        original_frame: pd.DataFrame,
        client_cols: Sequence[str],
        ts_col: str,
        timestamps: pd.Series,
        seed: int,
        rng: np.random.Generator,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        fault_fraction = float(self.config.get("fault_fraction", 0.0))
        if not (0.0 <= fault_fraction <= 1.0):
            raise ValueError("fault_fraction must be in [0, 1].")

        events: list[FaultEvent] = []
        n_rows = len(frame)
        simple_fault_types = list(self.config.get("simple_fault_types", SIMPLE_FAULT_TYPES) or SIMPLE_FAULT_TYPES)
        if not simple_fault_types:
            simple_fault_types = list(SIMPLE_FAULT_TYPES)

        probabilities = self.config.get("simple_fault_probabilities")
        if probabilities is None:
            fault_probabilities = np.full(len(simple_fault_types), 1.0 / len(simple_fault_types), dtype=float)
        else:
            fault_probabilities = np.asarray(probabilities, dtype=float)
            if fault_probabilities.shape[0] != len(simple_fault_types):
                raise ValueError("simple_fault_probabilities must match simple_fault_types length.")
            if np.any(fault_probabilities < 0) or fault_probabilities.sum() <= 0:
                raise ValueError("simple_fault_probabilities must be non-negative and sum to a positive value.")
            fault_probabilities = fault_probabilities / fault_probabilities.sum()

        invalid_high_low, invalid_high_high = self.config.get(
            "simple_invalid_high_range",
            self.config.get("simple_out_of_range_range", [1.5, 3.0]),
        )
        negative_low, negative_high = self.config.get(
            "simple_negative_value_range",
            self.config.get("simple_inconsistent_range", [4.0, 8.0]),
        )
        noise_std_fraction = float(self.config.get("simple_noise_std_fraction", 0.05))
        spike_multiplier = float(self.config.get("simple_spike_multiplier", 4.0))
        replayed_steps = np.asarray(
            self.config.get("simple_replayed_steps", self.config.get("simple_timeliness_steps", [-1, 1])),
            dtype=int,
        )
        if replayed_steps.size == 0:
            replayed_steps = np.array([-1, 1], dtype=int)

        for client_id in client_cols:
            values = pd.to_numeric(frame[client_id], errors="coerce").to_numpy(dtype=float, copy=True)
            valid_positions = np.flatnonzero(np.isfinite(values))
            if valid_positions.size == 0 or fault_fraction <= 0:
                continue

            fault_positions = _sample_positions(valid_positions, fault_fraction, rng)
            if fault_positions.size == 0:
                continue

            scale = _series_scale(values)
            fault_types = rng.choice(
                np.asarray(simple_fault_types, dtype=object),
                size=fault_positions.size,
                replace=True,
                p=fault_probabilities,
            )
            original_values = values[fault_positions].copy()
            mutated_values = original_values.copy()

            for fault_type in simple_fault_types:
                mask = fault_types == fault_type
                if not mask.any():
                    continue
                idxs = fault_positions[mask]

                if fault_type == "missing":
                    mutated_values[mask] = np.nan
                elif fault_type in {"invalid_high", "out_of_range"}:
                    magnitudes = np.maximum(np.abs(np.nan_to_num(original_values[mask], nan=0.0)), 1.0)
                    mutated_values[mask] = magnitudes * rng.uniform(invalid_high_low, invalid_high_high, size=mask.sum())
                elif fault_type in {"negative_value", "inconsistent"}:
                    magnitudes = np.maximum(np.abs(np.nan_to_num(original_values[mask], nan=0.0)), 1.0)
                    mutated_values[mask] = -magnitudes * rng.uniform(negative_low, negative_high, size=mask.sum())
                elif fault_type in {"replayed_value", "timeliness"}:
                    steps = rng.choice(replayed_steps, size=mask.sum(), replace=True)
                    source_pos = np.clip(idxs + steps, 0, n_rows - 1)
                    stale_values = values[source_pos].copy()

                    same_mask = _values_equal(stale_values, original_values[mask])
                    if same_mask.any():
                        left_pos = np.clip(idxs[same_mask] - 1, 0, n_rows - 1)
                        right_pos = np.clip(idxs[same_mask] + 1, 0, n_rows - 1)
                        left_values = values[left_pos]
                        right_values = values[right_pos]
                        orig_sub = original_values[mask][same_mask]

                        replacement_values = left_values.copy()
                        pick_right = ~_values_equal(right_values, orig_sub)
                        replacement_values[pick_right] = right_values[pick_right]

                        still_same = _values_equal(replacement_values, orig_sub)
                        if still_same.any():
                            magnitudes = np.maximum(np.abs(np.nan_to_num(orig_sub[still_same], nan=0.0)), 1.0)
                            replacement_values[still_same] = -magnitudes * 0.5

                        stale_values[same_mask] = replacement_values

                    mutated_values[mask] = stale_values
                elif fault_type == "noise":
                    noise = rng.normal(loc=0.0, scale=max(scale * noise_std_fraction, 1e-6), size=mask.sum())
                    mutated_values[mask] = original_values[mask] + noise
                elif fault_type == "spike":
                    mutated_values[mask] = original_values[mask] + spike_multiplier * scale
                else:
                    raise ValueError(f"Unsupported simple fault type: {fault_type}")

            frame.loc[fault_positions, client_id] = mutated_values

            for fault_type in simple_fault_types:
                mask = fault_types == fault_type
                if not mask.any():
                    continue
                idxs = fault_positions[mask]
                dimension, subcheck = SIMPLE_FAULT_META.get(fault_type, ("accuracy", "A2"))
                events.append(
                    self._make_event(
                        client_id=client_id,
                        fault_type=str(fault_type),
                        dimension=dimension,
                        subcheck=subcheck,
                        idxs=idxs,
                        timestamps=timestamps,
                        n_rows=n_rows,
                        parameters={
                            "mode": "simple_fault_fraction",
                            "fault_fraction": fault_fraction,
                            "seed": seed,
                            "simple_fault_types": simple_fault_types,
                        },
                    )
                )

        return self._finalize_output(
            frame=frame,
            original_frame=original_frame,
            client_cols=client_cols,
            ts_col=ts_col,
            events=events,
        )

    def _finalize_output(
        self,
        frame: pd.DataFrame,
        original_frame: pd.DataFrame,
        client_cols: Sequence[str],
        ts_col: str,
        events: Sequence[FaultEvent],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        summary_events = self._build_final_corruption_summary(
            original_frame=original_frame,
            final_frame=frame,
            client_cols=client_cols,
            ts_col=ts_col,
        )
        report_df = pd.DataFrame([asdict(event) for event in [*events, *summary_events]])
        if not report_df.empty:
            report_df = report_df.sort_values(
                ["dimension", "subcheck", "client_id", "start_time"],
                kind="mergesort",
            ).reset_index(drop=True)
        restore_expected_timestamps = bool(
            self.config.get("restore_expected_timestamps_after_injection", True)
        )
        if restore_expected_timestamps:
            clean_frame = self._restore_expected_timestamp_grid(
                frame=frame,
                original_frame=original_frame,
                client_cols=client_cols,
                ts_col=ts_col,
            )
        else:
            clean_frame = frame.drop(
                columns=[
                    _ORIG_ROW_ID_COL,
                    _IS_DUPLICATE_COL,
                    _MISSING_TIMESTAMP_COL,
                    _DUPLICATE_TIMESTAMP_COUNT_COL,
                ],
                errors="ignore",
            )
        return clean_frame, report_df

    def _restore_expected_timestamp_grid(
        self,
        *,
        frame: pd.DataFrame,
        original_frame: pd.DataFrame,
        client_cols: Sequence[str],
        ts_col: str,
    ) -> pd.DataFrame:
        row_ids = np.arange(len(original_frame), dtype=int)
        nonduplicate = (
            frame.loc[~frame[_IS_DUPLICATE_COL].astype(bool)]
            .set_index(_ORIG_ROW_ID_COL, drop=False)
            .reindex(row_ids)
        )
        duplicate_counts = (
            frame.loc[frame[_IS_DUPLICATE_COL].astype(bool), _ORIG_ROW_ID_COL]
            .value_counts()
            .to_dict()
        )

        restored_columns = {
            ts_col: pd.to_datetime(original_frame[ts_col], errors="coerce").reset_index(drop=True),
        }
        for client_id in client_cols:
            restored_columns[client_id] = pd.to_numeric(nonduplicate[client_id], errors="coerce").to_numpy(dtype=float)
        restored_columns[_MISSING_TIMESTAMP_COL] = nonduplicate[_ORIG_ROW_ID_COL].isna().astype(int).to_numpy()
        restored_columns[_DUPLICATE_TIMESTAMP_COUNT_COL] = np.asarray(
            [int(duplicate_counts.get(int(row_id), 0)) for row_id in row_ids],
            dtype=int,
        )
        return pd.DataFrame(restored_columns)

    def _build_final_corruption_summary(
        self,
        original_frame: pd.DataFrame,
        final_frame: pd.DataFrame,
        client_cols: Sequence[str],
        ts_col: str,
    ) -> list[FaultEvent]:
        n_rows = len(original_frame)
        if n_rows == 0:
            return []

        original_timestamps = original_frame[ts_col].reset_index(drop=True)
        row_ids = np.arange(n_rows, dtype=int)
        nonduplicate = (
            final_frame.loc[~final_frame[_IS_DUPLICATE_COL].astype(bool)]
            .set_index(_ORIG_ROW_ID_COL, drop=False)
            .reindex(row_ids)
        )

        original_present = np.zeros(n_rows, dtype=bool)
        duplicate_present = np.zeros(n_rows, dtype=bool)
        for row_id, group in final_frame.groupby(_ORIG_ROW_ID_COL, sort=False):
            row_idx = int(row_id)
            dup_flags = group[_IS_DUPLICATE_COL].astype(bool).to_numpy()
            original_present[row_idx] = bool((~dup_flags).any())
            duplicate_present[row_idx] = bool(dup_flags.any())
        shared_structural_change = (~original_present) | duplicate_present

        summary_events: list[FaultEvent] = []
        overall_changed = shared_structural_change.copy()

        for client_id in client_cols:
            original_values = pd.to_numeric(original_frame[client_id], errors="coerce").to_numpy(dtype=float)
            final_values = pd.to_numeric(nonduplicate[client_id], errors="coerce").to_numpy(dtype=float)
            value_changed = ~_values_equal(original_values, final_values)
            changed = shared_structural_change | value_changed
            overall_changed |= changed
            idxs = np.flatnonzero(changed)
            fraction = float(idxs.size / max(1, n_rows))
            summary_events.append(
                self._make_event(
                    client_id=client_id,
                    fault_type="final_net_corruption",
                    dimension="summary",
                    subcheck="final_rows_changed",
                    idxs=idxs,
                    timestamps=original_timestamps,
                    n_rows=n_rows,
                    parameters={
                        "summary_kind": "per_client_final_net_corruption",
                        "total_original_rows": n_rows,
                        "corrupted_rows": int(idxs.size),
                        "corrupted_percent": 100.0 * fraction,
                    },
                )
            )

        overall_idxs = np.flatnonzero(overall_changed)
        overall_fraction = float(overall_idxs.size / max(1, n_rows))
        summary_events.append(
            self._make_event(
                client_id="__all_selected_clients__",
                fault_type="final_net_corruption",
                dimension="summary",
                subcheck="final_rows_changed",
                idxs=overall_idxs,
                timestamps=original_timestamps,
                n_rows=n_rows,
                parameters={
                    "summary_kind": "overall_final_net_corruption",
                    "total_original_rows": n_rows,
                    "corrupted_rows": int(overall_idxs.size),
                    "corrupted_percent": 100.0 * overall_fraction,
                    "selected_clients": list(map(str, client_cols)),
                },
            )
        )
        return summary_events

    def _inject_completeness_faults(
        self,
        frame: pd.DataFrame,
        client_id: str,
        values: np.ndarray,
        valid_positions: np.ndarray,
        config: dict,
        client_factor: float,
        rng: np.random.Generator,
        timestamps: pd.Series,
        events: list[FaultEvent],
    ) -> None:
        n_rows = len(frame)
        dropout_fraction = float(config.get("client_dropout_fraction", 0.0))
        if dropout_fraction > 0 and rng.random() < min(1.0, dropout_fraction * max(client_factor, 0.0)):
            all_valid = np.flatnonzero(np.isfinite(pd.to_numeric(frame[client_id], errors="coerce").to_numpy(dtype=float)))
            if all_valid.size:
                frame.loc[all_valid, client_id] = np.nan
                events.append(
                    self._make_event(
                        client_id,
                        "client_dropout",
                        "completeness",
                        "C2",
                        all_valid,
                        timestamps,
                        n_rows,
                        {"fraction": dropout_fraction},
                    )
                )
            return

        random_missing_rate = float(config.get("random_missing_rate", 0.0)) * client_factor
        idxs = _sample_positions(valid_positions, random_missing_rate, rng)
        if idxs.size:
            frame.loc[idxs, client_id] = np.nan
            events.append(
                self._make_event(
                    client_id,
                    "random_missing",
                    "completeness",
                    "C1",
                    idxs,
                    timestamps,
                    n_rows,
                    {"rate": random_missing_rate},
                )
            )

        block_cfg = dict(config.get("block_missing", {}) or {})
        block_count = int(round(float(block_cfg.get("events_per_client", 0)) * max(client_factor, 0.0)))
        blocks = _sample_blocks(
            n_rows=n_rows,
            count=block_count,
            min_length=int(block_cfg.get("min_length", 1)),
            max_length=int(block_cfg.get("max_length", 1)),
            rng=rng,
        )
        for block in blocks:
            frame.loc[block, client_id] = np.nan
            events.append(
                self._make_event(
                    client_id,
                    "block_missing",
                    "completeness",
                    "C1",
                    block,
                    timestamps,
                    n_rows,
                    dict(block_cfg),
                )
            )
        hole_fraction = float(config.get("calendar_hole_fraction", 0.0)) * min(client_factor, 1.0)
        if hole_fraction > 0:
            block = _sample_single_block(
                n_rows=n_rows,
                min_length=int(config.get("calendar_hole_min_length", 72)),
                max_length=int(config.get("calendar_hole_max_length", 336)),
                rng=rng,
            )
            if block.size:
                ts_block = pd.Series(pd.to_datetime(timestamps.iloc[block], errors="coerce"))
                available_cells = sorted(
                    {
                        (int(ts.dayofweek), int(ts.hour))
                        for ts in ts_block.dropna()
                    }
                )
                if available_cells:
                    hole_count = max(1, int(round(len(available_cells) * hole_fraction)))
                    chosen_cell_ids = rng.choice(
                        len(available_cells),
                        size=min(hole_count, len(available_cells)),
                        replace=False,
                    )
                    chosen_cells = {available_cells[int(idx)] for idx in chosen_cell_ids}
                    idxs: list[int] = []
                    for row_idx, ts_value in zip(block.tolist(), ts_block.tolist()):
                        if pd.isna(ts_value):
                            continue
                        cell = (int(ts_value.dayofweek), int(ts_value.hour))
                        if cell in chosen_cells:
                            idxs.append(int(row_idx))
                    if idxs:
                        idx_array = np.array(sorted(set(idxs)), dtype=int)
                        frame.loc[idx_array, client_id] = np.nan
                        events.append(
                            self._make_event(
                                client_id,
                                "calendar_hole",
                                "completeness",
                                "C3",
                                idx_array,
                                timestamps,
                                n_rows,
                                {
                                    "fraction": hole_fraction,
                                    "window_length": int(block.size),
                                    "available_cells": int(len(available_cells)),
                                },
                            )
                        )

    def _inject_validity_faults(
        self,
        frame: pd.DataFrame,
        client_id: str,
        values: np.ndarray,
        valid_positions: np.ndarray,
        scale: float,
        upper_anchor: float,
        config: dict,
        client_factor: float,
        rng: np.random.Generator,
        timestamps: pd.Series,
        events: list[FaultEvent],
    ) -> None:
        n_rows = len(frame)

        negative_rate = float(config.get("negative_rate", 0.0)) * client_factor
        idxs = _sample_positions(valid_positions, negative_rate, rng)
        if idxs.size:
            base = np.maximum(
                np.abs(pd.to_numeric(frame.loc[idxs, client_id], errors="coerce").to_numpy(dtype=float)),
                1.0,
            )
            frame.loc[idxs, client_id] = -base
            events.append(
                self._make_event(
                    client_id,
                    "negative_values",
                    "validity",
                    "V3",
                    idxs,
                    timestamps,
                    n_rows,
                    {"rate": negative_rate},
                )
            )

        high_rate = float(config.get("impossible_high_rate", 0.0)) * client_factor
        high_multiplier = float(config.get("high_multiplier", 3.0))
        idxs = _sample_positions(valid_positions, high_rate, rng)
        if idxs.size:
            frame.loc[idxs, client_id] = upper_anchor * high_multiplier
            events.append(
                self._make_event(
                    client_id,
                    "impossible_high",
                    "validity",
                    "V3",
                    idxs,
                    timestamps,
                    n_rows,
                    {"rate": high_rate, "multiplier": high_multiplier},
                )
            )

    def _inject_consistency_faults(
        self,
        frame: pd.DataFrame,
        client_id: str,
        values: np.ndarray,
        valid_positions: np.ndarray,
        config: dict,
        client_factor: float,
        rng: np.random.Generator,
        timestamps: pd.Series,
        events: list[FaultEvent],
    ) -> None:
        n_rows = len(frame)
        scale = _series_scale(values)

        stale_blocks = int(round(float(config.get("stale_blocks_per_client", 0)) * max(client_factor, 0.0)))
        blocks = _sample_blocks(
            n_rows=n_rows,
            count=stale_blocks,
            min_length=int(config.get("stale_min_length", 1)),
            max_length=int(config.get("stale_max_length", 1)),
            rng=rng,
        )
        lag_choices = np.asarray(config.get("lag_choices", [1]), dtype=int)
        lag_probs = np.asarray(config.get("lag_probabilities", np.ones(len(lag_choices))), dtype=float)
        lag_probs = lag_probs / lag_probs.sum()
        for block in blocks:
            if block.size == 0:
                continue
            lag = int(rng.choice(lag_choices, p=lag_probs))
            src = np.clip(block - lag, 0, n_rows - 1)
            frame.loc[block, client_id] = pd.to_numeric(frame.loc[src, client_id], errors="coerce").to_numpy(dtype=float)
            events.append(
                self._make_event(
                    client_id,
                    "stale_block",
                    "consistency",
                    "CS4",
                    block,
                    timestamps,
                    n_rows,
                    {"lag": lag},
                )
            )

        jump_rate = float(config.get("jump_rate", 0.0)) * client_factor
        jump_multiplier = float(config.get("jump_multiplier", 6.0))
        idxs = _sample_positions(valid_positions[valid_positions > 0], jump_rate, rng)
        if idxs.size:
            previous = pd.to_numeric(frame.loc[idxs - 1, client_id], errors="coerce").to_numpy(dtype=float)
            signs = rng.choice(np.array([-1.0, 1.0]), size=idxs.size)
            frame.loc[idxs, client_id] = np.maximum(0.0, previous + signs * jump_multiplier * scale)
            events.append(
                self._make_event(
                    client_id,
                    "rate_jumps",
                    "consistency",
                    "CS5",
                    idxs,
                    timestamps,
                    n_rows,
                    {"rate": jump_rate, "multiplier": jump_multiplier},
                )
            )

        stuck_segments = int(round(float(config.get("stuck_segments_per_client", 0)) * max(client_factor, 0.0)))
        blocks = _sample_blocks(
            n_rows=n_rows,
            count=stuck_segments,
            min_length=int(config.get("stuck_min_length", 1)),
            max_length=int(config.get("stuck_max_length", 1)),
            rng=rng,
        )
        for block in blocks:
            if block.size == 0:
                continue
            constant = float(pd.to_numeric(frame.loc[[int(block[0])], client_id], errors="coerce").iloc[0])
            frame.loc[block, client_id] = constant
            events.append(
                self._make_event(
                    client_id,
                    "stuck_constant",
                    "consistency",
                    "CS4",
                    block,
                    timestamps,
                    n_rows,
                    {
                        "min_length": int(config.get("stuck_min_length", 1)),
                        "max_length": int(config.get("stuck_max_length", 1)),
                    },
                )
            )

    def _inject_accuracy_faults(
        self,
        frame: pd.DataFrame,
        client_id: str,
        values: np.ndarray,
        valid_positions: np.ndarray,
        scale: float,
        config: dict,
        client_factor: float,
        rng: np.random.Generator,
        timestamps: pd.Series,
        events: list[FaultEvent],
    ) -> None:
        n_rows = len(frame)

        spike_rate = float(config.get("spike_rate", 0.0)) * client_factor
        spike_multiplier = float(config.get("spike_multiplier", 4.0))
        idxs = _sample_positions(valid_positions, spike_rate, rng)
        if idxs.size:
            current = pd.to_numeric(frame.loc[idxs, client_id], errors="coerce").to_numpy(dtype=float)
            frame.loc[idxs, client_id] = current + spike_multiplier * scale
            events.append(
                self._make_event(
                    client_id,
                    "spikes",
                    "accuracy",
                    "A1",
                    idxs,
                    timestamps,
                    n_rows,
                    {"rate": spike_rate, "multiplier": spike_multiplier},
                )
            )

        noise_rate = float(config.get("noise_rate", 0.0)) * client_factor
        noise_std_fraction = float(config.get("noise_std_fraction", 0.0))
        idxs = _sample_positions(valid_positions, noise_rate, rng)
        if idxs.size:
            noise = rng.normal(loc=0.0, scale=max(scale * noise_std_fraction, 1e-6), size=idxs.size)
            current = pd.to_numeric(frame.loc[idxs, client_id], errors="coerce").to_numpy(dtype=float)
            frame.loc[idxs, client_id] = current + noise
            events.append(
                self._make_event(
                    client_id,
                    "noisy_values",
                    "accuracy",
                    "A3",
                    idxs,
                    timestamps,
                    n_rows,
                    {"rate": noise_rate, "noise_std_fraction": noise_std_fraction},
                )
            )

    def _inject_cluster_shared_noise(
        self,
        frame: pd.DataFrame,
        all_client_cols: Sequence[str],
        cluster_assignment_map: dict[str, str],
        config: dict,
        rng: np.random.Generator,
        timestamps: pd.Series,
        events: list[FaultEvent],
    ) -> None:
        n_rows = len(frame)
        selected_clusters = _normalize_identifier_list(config.get("clusters", config.get("cluster")))
        if not selected_clusters:
            return

        available_clients = [str(client_id) for client_id in all_client_cols if str(client_id) in frame.columns]
        cluster_members: dict[str, list[str]] = {
            cluster_id: [
                client_id
                for client_id in available_clients
                if cluster_assignment_map.get(str(client_id)) == cluster_id
            ]
            for cluster_id in selected_clusters
        }
        cluster_members = {
            cluster_id: members
            for cluster_id, members in cluster_members.items()
            if members
        }
        if not cluster_members:
            raise ValueError(
                "No available dataset clients matched the configured cluster_shared_noise clusters."
            )

        events_per_cluster = int(config.get("events_per_cluster", 1))
        if events_per_cluster <= 0:
            return
        noise_std_fraction = float(config.get("noise_std_fraction", 0.0))
        if noise_std_fraction <= 0:
            return
        same_pattern_within_cluster = bool(config.get("same_pattern_within_cluster", True))
        min_length = int(config.get("min_length", 1))
        max_length = int(config.get("max_length", max(min_length, 1)))

        for cluster_id, members in cluster_members.items():
            for event_index in range(events_per_cluster):
                block = _sample_single_block(
                    n_rows=n_rows,
                    min_length=min_length,
                    max_length=max_length,
                    rng=rng,
                )
                if block.size == 0:
                    continue

                shared_pattern = (
                    rng.normal(loc=0.0, scale=1.0, size=block.size)
                    if same_pattern_within_cluster
                    else None
                )
                for client_id in members:
                    current = pd.to_numeric(frame.loc[block, client_id], errors="coerce").to_numpy(dtype=float)
                    finite_mask = np.isfinite(current)
                    if not finite_mask.any():
                        continue

                    pattern = (
                        shared_pattern
                        if shared_pattern is not None
                        else rng.normal(loc=0.0, scale=1.0, size=block.size)
                    )
                    client_values = pd.to_numeric(frame[client_id], errors="coerce").to_numpy(dtype=float)
                    client_scale = _series_scale(client_values)
                    noise = pattern[finite_mask] * max(client_scale * noise_std_fraction, 1e-6)
                    frame.loc[block[finite_mask], client_id] = current[finite_mask] + noise
                    events.append(
                        self._make_event(
                            client_id=client_id,
                            fault_type="cluster_shared_noise",
                            dimension="accuracy",
                            subcheck="A3",
                            idxs=block[finite_mask],
                            timestamps=timestamps,
                            n_rows=n_rows,
                            parameters={
                                "mode": "cluster_shared_noise",
                                "cluster_id": cluster_id,
                                "event_index": int(event_index),
                                "window_length": int(block.size),
                                "noise_std_fraction": noise_std_fraction,
                                "same_pattern_within_cluster": same_pattern_within_cluster,
                                "cluster_client_count": int(len(members)),
                            },
                            cluster_id=cluster_id,
                            scope="cluster",
                        )
                    )

    def _inject_freshness_faults(
        self,
        frame: pd.DataFrame,
        client_id: str,
        values: np.ndarray,
        valid_positions: np.ndarray,
        config: dict,
        client_factor: float,
        rng: np.random.Generator,
        timestamps: pd.Series,
        events: list[FaultEvent],
    ) -> None:
        """Freshness faults are intentionally disabled for now."""
        _ = (frame, client_id, values, valid_positions, config, client_factor, rng, timestamps, events)
        return

    def _inject_drift_faults(
        self,
        frame: pd.DataFrame,
        client_id: str,
        values: np.ndarray,
        valid_positions: np.ndarray,
        scale: float,
        distribution_cfg: dict,
        temporal_pattern_cfg: dict,
        regime_cfg: dict,
        client_factor: float,
        rng: np.random.Generator,
        timestamps: pd.Series,
        events: list[FaultEvent],
    ) -> None:
        n_rows = len(frame)
        finite = np.isfinite(values)
        if not finite.any():
            return

        phase_fraction = float(temporal_pattern_cfg.get("phase_shift_client_fraction", 0.0))
        if phase_fraction > 0 and rng.random() < min(1.0, phase_fraction * max(client_factor, 0.0)):
            shift_choices = np.asarray(temporal_pattern_cfg.get("phase_shift_steps_choices", [6, 12]), dtype=int)
            if shift_choices.size:
                shift_steps = int(rng.choice(shift_choices))
                start = int(n_rows * float(temporal_pattern_cfg.get("phase_shift_start_fraction", 0.5)))
                idxs = np.arange(start, n_rows, dtype=int)
                if idxs.size:
                    current = pd.to_numeric(frame.loc[idxs, client_id], errors="coerce").to_numpy(dtype=float)
                    finite_mask = np.isfinite(current)
                    if finite_mask.any():
                        shifted = current.copy()
                        shifted[finite_mask] = np.roll(current[finite_mask], shift_steps)
                        frame.loc[idxs, client_id] = shifted
                        events.append(
                            self._make_event(
                                client_id,
                                "phase_shift",
                                "temporal_pattern_shift",
                                "TP1",
                                idxs[finite_mask],
                                timestamps,
                                n_rows,
                                {
                                    "shift_steps": shift_steps,
                                    "start_fraction": float(temporal_pattern_cfg.get("phase_shift_start_fraction", 0.5)),
                                },
                            )
                        )

        peak_fraction = float(temporal_pattern_cfg.get("peak_shift_client_fraction", 0.0))
        if peak_fraction > 0 and rng.random() < min(1.0, peak_fraction * max(client_factor, 0.0)):
            peak_hours = {int(hour) for hour in temporal_pattern_cfg.get("peak_hours", [17, 18, 19, 20, 21])}
            multiplier = float(temporal_pattern_cfg.get("peak_shift_multiplier", 1.5))
            block = _sample_single_block(
                n_rows=n_rows,
                min_length=int(temporal_pattern_cfg.get("peak_shift_min_length", 72)),
                max_length=int(temporal_pattern_cfg.get("peak_shift_max_length", 336)),
                rng=rng,
            )
            if block.size:
                ts = pd.Series(pd.to_datetime(timestamps.iloc[block], errors="coerce"))
                block_values = pd.to_numeric(frame.loc[block, client_id], errors="coerce").to_numpy(dtype=float)
                mask = ts.dt.hour.isin(peak_hours).to_numpy() & np.isfinite(block_values)
                idxs = block[mask]
                if idxs.size:
                    frame.loc[idxs, client_id] = (
                        pd.to_numeric(frame.loc[idxs, client_id], errors="coerce").to_numpy(dtype=float) * multiplier
                    )
                    events.append(
                        self._make_event(
                            client_id,
                            "peak_hour_shift",
                            "temporal_pattern_shift",
                            "TP2",
                            idxs,
                            timestamps,
                            n_rows,
                            {
                                "multiplier": multiplier,
                                "peak_hours": sorted(peak_hours),
                                "window_length": int(block.size),
                            },
                        )
                    )

        regime_fraction = float(regime_cfg.get("mid_window_regime_shift_fraction", regime_cfg.get("train_serve_shift_fraction", 0.0)))
        if regime_fraction > 0 and rng.random() < min(1.0, regime_fraction * max(client_factor, 0.0)):
            strength = float(regime_cfg.get("mid_window_regime_shift_strength", regime_cfg.get("train_serve_shift_strength", 0.25)))
            start = int(n_rows * float(regime_cfg.get("mid_window_regime_start_fraction", 0.5)))
            idxs = np.arange(start, n_rows, dtype=int)
            if idxs.size:
                current = pd.to_numeric(frame.loc[idxs, client_id], errors="coerce").to_numpy(dtype=float)
                finite_mask = np.isfinite(current)
                if finite_mask.any():
                    local_scale = _series_scale(current[finite_mask])
                    shift = strength * max(local_scale, 1.0)
                    current[finite_mask] = current[finite_mask] + shift
                    frame.loc[idxs, client_id] = current
                    events.append(
                        self._make_event(
                            client_id,
                            "mid_window_regime_shift",
                            "regime_shift",
                            "RG1",
                            idxs[finite_mask],
                            timestamps,
                            n_rows,
                            {
                                "shift": shift,
                                "start_fraction": float(regime_cfg.get("mid_window_regime_start_fraction", 0.5)),
                            },
                        )
                    )

        mean_shift_fraction = float(distribution_cfg.get("mean_shift_client_fraction", 0.0))
        if mean_shift_fraction > 0 and rng.random() < min(1.0, mean_shift_fraction * max(client_factor, 0.0)):
            multiplier = float(distribution_cfg.get("mean_shift_std_multiplier", 1.0))
            shift = multiplier * scale
            block = _sample_single_block(
                n_rows=n_rows,
                min_length=int(distribution_cfg.get("mean_shift_min_length", 72)),
                max_length=int(distribution_cfg.get("mean_shift_max_length", 336)),
                rng=rng,
            )
            if block.size:
                shifted = values.copy()
                finite_mask = np.isfinite(shifted[block])
                if finite_mask.any():
                    shifted_block = shifted[block].copy()
                    shifted_block[finite_mask] = shifted_block[finite_mask] + shift
                    shifted[block] = shifted_block
                    frame.loc[:, client_id] = shifted
                    idxs = block[finite_mask]
                    events.append(
                        self._make_event(
                            client_id,
                            "mean_shift",
                            "distribution_shift",
                            "DS1",
                            idxs,
                            timestamps,
                            n_rows,
                            {"shift": shift, "window_length": int(block.size)},
                        )
                    )
                    values = shifted

        variance_fraction = float(distribution_cfg.get("variance_shift_client_fraction", 0.0))
        if variance_fraction > 0 and rng.random() < min(1.0, variance_fraction * max(client_factor, 0.0)):
            low, high = distribution_cfg.get("variance_scale_range", [1.5, 2.0])
            factor = float(rng.uniform(low, high))
            block = _sample_single_block(
                n_rows=n_rows,
                min_length=int(distribution_cfg.get("variance_shift_min_length", 72)),
                max_length=int(distribution_cfg.get("variance_shift_max_length", 336)),
                rng=rng,
            )
            if block.size:
                scaled = values.copy()
                finite_mask = np.isfinite(scaled[block])
                if finite_mask.any():
                    block_values = scaled[block].copy()
                    center = float(np.nanmean(block_values[finite_mask]))
                    block_values[finite_mask] = center + (block_values[finite_mask] - center) * factor
                    scaled[block] = block_values
                    frame.loc[:, client_id] = scaled
                    idxs = block[finite_mask]
                    events.append(
                        self._make_event(
                            client_id,
                            "variance_shift",
                            "distribution_shift",
                            "DS2",
                            idxs,
                            timestamps,
                            n_rows,
                            {"factor": factor, "window_length": int(block.size)},
                        )
                    )
                    values = scaled

        long_term_fraction = float(distribution_cfg.get("long_term_shift_client_fraction", 0.0))
        if long_term_fraction > 0 and rng.random() < min(1.0, long_term_fraction * max(client_factor, 0.0)):
            multiplier = float(distribution_cfg.get("long_term_shift_multiplier", 0.5))
            shift = multiplier * scale
            block = _sample_single_block(
                n_rows=n_rows,
                min_length=int(distribution_cfg.get("long_term_shift_min_length", 336)),
                max_length=int(distribution_cfg.get("long_term_shift_max_length", 720)),
                rng=rng,
            )
            if block.size:
                current = pd.to_numeric(frame.loc[block, client_id], errors="coerce").to_numpy(dtype=float)
                finite_mask = np.isfinite(current)
                current[finite_mask] = current[finite_mask] + shift
                frame.loc[block, client_id] = current
                events.append(
                    self._make_event(
                        client_id,
                        "long_term_shift",
                        "distribution_shift",
                        "DS3",
                        block[finite_mask],
                        timestamps,
                        n_rows,
                        {"shift": shift, "window_length": int(block.size)},
                    )
                )

    def _make_event(
        self,
        client_id: str,
        fault_type: str,
        dimension: str,
        subcheck: str,
        idxs: np.ndarray,
        timestamps: pd.Series,
        n_rows: int,
        parameters: dict,
        shared: bool = False,
        cluster_id: Optional[str] = None,
        scope: str = "client",
        window_start_index: Optional[int] = None,
        window_end_index: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> FaultEvent:
        if idxs.size == 0:
            start_time = None
            end_time = None
        else:
            ts_values = pd.Series(timestamps.iloc[idxs]).dropna()
            start_time = ts_values.min() if not ts_values.empty else None
            end_time = ts_values.max() if not ts_values.empty else None
        if window_start_index is None or window_end_index is None:
            window_start = None
            window_end = None
        else:
            window_ts = pd.Series(timestamps.iloc[[window_start_index, window_end_index]]).reset_index(drop=True)
            window_start = window_ts.iloc[0] if not window_ts.empty else None
            window_end = window_ts.iloc[-1] if not window_ts.empty else None
        affected_timestamps = []
        if idxs.size:
            affected_timestamps = [str(value) for value in pd.Series(timestamps.iloc[idxs]).tolist()]
        return FaultEvent(
            client_id=str(client_id),
            fault_type=fault_type,
            dimension=dimension,
            subcheck=subcheck,
            start_time=start_time,
            end_time=end_time,
            affected_rows=int(idxs.size),
            affected_fraction=float(idxs.size / max(1, n_rows)),
            parameters_used=_json_params(parameters),
            cluster_id=str(cluster_id) if cluster_id is not None else None,
            scope=str(scope),
            shared_timestamp_event=shared,
            window_start=window_start,
            window_end=window_end,
            window_start_index=int(window_start_index) if window_start_index is not None else None,
            window_end_index=int(window_end_index) if window_end_index is not None else None,
            affected_positions=_json_params([int(idx) for idx in idxs.tolist()]),
            affected_timestamps=_json_params(affected_timestamps),
            seed=int(seed) if seed is not None else None,
        )


def inject_faults_to_csv(
    input_path: Path | str,
    output_path: Path | str,
    report_path: Path | str,
    config: Optional[dict] = None,
    timestamp_col: Optional[str] = None,
    seed: int = 42,
    cluster_assignment_path: Optional[Path | str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a clean demand CSV, inject faults, and save outputs."""
    dataset = load_wide_demand_csv(input_path, timestamp_col=timestamp_col)
    injector = WideDemandFaultInjector(config=config)
    faulty_df, report_df = injector.inject(
        dataset,
        seed=seed,
        cluster_assignment_path=cluster_assignment_path,
    )
    save_csv(faulty_df, output_path)
    save_csv(report_df, report_path)
    if config and config.get("fault_mode") == "controlled_window":
        summary_df = getattr(injector, "last_controlled_window_summary_df", pd.DataFrame())
        report_path_obj = Path(report_path)
        if summary_df is not None and not summary_df.empty:
            summary_path = config.get("controlled_window_summary_path")
            if summary_path is None:
                summary_path = report_path_obj.with_name(f"{report_path_obj.stem}_summary{report_path_obj.suffix}")
            save_csv(summary_df, summary_path)
        selected_clients_df = getattr(injector, "last_controlled_selected_clients_df", pd.DataFrame())
        if selected_clients_df is not None and not selected_clients_df.empty:
            selected_path = config.get("controlled_selected_clients_path")
            if selected_path is None:
                selected_path = report_path_obj.with_name("selected_clients.csv")
            save_csv(selected_clients_df, selected_path)
    return faulty_df, report_df
