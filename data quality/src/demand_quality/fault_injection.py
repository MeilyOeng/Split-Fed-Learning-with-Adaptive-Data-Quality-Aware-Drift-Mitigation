"""Fault injection for wide-format demand data."""

from __future__ import annotations

import json
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
    shared_timestamp_event: bool = False


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


class WideDemandFaultInjector:
    """Injects structured faults into a wide-format demand table."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def inject(
        self,
        dataset: WideDemandDataset,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Inject configured faults into a wide demand dataset."""
        rng = np.random.default_rng(seed)
        frame = dataset.frame.copy(deep=True)
        frame[_ORIG_ROW_ID_COL] = np.arange(len(frame), dtype=int)
        frame[_IS_DUPLICATE_COL] = False
        ts_col = dataset.timestamp_col
        client_cols = self._select_clients(dataset.client_cols, rng)
        for client_id in dataset.client_cols:
            frame[client_id] = pd.to_numeric(frame[client_id], errors="coerce").astype(float)
        original_frame = frame.copy(deep=True)
        n_rows = len(frame)
        timestamps = frame[ts_col].reset_index(drop=True)

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
        clean_frame = frame.drop(columns=[_ORIG_ROW_ID_COL, _IS_DUPLICATE_COL], errors="ignore")
        return clean_frame, report_df

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
    ) -> FaultEvent:
        if idxs.size == 0:
            start_time = None
            end_time = None
        else:
            ts_values = pd.Series(timestamps.iloc[idxs]).dropna()
            start_time = ts_values.min() if not ts_values.empty else None
            end_time = ts_values.max() if not ts_values.empty else None
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
            shared_timestamp_event=shared,
        )


def inject_faults_to_csv(
    input_path: Path | str,
    output_path: Path | str,
    report_path: Path | str,
    config: Optional[dict] = None,
    timestamp_col: Optional[str] = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a clean demand CSV, inject faults, and save outputs."""
    dataset = load_wide_demand_csv(input_path, timestamp_col=timestamp_col)
    injector = WideDemandFaultInjector(config=config)
    faulty_df, report_df = injector.inject(dataset, seed=seed)
    save_csv(faulty_df, output_path)
    save_csv(report_df, report_path)
    return faulty_df, report_df
