"""Offline preparation and analysis for the controlled 2x2 drift/fault study.

The module materializes condition inputs, computes DQ artifacts, and joins all
diagnostics offline.  It deliberately does not integrate oracle or DQ state
with the runtime stream.
"""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .contracts import DriftConfig, DriftSource, DriftType
from .framework import transition_progress
from .injector import inject_series
from .oracle import build_oracle_tables, validate_oracle_tables


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DQ_ROOT = PROJECT_ROOT / "data quality"
if str(DQ_ROOT) not in sys.path:
    sys.path.insert(0, str(DQ_ROOT))

# These are existing implementations.  Importing the internal row builder lets
# the diagnostic path restrict work to test windows without changing formulas.
from modules.io import (  # noqa: E402
    build_wide_demand_diagnostics,
    load_wide_demand_dataframe,
    load_yaml_config,
)
from modules.label_dataset import _build_window_row, _runtime_split_bounds  # noqa: E402
from modules.pca_quality import compute_average_quality_score, final_score_method  # noqa: E402
from modules.scoring import WideDemandQualityScorer  # noqa: E402
from modules.schema import WideDemandDataset  # noqa: E402
from modules.windowing import (  # noqa: E402
    ClientWindowSlice,
    load_window_source,
)
from modules.fault_injection import WideDemandFaultInjector  # noqa: E402

from data.clustering_processing import normalize_id, preprocess_clusters  # noqa: E402


CONDITIONS: dict[str, dict[str, bool]] = {
    "A": {"true_drift": False, "fault_injection": False},
    "B": {"true_drift": True, "fault_injection": False},
    "C": {"true_drift": False, "fault_injection": True},
    "D": {"true_drift": True, "fault_injection": True},
}

DQ_COLUMNS = [
    "quality_score",
    "dimension_completeness",
    "dimension_validity",
    "dimension_consistency",
    "dimension_accuracy",
    "dimension_freshness",
]

FORECAST_METRICS = ["mae", "mse", "rmse", "smape", "r2"]

CONTRASTS: dict[str, dict[str, float]] = {
    "drift_effect_B_minus_A": {"B": 1.0, "A": -1.0},
    "fault_effect_C_minus_A": {"C": 1.0, "A": -1.0},
    "combined_effect_D_minus_A": {"D": 1.0, "A": -1.0},
    "fault_effect_under_drift_D_minus_B": {"D": 1.0, "B": -1.0},
    "drift_effect_under_faults_D_minus_C": {"D": 1.0, "C": -1.0},
    "interaction_D_minus_B_minus_C_plus_A": {"D": 1.0, "B": -1.0, "C": -1.0, "A": 1.0},
}


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_rows_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    digest = hashlib.sha256()
    if frame.empty:
        return digest.hexdigest()
    selected = frame.loc[:, list(columns)].copy()
    selected = selected.sort_values(list(columns), kind="mergesort").reset_index(drop=True)
    digest.update(selected.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    return digest.hexdigest()


def _resolve_path(raw: object, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else (base / path).resolve()


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"diagnostic config must contain a mapping: {config_path}")
    config = copy.deepcopy(config)
    config["_config_path"] = str(config_path)
    for key in (
        "clean_wide_csv",
        "cluster_assignment_csv",
        "metadata_csv",
        "clean_cluster_dir",
        "selected_clients_csv",
        "fault_config",
        "dq_score_config",
        "baseline_checkpoint_dir",
    ):
        config[key] = str(_resolve_path(config[key]))
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: Mapping[str, object]) -> None:
    required_paths = (
        "clean_wide_csv",
        "cluster_assignment_csv",
        "metadata_csv",
        "clean_cluster_dir",
        "selected_clients_csv",
        "fault_config",
        "dq_score_config",
        "baseline_checkpoint_dir",
    )
    for key in required_paths:
        path = Path(str(config[key]))
        if not path.exists():
            raise FileNotFoundError(f"{key} does not exist: {path}")
    expected = {
        "window_size": 24,
        "forecast_horizon": 1,
        "test_window_start_first": 6988,
        "test_window_start_last": 8735,
        "test_input_observation_first": 6988,
        "test_input_observation_last": 8758,
        "test_detector_endpoint_first": 7011,
        "test_detector_endpoint_last": 8758,
        "last_clean_target_index": 8759,
        "drift_onset_index": 7885,
    }
    coordinates = dict(config.get("coordinates", {}) or {})
    for key, value in expected.items():
        actual = int(config.get(key, coordinates.get(key, -1)))
        if actual != value:
            raise ValueError(f"{key} must be {value}, got {actual}")
    split = dict(config.get("split", {}) or {})
    if (float(split.get("train", -1)), float(split.get("val", -1)), float(split.get("test", -1))) != (
        0.7,
        0.1,
        0.2,
    ):
        raise ValueError("split must be exactly train=0.7, val=0.1, test=0.2")
    drift = dict(config.get("drift", {}) or {})
    required_drift = {
        "type": "sudden",
        "source": "input",
        "scope": "test",
        "start_ratio": 0.5,
        "magnitude": 0.5,
        "offset": 0,
        "noise": 0,
        "seed": 42,
    }
    for key, value in required_drift.items():
        if drift.get(key) != value:
            raise ValueError(f"drift.{key} must be {value!r}, got {drift.get(key)!r}")
    detector = dict(config.get("detectors", {}) or {})
    expected_detector = {
        "PH_MIN_INSTANCES": 500,
        "PH_DELTA": 0.02,
        "PH_THRESHOLD": 450,
        "PH_ALPHA": 0.9999,
        "ADWIN_DELTA": 1e-16,
        "ADWIN_CLOCK": 32,
        "ADWIN_MAX_BUCKETS": 5,
        "ADWIN_MIN_WINDOW_LENGTH": 192,
        "ADWIN_GRACE_PERIOD": 500,
    }
    for key, value in expected_detector.items():
        if float(detector.get(key, math.nan)) != float(value):
            raise ValueError(f"detectors.{key} must be {value!r}, got {detector.get(key)!r}")


def _coordinate(config: Mapping[str, object], name: str) -> int:
    if name in config:
        return int(config[name])
    return int(dict(config["coordinates"])[name])


def load_cohort(config: Mapping[str, object], *, smoke: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = pd.read_csv(str(config["clean_wide_csv"]), low_memory=False)
    timestamp_col = str(config.get("timestamp_column", "Fecha_Hora"))
    if timestamp_col not in clean.columns:
        raise ValueError(f"clean input is missing timestamp column {timestamp_col}")
    clean[timestamp_col] = pd.to_datetime(clean[timestamp_col], errors="raise")
    client_cols = [
        str(column)
        for column in clean.columns
        if column != timestamp_col and not str(column).startswith("__dq_")
    ]
    clean = clean.rename(columns={column: normalize_id(column) for column in client_cols})
    client_cols = [normalize_id(column) for column in client_cols]
    if len(client_cols) != 84:
        raise AssertionError(f"E3 clean cohort must contain 84 clients, got {len(client_cols)}")

    selected_source = pd.read_csv(str(config["selected_clients_csv"])).copy()
    selected_source["client_id"] = selected_source["client_id"].map(normalize_id)
    selected_source["cluster_id"] = selected_source["cluster_id"].map(normalize_id)
    selected_flag = selected_source["selected"].astype(str).str.lower().isin({"1", "true", "yes"})
    selected = selected_source.loc[selected_flag, ["cluster_id", "client_id"]].copy()
    if len(selected) != 32 or selected["client_id"].nunique() != 32:
        raise AssertionError(f"E3 affected cohort must contain 32 unique clients, got {len(selected)}")
    if selected.groupby("cluster_id")["client_id"].nunique().to_dict() != {
        "1": 8,
        "2": 8,
        "3": 8,
        "4": 8,
    }:
        raise AssertionError("E3 affected cohort must contain exactly eight clients per cluster")
    if not set(selected["client_id"]).issubset(client_cols):
        raise AssertionError("selected client manifest contains clients absent from clean_84_clients.csv")

    assignments = pd.read_csv(str(config["cluster_assignment_csv"]))
    assignments["client_id"] = assignments["ID"].map(normalize_id)
    assignments["cluster_id"] = assignments["Cluster"].map(normalize_id)
    cohort = assignments.loc[assignments["client_id"].isin(client_cols), ["cluster_id", "client_id"]].copy()
    cohort = cohort.drop_duplicates("client_id").sort_values(["cluster_id", "client_id"]).reset_index(drop=True)
    if len(cohort) != 84:
        raise AssertionError(f"cluster assignments must cover all 84 clients, got {len(cohort)}")
    selected_ids = set(selected["client_id"])
    cohort["affected_client"] = cohort["client_id"].isin(selected_ids)
    if smoke:
        cohort = cohort.loc[cohort["affected_client"]].reset_index(drop=True)
        keep = [timestamp_col, *cohort["client_id"].tolist()]
        clean = clean.loc[:, keep].copy()
    return clean, cohort


def condition_clients_manifest(cohort: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "cluster_id": str(row.cluster_id),
            "client_id": str(row.client_id),
            "affected_client": bool(row.affected_client),
        }
        for row in cohort.itertuples(index=False)
    ]


def _shift_json_positions(raw: object, offset: int) -> str:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)) or str(raw).strip() == "":
        return "[]"
    values = json.loads(str(raw))
    return json.dumps([int(value) + offset for value in values], separators=(",", ":"))


def build_test_fault_template(
    clean: pd.DataFrame,
    cohort: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the existing controlled injector on exactly the test-input union."""
    timestamp_col = str(config.get("timestamp_column", "Fecha_Hora"))
    first_obs = _coordinate(config, "test_input_observation_first")
    last_obs = _coordinate(config, "test_input_observation_last")
    selected = cohort.loc[cohort["affected_client"], "client_id"].astype(str).tolist()
    fault_config = load_yaml_config(str(config["fault_config"]))
    fault_config = copy.deepcopy(fault_config)
    fault_config["clients"] = {
        **dict(fault_config.get("clients", {}) or {}),
        "selection_mode": "explicit",
        "selected": selected,
        "random_count": None,
        "random_fraction": None,
    }
    fault_config["restore_expected_timestamps_after_injection"] = True
    fault_slice = clean.iloc[first_obs : last_obs + 1].reset_index(drop=True).copy()
    dataset = load_wide_demand_dataframe(
        fault_slice,
        timestamp_col=timestamp_col,
        configured_freq="1h",
    )
    injector = WideDemandFaultInjector(config=fault_config)
    faulty_slice, events = injector.inject(
        dataset,
        seed=int(dict(config["drift"])["seed"]),
        cluster_assignment_path=str(config["cluster_assignment_csv"]),
    )
    if len(faulty_slice) != last_obs - first_obs + 1:
        raise AssertionError("controlled fault injector did not preserve the test timestamp grid")

    selected_actual = getattr(injector, "last_controlled_selected_clients_df", pd.DataFrame()).copy()
    actual_ids = set(
        selected_actual.loc[
            selected_actual["selected"].astype(str).str.lower().isin({"1", "true", "yes"}), "client_id"
        ].astype(str)
    )
    if actual_ids != set(selected):
        raise AssertionError("controlled injector selected clients differ from the fixed E3 cohort")

    shifted = events.copy()
    for column in ("window_start_index", "window_end_index"):
        if column in shifted:
            shifted[f"{column}_local"] = shifted[column]
            shifted[column] = pd.to_numeric(shifted[column], errors="coerce").map(
                lambda value: int(value) + first_obs if pd.notna(value) else value
            )
    if "affected_positions" in shifted:
        shifted["affected_positions_local"] = shifted["affected_positions"]
        shifted["affected_positions"] = shifted["affected_positions"].map(
            lambda raw: _shift_json_positions(raw, first_obs)
        )

    full_fault = clean.copy()
    for metadata_col in [column for column in faulty_slice.columns if str(column).startswith("__dq_")]:
        full_fault[metadata_col] = 0
        full_fault.loc[first_obs:last_obs, metadata_col] = faulty_slice[metadata_col].to_numpy()
    for client_id in selected:
        full_fault.loc[first_obs:last_obs, client_id] = pd.to_numeric(
            faulty_slice[client_id], errors="coerce"
        ).to_numpy()

    summary = getattr(injector, "last_controlled_window_summary_df", pd.DataFrame()).copy()
    return full_fault, shifted, summary, selected_actual


def _make_drift_config(
    condition_id: str,
    cluster_id: str,
    client_id: str,
    config: Mapping[str, object],
    *,
    series_length: int,
) -> DriftConfig:
    drift = dict(config["drift"])
    return DriftConfig(
        run_id=f"{config['experiment_id']}_{condition_id}",
        configuration_id=f"{config['experiment_id']}_{condition_id}",
        cluster_id=str(cluster_id),
        client_id=str(client_id),
        mode="test",
        scope="test",
        drift_type=DriftType.SUDDEN,
        drift_source=DriftSource.INPUT,
        start_index=_coordinate(config, "drift_onset_index"),
        magnitude=float(drift["magnitude"]),
        offset_delta=float(drift["offset"]),
        noise_sigma=float(drift["noise"]),
        random_seed=int(drift["seed"]),
        requested_start_ratio=float(drift["start_ratio"]),
        segment_start_index=_coordinate(config, "test_detector_endpoint_first"),
        segment_end_index_exclusive=_coordinate(config, "test_detector_endpoint_last") + 1,
    )


def materialize_drift(
    post_fault: pd.DataFrame,
    cohort: pd.DataFrame,
    config: Mapping[str, object],
    *,
    condition_id: str,
    enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize G(x) through drift.injector and preserve fault NaNs."""
    result_frame = post_fault.copy()
    first_obs = _coordinate(config, "test_input_observation_first")
    last_obs = _coordinate(config, "test_input_observation_last")
    affected_rows = cohort.loc[cohort["affected_client"]]
    schedule_rows: list[dict[str, object]] = []
    for row in affected_rows.itertuples(index=False):
        client_id = str(row.client_id)
        values = pd.to_numeric(post_fault[client_id], errors="coerce").to_numpy(dtype=float)
        missing = ~np.isfinite(values)
        if enabled:
            # inject_series validates finite input.  A zero surrogate is used at
            # missing cells and those cells are restored to NaN immediately.
            finite_surrogate = np.where(missing, 0.0, values)
            drift_config = _make_drift_config(
                condition_id,
                str(row.cluster_id),
                client_id,
                config,
                series_length=len(values),
            )
            injection = inject_series(
                finite_surrogate,
                drift_config,
                timestamps=post_fault[str(config.get("timestamp_column", "Fecha_Hora"))].tolist(),
            )
            runtime = np.asarray(injection.runtime_values, dtype=float).copy()
            runtime[missing] = np.nan
            result_frame[client_id] = runtime
            for record in injection.records[first_obs : last_obs + 1]:
                expected_weight = transition_progress(record.observation_index, drift_config)
                if record.drift_weight != expected_weight:
                    raise AssertionError("injector record disagrees with drift.framework transition")
                schedule_rows.append(
                    {
                        "condition_id": condition_id,
                        "cluster_id": str(row.cluster_id),
                        "client_id": client_id,
                        "observation_index": int(record.observation_index),
                        "timestamp": record.timestamp,
                        "drift_weight": float(record.drift_weight),
                        "regime_active": bool(record.regime_active),
                        "transform_applied": bool(record.transform_applied and not missing[record.observation_index]),
                        "missing_preserved": bool(missing[record.observation_index]),
                        "configured_start_index": int(drift_config.start_index),
                        "magnitude": float(drift_config.magnitude),
                        "offset_delta": float(drift_config.offset_delta),
                        "noise_sigma": float(drift_config.noise_sigma),
                        "random_seed": int(drift_config.random_seed),
                        "implementation": "drift.injector.inject_series->drift.framework",
                    }
                )
        else:
            for observation_index in range(first_obs, last_obs + 1):
                schedule_rows.append(
                    {
                        "condition_id": condition_id,
                        "cluster_id": str(row.cluster_id),
                        "client_id": client_id,
                        "observation_index": observation_index,
                        "timestamp": pd.Timestamp(
                            post_fault[str(config.get("timestamp_column", "Fecha_Hora"))].iloc[observation_index]
                        ).isoformat(),
                        "drift_weight": 0.0,
                        "regime_active": False,
                        "transform_applied": False,
                        "missing_preserved": bool(missing[observation_index]),
                        "configured_start_index": None,
                        "magnitude": 0.0,
                        "offset_delta": 0.0,
                        "noise_sigma": 0.0,
                        "random_seed": int(dict(config["drift"])["seed"]),
                        "implementation": "disabled",
                    }
                )
    return result_frame, pd.DataFrame(schedule_rows)


def _different(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_nan = ~np.isfinite(left)
    right_nan = ~np.isfinite(right)
    return (left_nan != right_nan) | (
        ~(left_nan | right_nan) & ~np.isclose(left, right, rtol=0.0, atol=1e-12)
    )


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def build_value_lineage(
    clean: pd.DataFrame,
    post_fault: pd.DataFrame,
    post_drift: pd.DataFrame,
    cohort: pd.DataFrame,
    config: Mapping[str, object],
    *,
    condition_id: str,
) -> pd.DataFrame:
    timestamp_col = str(config.get("timestamp_column", "Fecha_Hora"))
    first_obs = _coordinate(config, "test_input_observation_first")
    last_obs = _coordinate(config, "test_input_observation_last")
    rows: list[pd.DataFrame] = []
    for client in cohort.itertuples(index=False):
        client_id = str(client.client_id)
        clean_values = pd.to_numeric(clean[client_id], errors="coerce").to_numpy(dtype=float)
        fault_values = pd.to_numeric(post_fault[client_id], errors="coerce").to_numpy(dtype=float)
        drift_values = pd.to_numeric(post_drift[client_id], errors="coerce").to_numpy(dtype=float)
        runtime_values = pd.Series(drift_values).ffill().bfill().to_numpy(dtype=float)
        positions = np.arange(first_obs, last_obs + 1, dtype=int)
        rows.append(
            pd.DataFrame(
                {
                    "condition_id": condition_id,
                    "cluster_id": str(client.cluster_id),
                    "client_id": client_id,
                    "affected_client": bool(client.affected_client),
                    "observation_index": positions,
                    "timestamp": [
                        pd.Timestamp(value).isoformat()
                        for value in clean[timestamp_col].iloc[first_obs : last_obs + 1]
                    ],
                    "clean_value": clean_values[first_obs : last_obs + 1],
                    "post_fault_value": fault_values[first_obs : last_obs + 1],
                    "post_drift_value": drift_values[first_obs : last_obs + 1],
                    "model_runtime_value": runtime_values[first_obs : last_obs + 1],
                    "fault_applied": _different(clean_values, fault_values)[first_obs : last_obs + 1],
                    "drift_applied": _different(fault_values, drift_values)[first_obs : last_obs + 1],
                    "runtime_imputed": _different(drift_values, runtime_values)[first_obs : last_obs + 1],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _event_observation_map(events: pd.DataFrame) -> dict[str, dict[int, dict[str, set[str]]]]:
    result: dict[str, dict[int, dict[str, set[str]]]] = {}
    if events.empty:
        return result
    for row in events.to_dict(orient="records"):
        client_id = normalize_id(row["client_id"])
        if client_id.startswith("__"):
            continue
        positions = json.loads(str(row.get("affected_positions") or "[]"))
        for position in positions:
            entry = result.setdefault(client_id, {}).setdefault(
                int(position), {"dimensions": set(), "fault_types": set(), "subchecks": set()}
            )
            entry["dimensions"].add(str(row.get("dimension", "")))
            entry["fault_types"].add(str(row.get("fault_type", "")))
            entry["subchecks"].add(str(row.get("subcheck", "")))
    return result


def build_fault_window_labels(
    cohort: pd.DataFrame,
    events: pd.DataFrame,
    config: Mapping[str, object],
    *,
    condition_id: str,
    enabled: bool,
) -> pd.DataFrame:
    first_start = _coordinate(config, "test_window_start_first")
    last_start = _coordinate(config, "test_window_start_last")
    window_size = _coordinate(config, "window_size")
    event_map = _event_observation_map(events) if enabled else {}
    rows: list[dict[str, object]] = []
    for client in cohort.itertuples(index=False):
        client_events = event_map.get(str(client.client_id), {})
        event_positions = sorted(client_events)
        for start in range(first_start, last_start + 1):
            end = start + window_size - 1
            positions = [position for position in event_positions if start <= position <= end]
            history = [position for position in event_positions if position <= end]
            dimensions: set[str] = set()
            fault_types: set[str] = set()
            subchecks: set[str] = set()
            for position in positions:
                dimensions.update(client_events[position]["dimensions"])
                fault_types.update(client_events[position]["fault_types"])
                subchecks.update(client_events[position]["subchecks"])
            rows.append(
                {
                    "condition_id": condition_id,
                    "cluster_id": str(client.cluster_id),
                    "client_id": str(client.client_id),
                    "mode": "test",
                    "window_start_index": start,
                    "window_end_index": end,
                    "affected_client": bool(client.affected_client),
                    "fault_at_endpoint": end in client_events,
                    "fault_in_window": bool(positions),
                    "fault_seen_in_history": bool(history),
                    "fault_observation_count": len(positions),
                    "fault_observation_fraction": len(positions) / window_size,
                    "fault_dimensions": "|".join(sorted(value for value in dimensions if value)),
                    "fault_types": "|".join(sorted(value for value in fault_types if value)),
                    "fault_subchecks": "|".join(sorted(value for value in subchecks if value)),
                }
            )
    return pd.DataFrame(rows)


def _raw_client_slice(
    frame: pd.DataFrame,
    timestamp_col: str,
    client_id: str,
    start: int,
    end: int,
    *,
    include_metadata: bool = True,
) -> pd.DataFrame:
    columns = [timestamp_col, client_id]
    if include_metadata:
        columns.extend(column for column in frame.columns if str(column).startswith("__dq_"))
    return frame.iloc[start : end + 1].loc[:, columns].copy()


def _wrap_aligned_dq_window(
    raw_window: pd.DataFrame,
    *,
    timestamp_col: str,
    client_id: str,
    frequency: pd.Timedelta,
) -> WideDemandDataset:
    """Wrap an already sorted/aligned window using existing DQ diagnostics."""
    frame = raw_window.copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="raise")
    frame[client_id] = pd.to_numeric(frame[client_id], errors="coerce")
    return WideDemandDataset(
        frame=frame,
        timestamp_col=timestamp_col,
        client_cols=[client_id],
        frequency=frequency,
        diagnostics=build_wide_demand_diagnostics(
            raw_window,
            timestamp_col=timestamp_col,
            frequency=frequency,
        ),
    )


def build_test_dq_artifact(
    *,
    clean_path: str | Path,
    current_path: str | Path,
    output_path: str | Path,
    score_config_path: str | Path,
    clients: Sequence[str],
    config: Mapping[str, object],
    condition_id: str,
    progress_every: int = 5000,
    workers: int = 1,
) -> pd.DataFrame:
    """Score only test windows while reusing all existing DQ formulas."""
    workers = max(1, int(workers))
    client_list = list(map(str, clients))
    if workers > 1 and len(client_list) > 1:
        partitions = [
            [client_list[index] for index in range(partition, len(client_list), workers)]
            for partition in range(workers)
        ]
        partitions = [partition for partition in partitions if partition]
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent, prefix="dq_parts_") as temp_dir:
            arguments = []
            for index, partition in enumerate(partitions):
                arguments.append(
                    {
                        "clean_path": str(clean_path),
                        "current_path": str(current_path),
                        "output_path": str(Path(temp_dir) / f"part_{index:03d}.csv"),
                        "score_config_path": str(score_config_path),
                        "clients": partition,
                        "config": dict(config),
                        "condition_id": condition_id,
                        "progress_every": 0,
                        "workers": 1,
                    }
                )
            with ProcessPoolExecutor(max_workers=len(arguments)) as executor:
                part_paths = list(executor.map(_build_dq_partition, arguments))
            artifact = pd.concat(
                [pd.read_csv(path, dtype={"client_id": str}) for path in part_paths],
                ignore_index=True,
            )
        artifact = artifact.sort_values(
            ["client_id", "window_start_index"], kind="mergesort"
        ).reset_index(drop=True)
        artifact.to_csv(target, index=False)
        return artifact

    score_config = load_yaml_config(score_config_path)
    if final_score_method(score_config) != "average":
        raise ValueError("diagnostic_2x2 requires the approved average final DQ score")
    timestamp_col = str(config.get("timestamp_column", "Fecha_Hora"))
    source = load_window_source(
        clean_path,
        current_path,
        timestamp_col=timestamp_col,
        configured_freq=score_config.get("frequency", "1h"),
        selected_clients=list(clients),
    )
    expected = pd.DatetimeIndex(source.expected_grid)
    clean_timestamps = pd.DatetimeIndex(pd.to_datetime(source.clean_raw[timestamp_col], errors="raise"))
    current_timestamps = pd.DatetimeIndex(pd.to_datetime(source.current_raw[timestamp_col], errors="raise"))
    if not expected.equals(clean_timestamps) or not expected.equals(current_timestamps):
        raise AssertionError("offline DQ optimization requires the restored one-row-per-expected-timestamp grid")
    source.clean_raw[timestamp_col] = clean_timestamps
    source.current_raw[timestamp_col] = current_timestamps
    for client_id in client_list:
        source.clean_raw[client_id] = pd.to_numeric(source.clean_raw[client_id], errors="coerce")
        source.current_raw[client_id] = pd.to_numeric(source.current_raw[client_id], errors="coerce")

    scorer = WideDemandQualityScorer(config=score_config)
    split = dict(config["split"])
    window_size = _coordinate(config, "window_size")
    horizon = _coordinate(config, "forecast_horizon")
    first_start = _coordinate(config, "test_window_start_first")
    last_start = _coordinate(config, "test_window_start_last")
    split_bounds = _runtime_split_bounds(
        len(expected),
        window_size_steps=window_size,
        forecast_horizon=horizon,
        train_ratio=float(split["train"]),
        val_ratio=float(split["val"]),
        test_ratio=float(split["test"]),
    )
    rows: list[dict[str, float | str]] = []
    started = time.monotonic()
    for client_id in client_list:
        for start in range(first_start, last_start + 1):
            end = start + window_size - 1
            window_ts = expected[start : end + 1]
            clean_raw_window = _raw_client_slice(source.clean_raw, timestamp_col, client_id, start, end)
            current_raw_window = _raw_client_slice(source.current_raw, timestamp_col, client_id, start, end)
            clean_history = _raw_client_slice(
                source.clean_raw, timestamp_col, client_id, 0, end, include_metadata=False
            )
            current_history = _raw_client_slice(
                source.current_raw, timestamp_col, client_id, 0, end, include_metadata=False
            )
            window = ClientWindowSlice(
                client_id=client_id,
                window_id=(
                    f"{client_id}__{pd.Timestamp(window_ts[0]).isoformat().replace(':', '-')}"
                    f"__{pd.Timestamp(window_ts[-1]).isoformat().replace(':', '-')}"
                ),
                window_index=end,
                window_start_index=start,
                window_end_index=end,
                window_start=pd.Timestamp(window_ts[0]),
                window_end=pd.Timestamp(window_ts[-1]),
                timestamp_col=timestamp_col,
                frequency=source.frequency,
                expected_timestamps=window_ts,
                clean_raw_window=clean_raw_window,
                current_raw_window=current_raw_window,
                clean_raw_history=clean_history,
                current_raw_history=current_history,
                clean_window=_wrap_aligned_dq_window(
                    clean_raw_window,
                    timestamp_col=timestamp_col,
                    client_id=client_id,
                    frequency=source.frequency,
                ),
                current_window=_wrap_aligned_dq_window(
                    current_raw_window,
                    timestamp_col=timestamp_col,
                    client_id=client_id,
                    frequency=source.frequency,
                ),
            )
            rows.append(
                _build_window_row(
                    window,
                    scorer=scorer,
                    score_config=score_config,
                    baseline_artifact=None,
                    include_features=False,
                    split_bounds=split_bounds,
                )
            )
            if progress_every and len(rows) % progress_every == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"[diagnostic_2x2][DQ][{condition_id}] rows={len(rows)} "
                    f"windows_per_s={len(rows) / elapsed:.1f}",
                    file=sys.stderr,
                    flush=True,
                )
    artifact = pd.DataFrame(rows)
    dimensions = pd.DataFrame(
        {
            "completeness": artifact["y_completeness"].astype(float),
            "validity": artifact["y_validity"].astype(float),
            "consistency": artifact["y_consistency"].astype(float),
            "accuracy": artifact["y_accuracy"].astype(float),
            "freshness": artifact["y_freshness"].astype(float),
        }
    )
    artifact["quality_score"] = compute_average_quality_score(dimensions, score_config)
    artifact["dimension_completeness"] = artifact["y_completeness"].astype(float)
    artifact["dimension_validity"] = artifact["y_validity"].astype(float)
    artifact["dimension_consistency"] = artifact["y_consistency"].astype(float)
    artifact["dimension_accuracy"] = artifact["y_accuracy"].astype(float)
    artifact["dimension_freshness"] = artifact["y_freshness"].astype(float)
    artifact["score_method"] = "average"
    artifact["condition_id"] = condition_id
    artifact["condition_input_sha256"] = sha256_file(current_path)
    artifact["clean_reference_sha256"] = sha256_file(clean_path)
    artifact["dq_formula_source"] = "modules.label_dataset._build_window_row"
    output_columns = [
        "condition_id",
        "client_id",
        "mode",
        "window_id",
        "window_index",
        "window_start_index",
        "window_end_index",
        "window_start",
        "window_end",
        *DQ_COLUMNS,
        "offline_F1",
        "offline_F2",
        "score_method",
        "condition_input_sha256",
        "clean_reference_sha256",
        "dq_formula_source",
    ]
    artifact = artifact.loc[:, output_columns]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    artifact.to_csv(target, index=False)
    return artifact


def _build_dq_partition(arguments: Mapping[str, object]) -> str:
    output_path = str(arguments["output_path"])
    build_test_dq_artifact(**dict(arguments))
    return output_path


def build_condition_manifest(
    config: Mapping[str, object],
    cohort: pd.DataFrame,
    *,
    condition_id: str,
    smoke: bool,
    condition_input_path: Path,
    cluster_input_dir: Path,
    dq_path: Path,
    fault_labels_path: Path,
    drift_schedule_path: Path,
) -> dict[str, object]:
    factors = CONDITIONS[condition_id]
    affected = cohort.loc[cohort["affected_client"], "client_id"].astype(str).tolist()
    return {
        "experiment_id": str(config["experiment_id"]),
        "condition_id": condition_id,
        "execution_mode": "smoke_32_clients" if smoke else "primary_84_clients",
        "client_count": int(len(cohort)),
        "affected_client_count": int(cohort["affected_client"].sum()),
        "clean_control_count": int((~cohort["affected_client"]).sum()),
        "clients": condition_clients_manifest(cohort),
        "affected_clients": affected,
        "true_drift": bool(factors["true_drift"]),
        "fault_injection": bool(factors["fault_injection"]),
        "condition_equation": {
            "A": "x",
            "B": "G(x)",
            "C": "F(x)",
            "D": "G(F(x))",
        }[condition_id],
        "transformation_order": [
            "clean_raw_series",
            "existing_controlled_test_only_fault_injection" if factors["fault_injection"] else "fault_stage_identity",
            "drift.injector.inject_series_using_drift.framework" if factors["true_drift"] else "drift_stage_identity",
            "existing_stream_ffill_bfill",
            "unchanged_clean_training_normalization",
        ],
        "window_size": _coordinate(config, "window_size"),
        "forecast_horizon": _coordinate(config, "forecast_horizon"),
        "test_window_start_first": _coordinate(config, "test_window_start_first"),
        "test_window_start_last": _coordinate(config, "test_window_start_last"),
        "test_input_observation_first": _coordinate(config, "test_input_observation_first"),
        "test_input_observation_last": _coordinate(config, "test_input_observation_last"),
        "test_detector_endpoint_first": _coordinate(config, "test_detector_endpoint_first"),
        "test_detector_endpoint_last": _coordinate(config, "test_detector_endpoint_last"),
        "last_clean_target_index": _coordinate(config, "last_clean_target_index"),
        "drift_onset_index": _coordinate(config, "drift_onset_index"),
        "condition_input_path": str(condition_input_path),
        "condition_input_sha256": sha256_file(condition_input_path),
        "cluster_input_dir": str(cluster_input_dir),
        "dq_scores_path": str(dq_path),
        "dq_scores_sha256": sha256_file(dq_path),
        "fault_window_labels_path": str(fault_labels_path),
        "fault_window_labels_sha256": sha256_file(fault_labels_path),
        "drift_schedule_path": str(drift_schedule_path),
        "drift_schedule_sha256": sha256_file(drift_schedule_path),
        "oracle_execution": "offline_only",
        "oracle_controls_runtime": False,
        "dq_execution": "offline_only",
        "dq_controls_runtime": False,
        "runtime_drift_injection_enabled": False,
        "model_affecting_dq_flags": {
            "USE_QUALITY_WEIGHTS": False,
            "USE_QUALITY_AS_INPUT": False,
            "USE_QUALITY_AGGREGATION_WEIGHTS": False,
            "USE_QUALITY_ANCHOR_AGGREGATION": False,
            "USE_GRADIENT_ALIGNMENT_FILTER": False,
            "USE_HIGH_QUALITY_VALIDATION_GUARDRAIL": False,
        },
    }


def prepare_experiment(
    config: Mapping[str, object],
    output_root: str | Path,
    *,
    smoke: bool = False,
    dq_progress_every: int = 5000,
) -> dict[str, object]:
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"refusing to reuse existing experiment root: {root}")
    root.mkdir(parents=True)
    prepared = root / "prepared"
    prepared.mkdir()

    clean, cohort = load_cohort(config, smoke=smoke)
    timestamp_col = str(config.get("timestamp_column", "Fecha_Hora"))
    clean_path = prepared / "clean_cohort.csv"
    clean.to_csv(clean_path, index=False)
    cohort.to_csv(root / "selected_clients.csv", index=False)
    full_fault, fault_events, fault_summary, injector_selection = build_test_fault_template(
        clean, cohort, config
    )
    fault_template_dir = prepared / "fault_template"
    fault_template_dir.mkdir()
    full_fault.to_csv(fault_template_dir / "post_fault_full.csv", index=False)
    fault_events.to_csv(fault_template_dir / "fault_events.csv", index=False)
    fault_summary.to_csv(fault_template_dir / "fault_injection_summary.csv", index=False)
    injector_selection.to_csv(fault_template_dir / "injector_selected_clients.csv", index=False)

    manifests: dict[str, dict[str, object]] = {}
    for condition_id, factors in CONDITIONS.items():
        print(f"[diagnostic_2x2] preparing condition {condition_id}", flush=True)
        condition_dir = prepared / condition_id
        condition_dir.mkdir()
        post_fault = full_fault.copy() if factors["fault_injection"] else clean.copy()
        post_drift, drift_schedule = materialize_drift(
            post_fault,
            cohort,
            config,
            condition_id=condition_id,
            enabled=factors["true_drift"],
        )
        condition_input_path = condition_dir / "condition_input.csv"
        post_drift.to_csv(condition_input_path, index=False)
        drift_schedule_path = condition_dir / "drift_schedule.csv"
        drift_schedule.to_csv(drift_schedule_path, index=False)

        lineage = build_value_lineage(
            clean,
            post_fault,
            post_drift,
            cohort,
            config,
            condition_id=condition_id,
        )
        lineage.to_csv(condition_dir / "value_lineage.csv", index=False)

        fault_labels = build_fault_window_labels(
            cohort,
            fault_events,
            config,
            condition_id=condition_id,
            enabled=factors["fault_injection"],
        )
        fault_labels_path = condition_dir / "fault_window_labels.csv"
        fault_labels.to_csv(fault_labels_path, index=False)

        cluster_input_dir = condition_dir / "cluster_input"
        preprocess_clusters(
            condition_input_path,
            str(config["cluster_assignment_csv"]),
            cluster_input_dir,
            metadata_path=str(config["metadata_csv"]),
            timestamp_col=timestamp_col,
        )

        dq_path = condition_dir / "dq_scores.csv"
        build_test_dq_artifact(
            clean_path=clean_path,
            current_path=condition_input_path,
            output_path=dq_path,
            score_config_path=str(config["dq_score_config"]),
            clients=cohort["client_id"].astype(str).tolist(),
            config=config,
            condition_id=condition_id,
            progress_every=dq_progress_every,
            workers=int(dict(config.get("runner", {}) or {}).get("dq_workers", 1)),
        )
        manifest = build_condition_manifest(
            config,
            cohort,
            condition_id=condition_id,
            smoke=smoke,
            condition_input_path=condition_input_path,
            cluster_input_dir=cluster_input_dir,
            dq_path=dq_path,
            fault_labels_path=fault_labels_path,
            drift_schedule_path=drift_schedule_path,
        )
        labels, events = build_oracle_tables(manifest, clean[timestamp_col].tolist())
        labels_path = condition_dir / "oracle_labels.csv"
        events_path = condition_dir / "oracle_onset_events.csv"
        labels.to_csv(labels_path, index=False)
        events.to_csv(events_path, index=False)
        validate_oracle_tables(labels, events, manifest)
        manifest["oracle_labels_path"] = str(labels_path)
        manifest["oracle_labels_sha256"] = sha256_file(labels_path)
        manifest["oracle_onset_events_path"] = str(events_path)
        manifest["oracle_onset_events_sha256"] = sha256_file(events_path)
        write_json(condition_dir / "condition_manifest.json", manifest)
        manifests[condition_id] = manifest

    experiment_manifest = {
        "experiment_id": str(config["experiment_id"]),
        "execution_mode": "smoke_32_clients" if smoke else "primary_84_clients",
        "client_count": int(len(cohort)),
        "affected_client_count": int(cohort["affected_client"].sum()),
        "clean_control_count": int((~cohort["affected_client"]).sum()),
        "conditions": manifests,
        "clean_wide_csv": str(config["clean_wide_csv"]),
        "clean_wide_sha256": sha256_file(str(config["clean_wide_csv"])),
        "prepared_clean_cohort_path": str(clean_path),
        "prepared_clean_cohort_sha256": sha256_file(clean_path),
        "fault_config": str(config["fault_config"]),
        "fault_config_sha256": sha256_file(str(config["fault_config"])),
        "dq_score_config": str(config["dq_score_config"]),
        "dq_score_config_sha256": sha256_file(str(config["dq_score_config"])),
        "baseline_checkpoint_dir": str(config["baseline_checkpoint_dir"]),
        "oracle_role": "primary_ground_truth_offline",
        "page_hinkley_role": "primary_statistical_comparison_logging_only",
        "adwin_role": "secondary_diagnostic_logging_only",
        "condition_D_order": "clean -> F(test only) -> G(sudden) -> stream ffill/bfill -> clean-train normalization",
        "protected_runtime_files_modified_by_experiment": False,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    }
    write_json(root / "experiment_manifest.json", experiment_manifest)
    validation = validate_prepared_experiment(root)
    return {"experiment_manifest": experiment_manifest, "validation": validation}


def _load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_prepared_experiment(output_root: str | Path) -> dict[str, object]:
    root = Path(output_root).resolve()
    experiment = _load_manifest(root / "experiment_manifest.json")
    prepared = root / "prepared"
    cohort = pd.read_csv(root / "selected_clients.csv", dtype={"client_id": str, "cluster_id": str})
    clean = pd.read_csv(prepared / "clean_cohort.csv", low_memory=False)
    client_ids = cohort["client_id"].astype(str).tolist()
    affected_flags = _boolean_series(cohort["affected_client"])
    affected = set(cohort.loc[affected_flags, "client_id"].astype(str))
    controls = set(client_ids) - affected
    expected_client_count = 32 if experiment["execution_mode"] == "smoke_32_clients" else 84
    if len(client_ids) != expected_client_count:
        raise AssertionError(f"expected {expected_client_count} clients, got {len(client_ids)}")
    if len(affected) != 32:
        raise AssertionError("affected cohort must contain exactly 32 clients")
    if experiment["execution_mode"] == "primary_84_clients" and len(controls) != 52:
        raise AssertionError("primary experiment must contain exactly 52 clean controls")

    frames = {
        condition_id: pd.read_csv(prepared / condition_id / "condition_input.csv", low_memory=False)
        for condition_id in CONDITIONS
    }
    first_obs = _coordinate(experiment["config"], "test_input_observation_first")
    last_obs = _coordinate(experiment["config"], "test_input_observation_last")
    onset = _coordinate(experiment["config"], "drift_onset_index")
    checks: dict[str, object] = {}
    for condition_id, frame in frames.items():
        if len(frame) != len(clean):
            raise AssertionError(f"condition {condition_id} row count differs from clean")
        for client_id in controls:
            if _different(
                pd.to_numeric(clean[client_id], errors="coerce").to_numpy(float),
                pd.to_numeric(frame[client_id], errors="coerce").to_numpy(float),
            ).any():
                raise AssertionError(f"control client {client_id} changed in condition {condition_id}")

    for client_id in affected:
        clean_values = pd.to_numeric(clean[client_id], errors="coerce").to_numpy(float)
        a_values = pd.to_numeric(frames["A"][client_id], errors="coerce").to_numpy(float)
        b_values = pd.to_numeric(frames["B"][client_id], errors="coerce").to_numpy(float)
        c_values = pd.to_numeric(frames["C"][client_id], errors="coerce").to_numpy(float)
        d_values = pd.to_numeric(frames["D"][client_id], errors="coerce").to_numpy(float)
        if _different(clean_values, a_values).any():
            raise AssertionError("condition A differs from clean")
        if _different(clean_values[:onset], b_values[:onset]).any():
            raise AssertionError(f"condition B changed before onset for client {client_id}")
        if _different(c_values[:onset], d_values[:onset]).any():
            raise AssertionError(f"condition D differs from C before onset for client {client_id}")
        finite_b = np.isfinite(clean_values[onset:])
        if not np.allclose(
            b_values[onset:][finite_b],
            clean_values[onset:][finite_b] * 1.5,
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(f"condition B drift transform mismatch for client {client_id}")
        finite_d = np.isfinite(c_values[onset:])
        if not np.allclose(
            d_values[onset:][finite_d],
            c_values[onset:][finite_d] * 1.5,
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(f"condition D drift transform mismatch for client {client_id}")
        if not np.array_equal(np.isfinite(c_values), np.isfinite(d_values)):
            raise AssertionError(f"condition D did not preserve condition C missing mask for client {client_id}")

    fault_c = pd.read_csv(prepared / "C" / "fault_window_labels.csv", dtype={"client_id": str})
    fault_d = pd.read_csv(prepared / "D" / "fault_window_labels.csv", dtype={"client_id": str})
    fault_compare = [
        "client_id",
        "window_start_index",
        "window_end_index",
        "fault_at_endpoint",
        "fault_in_window",
        "fault_seen_in_history",
        "fault_observation_count",
        "fault_dimensions",
        "fault_types",
        "fault_subchecks",
    ]
    if canonical_rows_sha256(fault_c, fault_compare) != canonical_rows_sha256(fault_d, fault_compare):
        raise AssertionError("conditions C and D do not have identical fault masks")

    drift_b = pd.read_csv(prepared / "B" / "drift_schedule.csv", dtype={"client_id": str})
    drift_d = pd.read_csv(prepared / "D" / "drift_schedule.csv", dtype={"client_id": str})
    drift_compare = [
        "client_id",
        "observation_index",
        "drift_weight",
        "regime_active",
        "configured_start_index",
        "magnitude",
        "offset_delta",
        "noise_sigma",
        "random_seed",
    ]
    if canonical_rows_sha256(drift_b, drift_compare) != canonical_rows_sha256(drift_d, drift_compare):
        raise AssertionError("conditions B and D do not have identical drift schedules")

    expected_windows = _coordinate(experiment["config"], "test_window_start_last") - _coordinate(
        experiment["config"], "test_window_start_first"
    ) + 1
    expected_rows = len(client_ids) * expected_windows
    condition_checks: dict[str, object] = {}
    for condition_id in CONDITIONS:
        manifest = _load_manifest(prepared / condition_id / "condition_manifest.json")
        dq_path = Path(str(manifest["dq_scores_path"]))
        dq = pd.read_csv(dq_path, dtype={"client_id": str})
        if len(dq) != expected_rows or dq.duplicated(
            ["condition_id", "client_id", "mode", "window_start_index", "window_end_index"]
        ).any():
            raise AssertionError(f"condition {condition_id} DQ coverage is not exact")
        input_hash = sha256_file(manifest["condition_input_path"])
        if set(dq["condition_input_sha256"]) != {input_hash}:
            raise AssertionError(f"condition {condition_id} DQ provenance does not match its input")
        if not np.isfinite(dq[DQ_COLUMNS].to_numpy(float)).all():
            raise AssertionError(f"condition {condition_id} DQ contains non-finite scores")
        if ((dq[DQ_COLUMNS] < 0) | (dq[DQ_COLUMNS] > 1)).any().any():
            raise AssertionError(f"condition {condition_id} DQ scores fall outside [0, 1]")
        labels = pd.read_csv(manifest["oracle_labels_path"], dtype={"client_id": str})
        events = pd.read_csv(manifest["oracle_onset_events_path"], dtype={"client_id": str})
        validate_oracle_tables(labels, events, manifest)
        condition_checks[condition_id] = {
            "input_sha256": input_hash,
            "dq_sha256": sha256_file(dq_path),
            "dq_rows": len(dq),
            "oracle_label_rows": len(labels),
            "oracle_event_rows": len(events),
        }
    checks.update(
        {
            "execution_mode": experiment["execution_mode"],
            "client_count": len(client_ids),
            "affected_client_count": len(affected),
            "clean_control_count": len(controls),
            "expected_windows_per_client": expected_windows,
            "expected_rows_per_condition": expected_rows,
            "fault_mask_C_sha256": canonical_rows_sha256(fault_c, fault_compare),
            "fault_mask_D_sha256": canonical_rows_sha256(fault_d, fault_compare),
            "drift_schedule_B_sha256": canonical_rows_sha256(drift_b, drift_compare),
            "drift_schedule_D_sha256": canonical_rows_sha256(drift_d, drift_compare),
            "condition_checks": condition_checks,
            "test_input_union": [first_obs, last_obs],
            "drift_onset_index": onset,
            "prepared_validation_passed": True,
        }
    )
    write_json(root / "prepared_validation.json", checks)
    return checks


def stage_required_checkpoints(
    baseline_dir: str | Path,
    run_dir: str | Path,
    cohort: pd.DataFrame,
) -> dict[str, str]:
    """Stage only final client/server checkpoints required by final_shared evaluation."""
    baseline = Path(baseline_dir)
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=False)
    staged: dict[str, str] = {}
    for cluster_id in sorted(cohort["cluster_id"].astype(str).unique()):
        cluster_target = destination / f"cluster_{cluster_id}"
        cluster_target.mkdir()
        server_source = baseline / f"cluster_{cluster_id}" / "server_final.pt"
        if not server_source.exists():
            raise FileNotFoundError(server_source)
        server_target = cluster_target / "server_final.pt"
        shutil.copy2(server_source, server_target)
        staged[str(server_target.relative_to(destination))] = sha256_file(server_target)
        clients = cohort.loc[cohort["cluster_id"].astype(str) == cluster_id, "client_id"].astype(str)
        for client_id in clients:
            source = baseline / f"cluster_{cluster_id}" / f"client_{client_id}_final.pt"
            if not source.exists():
                raise FileNotFoundError(source)
            target = cluster_target / source.name
            shutil.copy2(source, target)
            staged[str(target.relative_to(destination))] = sha256_file(target)
    return staged


def build_run_environment(
    config: Mapping[str, object],
    condition_manifest: Mapping[str, object],
    run_dir: str | Path,
) -> dict[str, str]:
    detector = dict(config["detectors"])
    runner = dict(config.get("runner", {}) or {})
    client_ids = [str(row["client_id"]) for row in condition_manifest["clients"]]
    cluster_ids = sorted(
        {str(row["cluster_id"]) for row in condition_manifest["clients"]},
        key=lambda value: int(value) if value.isdigit() else value,
    )
    false_flags = {
        "USE_QUALITY_WEIGHTS": "false",
        "USE_QUALITY_AS_INPUT": "false",
        "USE_QUALITY_AGGREGATION_WEIGHTS": "false",
        "USE_QUALITY_ANCHOR_AGGREGATION": "false",
        "USE_GRADIENT_ALIGNMENT_FILTER": "false",
        "USE_HIGH_QUALITY_VALIDATION_GUARDRAIL": "false",
    }
    env = {
        "OUTPUT_DIR": str(Path(run_dir).resolve()),
        "CLUSTER_IDS": ",".join(cluster_ids),
        "CLIENT_IDS": ",".join(client_ids),
        "CLUSTER_DATA_DIR": str(Path(str(config["clean_cluster_dir"])).resolve()),
        "TRAIN_CLUSTER_DATA_DIR": str(Path(str(config["clean_cluster_dir"])).resolve()),
        "VAL_CLUSTER_DATA_DIR": str(Path(str(config["clean_cluster_dir"])).resolve()),
        "TEST_CLUSTER_DATA_DIR": str(Path(str(condition_manifest["cluster_input_dir"])).resolve()),
        "TRAIN_TARGET_CLUSTER_DATA_DIR": str(Path(str(config["clean_cluster_dir"])).resolve()),
        "VAL_TARGET_CLUSTER_DATA_DIR": str(Path(str(config["clean_cluster_dir"])).resolve()),
        "TEST_TARGET_CLUSTER_DATA_DIR": str(Path(str(config["clean_cluster_dir"])).resolve()),
        "WINDOW_SIZE": str(_coordinate(config, "window_size")),
        "FORECAST_HORIZON": str(_coordinate(config, "forecast_horizon")),
        "TRAIN_RATIO": str(dict(config["split"])["train"]),
        "VAL_RATIO": str(dict(config["split"])["val"]),
        "TEST_RATIO": str(dict(config["split"])["test"]),
        "BATCH_SIZE": str(runner.get("batch_size", 128)),
        "SKIP_TRAINING": "true",
        "FROZEN_VALIDATION_MODE": "true",
        "EVAL_MODE": "final_shared",
        "NORMALIZE": "true",
        "NORMALIZE_METHOD": "zscore",
        "USE_STATIC_FEATURES": "false",
        "AUTO_MODEL_INPUT_SIZE": "true",
        "DRIFT_INJECTION_ENABLED": "false",
        "DRIFT_DETECTION_ENABLED": "true",
        "DRIFT_DETECTORS": "page_hinkley,adwin",
        "DRIFT_TYPE": "none",
        "DRIFT_SOURCE": "input",
        "DRIFT_SCOPE": "test",
        "DRIFT_RUN_ID": f"{config['experiment_id']}_{condition_manifest['condition_id']}",
        "DRIFT_CONFIGURATION_ID": f"{config['experiment_id']}_{condition_manifest['condition_id']}",
        "DRIFT_RANDOM_SEED": str(dict(config["drift"])["seed"]),
        "QUALITY_SCORE_PATH": "",
        "PH_MIN_INSTANCES": str(detector["PH_MIN_INSTANCES"]),
        "PH_DELTA": str(detector["PH_DELTA"]),
        "PH_THRESHOLD": str(detector["PH_THRESHOLD"]),
        "PH_ALPHA": str(detector["PH_ALPHA"]),
        "ADWIN_DELTA": str(detector["ADWIN_DELTA"]),
        "ADWIN_CLOCK": str(detector["ADWIN_CLOCK"]),
        "ADWIN_MAX_BUCKETS": str(detector["ADWIN_MAX_BUCKETS"]),
        "ADWIN_MIN_WINDOW_LENGTH": str(detector["ADWIN_MIN_WINDOW_LENGTH"]),
        "ADWIN_GRACE_PERIOD": str(detector["ADWIN_GRACE_PERIOD"]),
        "RUN_CLUSTERS_IN_PARALLEL": str(runner.get("run_clusters_in_parallel", False)).lower(),
        "GPU_ENABLED": str(runner.get("gpu_enabled", False)).lower(),
        "STREAM_BASE_PORT": str(runner.get("stream_base_port", 5101)),
        "SERVER_BASE_PORT": str(runner.get("server_base_port", 5002)),
        "CLIENT_WORKER_BASE_PORT": str(runner.get("client_worker_base_port", 6101)),
    }
    env.update(false_flags)
    return env


def _prediction_frame(run_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(run_dir.glob("cluster_*/client_*_test_predictions.csv")):
        cluster_id = path.parent.name.replace("cluster_", "", 1)
        client_id = path.name.replace("client_", "", 1).replace("_test_predictions.csv", "")
        frame = pd.read_csv(path)
        frame["cluster_id"] = str(cluster_id)
        frame["client_id"] = str(client_id)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no prediction artifacts under {run_dir}")
    return pd.concat(frames, ignore_index=True)


def _detector_frame(run_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(run_dir.glob("cluster_*/drift/detector_events.csv")):
        frame = pd.read_csv(path, dtype={"client_id": str, "cluster_id": str})
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no PH/ADWIN event artifacts under {run_dir}")
    result = pd.concat(frames, ignore_index=True)
    result["alarm"] = result["alarm"].astype(str).str.lower().isin({"1", "true", "yes"})
    return result


def _segment(start: int, end: int, onset: int) -> str:
    if end < onset:
        return "pre_drift"
    if start < onset <= end:
        return "mixed_window"
    return "fully_drifted"


def join_condition_artifacts(output_root: str | Path, condition_id: str) -> pd.DataFrame:
    root = Path(output_root).resolve()
    prepared = root / "prepared" / condition_id
    run_dir = root / "runs" / condition_id
    manifest = _load_manifest(prepared / "condition_manifest.json")
    predictions = _prediction_frame(run_dir)
    predictions["client_id"] = predictions["client_id"].astype(str)
    predictions["cluster_id"] = predictions["cluster_id"].astype(str)
    predictions["condition_id"] = condition_id
    dq = pd.read_csv(manifest["dq_scores_path"], dtype={"client_id": str})
    oracle = pd.read_csv(manifest["oracle_labels_path"], dtype={"client_id": str, "cluster_id": str})
    faults = pd.read_csv(manifest["fault_window_labels_path"], dtype={"client_id": str, "cluster_id": str})
    keys = ["condition_id", "cluster_id", "client_id", "window_start_index", "window_end_index"]
    for frame in (predictions, dq, oracle, faults):
        frame["condition_id"] = condition_id
        frame["client_id"] = frame["client_id"].astype(str)
        if "cluster_id" in frame:
            frame["cluster_id"] = frame["cluster_id"].astype(str)
    dq = dq.merge(
        pd.DataFrame(manifest["clients"])[["client_id", "cluster_id"]].astype(str),
        on="client_id",
        how="left",
        validate="many_to_one",
    )
    merged = predictions.merge(dq[keys + DQ_COLUMNS], on=keys, how="inner", validate="one_to_one")
    merged = merged.merge(
        oracle[
            keys
            + [
                "detector_endpoint_index",
                "endpoint_timestamp",
                "affected_client",
                "true_drift_active",
                "oracle_alarm",
                "drift_onset_index",
                "oracle_source",
                "controls_runtime",
            ]
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        faults[
            keys
            + [
                "fault_at_endpoint",
                "fault_in_window",
                "fault_seen_in_history",
                "fault_observation_count",
                "fault_observation_fraction",
                "fault_dimensions",
                "fault_types",
                "fault_subchecks",
            ]
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    detector = _detector_frame(run_dir)
    detector["condition_id"] = condition_id
    detector["window_end_index"] = detector["observation_index"].astype(int)
    detector["client_id"] = detector["client_id"].astype(str)
    detector["cluster_id"] = detector["cluster_id"].astype(str)
    event_keys = ["condition_id", "cluster_id", "client_id", "window_end_index"]
    if detector.duplicated(event_keys + ["detector_name"]).any():
        raise AssertionError(f"condition {condition_id} contains duplicate detector endpoint updates")
    alarm_wide = detector.pivot(index=event_keys, columns="detector_name", values="alarm").reset_index()
    value_wide = detector.pivot(
        index=event_keys, columns="detector_name", values="detector_value"
    ).reset_index()
    alarm_wide = alarm_wide.rename(
        columns={"page_hinkley": "page_hinkley_alarm", "adwin": "adwin_alarm"}
    )
    value_wide = value_wide.rename(
        columns={"page_hinkley": "page_hinkley_value", "adwin": "adwin_value"}
    )
    merged = merged.merge(alarm_wide, on=event_keys, how="inner", validate="one_to_one")
    merged = merged.merge(value_wide, on=event_keys, how="inner", validate="one_to_one")
    onset = int(manifest["drift_onset_index"])
    merged["analysis_segment"] = [
        _segment(int(start), int(end), onset)
        for start, end in zip(merged["window_start_index"], merged["window_end_index"])
    ]
    merged["oracle_controls_runtime"] = _boolean_series(merged["controls_runtime"])
    merged["page_hinkley_role"] = "primary_statistical_comparison_logging_only"
    merged["adwin_role"] = "secondary_diagnostic_logging_only"
    merged["detectors_control_runtime"] = False
    expected = int(manifest["client_count"]) * (
        int(manifest["test_window_start_last"]) - int(manifest["test_window_start_first"]) + 1
    )
    if len(merged) != expected:
        raise AssertionError(
            f"condition {condition_id} offline join expected {expected} rows, got {len(merged)}"
        )
    return merged.sort_values(
        ["cluster_id", "client_id", "window_start_index"], kind="mergesort"
    ).reset_index(drop=True)


def _forecast_summary(group: pd.DataFrame) -> dict[str, float | None]:
    prediction = group["y_pred"].to_numpy(float)
    target = group["y_true"].to_numpy(float)
    error = prediction - target
    mse = float(np.mean(error**2))
    denom = np.abs(prediction) + np.abs(target)
    valid = denom > 0
    ss_total = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "smape": float(np.mean(2.0 * np.abs(error[valid]) / denom[valid]) * 100.0)
        if valid.any()
        else None,
        "r2": None if ss_total <= 0 else float(1.0 - np.sum(error**2) / ss_total),
    }


def build_client_metric_summary(windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouping = ["condition_id", "cluster_id", "client_id", "affected_client"]
    for identity, client_rows in windows.groupby(grouping, sort=True, dropna=False):
        condition_id, cluster_id, client_id, affected_client = identity
        segments = {"overall": client_rows}
        segments.update(
            {
                segment: segment_rows
                for segment, segment_rows in client_rows.groupby("analysis_segment", sort=True)
            }
        )
        for segment, segment_rows in segments.items():
            for column in DQ_COLUMNS:
                rows.append(
                    {
                        "condition_id": condition_id,
                        "cluster_id": cluster_id,
                        "client_id": client_id,
                        "affected_client": bool(affected_client),
                        "segment": segment,
                        "metric_family": "data_quality",
                        "metric": column,
                        "value": float(segment_rows[column].mean()),
                        "window_count": len(segment_rows),
                    }
                )
            for metric, value in _forecast_summary(segment_rows).items():
                rows.append(
                    {
                        "condition_id": condition_id,
                        "cluster_id": cluster_id,
                        "client_id": client_id,
                        "affected_client": bool(affected_client),
                        "segment": segment,
                        "metric_family": "forecast",
                        "metric": metric,
                        "value": value,
                        "window_count": len(segment_rows),
                    }
                )
    return pd.DataFrame(rows)


def exact_sign_test_p(values: Sequence[float]) -> float | None:
    nonzero = [float(value) for value in values if math.isfinite(float(value)) and float(value) != 0.0]
    n = len(nonzero)
    if n == 0:
        return None
    positives = sum(value > 0 for value in nonzero)
    tail = min(positives, n - positives)
    probability = sum(math.comb(n, index) for index in range(0, tail + 1)) / (2**n)
    return min(1.0, 2.0 * probability)


def calculate_factorial_effects(client_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    effects: list[dict[str, object]] = []
    client_effects: list[dict[str, object]] = []
    for population, selected in (
        ("affected_32", client_metrics.loc[client_metrics["affected_client"].astype(bool)]),
        ("clean_controls", client_metrics.loc[~client_metrics["affected_client"].astype(bool)]),
        ("all_clients", client_metrics),
    ):
        for (family, metric, segment), rows in selected.groupby(
            ["metric_family", "metric", "segment"], sort=True
        ):
            pivot = rows.pivot(index=["cluster_id", "client_id"], columns="condition_id", values="value")
            for contrast_name, weights in CONTRASTS.items():
                required = list(weights)
                complete = pivot.dropna(subset=required)
                if complete.empty:
                    continue
                delta = sum(float(weight) * complete[condition] for condition, weight in weights.items())
                for (cluster_id, client_id), value in delta.items():
                    client_effects.append(
                        {
                            "population": population,
                            "metric_family": family,
                            "metric": metric,
                            "segment": segment,
                            "contrast": contrast_name,
                            "cluster_id": cluster_id,
                            "client_id": client_id,
                            "effect": float(value),
                        }
                    )
                values = delta.to_numpy(float)
                effects.append(
                    {
                        "population": population,
                        "metric_family": family,
                        "metric": metric,
                        "segment": segment,
                        "contrast": contrast_name,
                        "client_count": len(values),
                        "mean_effect": float(np.mean(values)),
                        "median_effect": float(np.median(values)),
                        "std_effect": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "min_effect": float(np.min(values)),
                        "max_effect": float(np.max(values)),
                        "positive_clients": int((values > 0).sum()),
                        "negative_clients": int((values < 0).sum()),
                        "zero_clients": int((values == 0).sum()),
                        "exact_sign_test_p": exact_sign_test_p(values),
                        "statistical_unit": "paired_client_summary",
                        "window_independence_assumed": False,
                    }
                )
    return pd.DataFrame(effects), pd.DataFrame(client_effects)


def build_detector_summaries(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    client_rows: list[dict[str, object]] = []
    detector_specs = (
        ("page_hinkley", "primary_statistical_comparison_logging_only"),
        ("adwin", "secondary_diagnostic_logging_only"),
    )
    for detector, role in detector_specs:
        alarm_col = f"{detector}_alarm"
        for identity, rows in windows.groupby(
            ["condition_id", "cluster_id", "client_id", "affected_client"], sort=True
        ):
            condition_id, cluster_id, client_id, affected_client = identity
            alarms = rows.loc[rows[alarm_col].astype(bool), "window_end_index"].astype(int).tolist()
            onset_values = rows["drift_onset_index"].dropna()
            onset = int(onset_values.iloc[0]) if not onset_values.empty else None
            true_drift_client = bool(rows["true_drift_active"].any())
            pre = [alarm for alarm in alarms if onset is not None and alarm < onset]
            post = [alarm for alarm in alarms if onset is not None and alarm >= onset]
            first_post = post[0] if post else None
            client_rows.append(
                {
                    "condition_id": condition_id,
                    "cluster_id": cluster_id,
                    "client_id": client_id,
                    "affected_client": bool(affected_client),
                    "true_drift_client": true_drift_client,
                    "detector_name": detector,
                    "detector_role": role,
                    "observations_seen": len(rows),
                    "alarm_count": len(alarms),
                    "any_alarm": bool(alarms),
                    "pre_onset_alarm_count": len(pre),
                    "post_onset_alarm_count": len(post),
                    "first_alarm": alarms[0] if alarms else None,
                    "first_post_onset_alarm": first_post,
                    "detection_delay": first_post - onset if first_post is not None and onset is not None else None,
                    "controls_runtime": False,
                }
            )
    client = pd.DataFrame(client_rows)
    summaries: list[dict[str, object]] = []
    for identity, rows in client.groupby(["condition_id", "detector_name", "detector_role"], sort=True):
        condition_id, detector_name, role = identity
        for population, population_rows in (
            ("affected_32", rows.loc[rows["affected_client"]]),
            ("clean_controls", rows.loc[~rows["affected_client"]]),
            ("all_clients", rows),
        ):
            if population_rows.empty:
                continue
            drifted = population_rows.loc[population_rows["true_drift_client"]]
            clean = population_rows.loc[~population_rows["true_drift_client"]]
            delays = pd.to_numeric(drifted["detection_delay"], errors="coerce").dropna()
            summaries.append(
                {
                    "condition_id": condition_id,
                    "detector_name": detector_name,
                    "detector_role": role,
                    "population": population,
                    "client_count": len(population_rows),
                    "event_count": int(population_rows["observations_seen"].sum()),
                    "alarm_count": int(population_rows["alarm_count"].sum()),
                    "clients_with_alarm_rate": float(population_rows["any_alarm"].mean()),
                    "clean_false_positive_rate": float(clean["any_alarm"].mean()) if not clean.empty else None,
                    "post_onset_detection_rate": float((drifted["post_onset_alarm_count"] > 0).mean())
                    if not drifted.empty
                    else None,
                    "mean_detection_delay": float(delays.mean()) if not delays.empty else None,
                    "controls_runtime": False,
                }
            )
    return client, pd.DataFrame(summaries)


def aggregate_checkpoint_proofs(output_root: str | Path) -> dict[str, object]:
    root = Path(output_root).resolve()
    conditions: dict[str, object] = {}
    reference_before: dict[str, str] | None = None
    all_passed = True
    identical_before = True
    for condition_id in CONDITIONS:
        path = root / "runs" / condition_id / "results" / "frozen_validation_report.json"
        if not path.exists():
            raise FileNotFoundError(path)
        report = _load_manifest(path)
        before = dict(report["checkpoint_hashes_before"])
        after = dict(report["checkpoint_hashes_after"])
        if reference_before is None:
            reference_before = before
        elif before != reference_before:
            identical_before = False
        passed = bool(report.get("frozen_validation_passed"))
        all_passed = all_passed and passed
        conditions[condition_id] = {
            "report_path": str(path),
            "checkpoint_hashes_before": before,
            "checkpoint_hashes_after": after,
            "changed_checkpoint_parameters": report.get("changed_checkpoint_parameters", []),
            "parameters_unchanged": bool(report.get("parameters_unchanged")),
            "round_start_events": int(report.get("round_start_events", -1)),
            "fedavg_events": int(report.get("fedavg_events", -1)),
            "frozen_validation_passed": passed,
        }
    payload = {
        "conditions": conditions,
        "all_conditions_frozen_validation_passed": all_passed,
        "identical_staged_checkpoint_hash_maps_across_conditions": identical_before,
    }
    write_json(root / "results" / "checkpoint_hashes.json", payload)
    return payload


def evaluate_experiment(output_root: str | Path) -> dict[str, object]:
    root = Path(output_root).resolve()
    validate_prepared_experiment(root)
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    condition_windows = [join_condition_artifacts(root, condition_id) for condition_id in CONDITIONS]
    windows = pd.concat(condition_windows, ignore_index=True)
    windows.to_csv(results_dir / "window_diagnostics.csv", index=False)
    client_metrics = build_client_metric_summary(windows)
    client_metrics.to_csv(results_dir / "client_metric_summary.csv", index=False)
    effects, client_effects = calculate_factorial_effects(client_metrics)
    effects.to_csv(results_dir / "factorial_effects.csv", index=False)
    client_effects.to_csv(results_dir / "client_factorial_effects.csv", index=False)
    detector_clients, detector_summary = build_detector_summaries(windows)
    detector_clients.to_csv(results_dir / "detector_client_summary.csv", index=False)
    detector_summary.to_csv(results_dir / "detector_summary.csv", index=False)
    checkpoint_proofs = aggregate_checkpoint_proofs(root)

    experiment = _load_manifest(root / "experiment_manifest.json")
    expected_clients = int(experiment["client_count"])
    expected_windows = (
        _coordinate(experiment["config"], "test_window_start_last")
        - _coordinate(experiment["config"], "test_window_start_first")
        + 1
    )
    expected_rows = expected_clients * expected_windows
    condition_counts = windows.groupby("condition_id").size().to_dict()
    detector_expected = expected_rows * 2
    detector_counts = (
        detector_clients.groupby("condition_id")["observations_seen"].sum().astype(int).to_dict()
    )
    acceptance = {
        "execution_mode": experiment["execution_mode"],
        "client_count": expected_clients,
        "affected_client_count": int(experiment["affected_client_count"]),
        "clean_control_count": int(experiment["clean_control_count"]),
        "expected_window_rows_per_condition": expected_rows,
        "window_rows_by_condition": {str(key): int(value) for key, value in condition_counts.items()},
        "expected_detector_updates_per_condition": detector_expected,
        "detector_updates_by_condition": {str(key): int(value) for key, value in detector_counts.items()},
        "oracle_execution": "offline_only",
        "oracle_controls_runtime": False,
        "dq_execution": "offline_only",
        "dq_controls_runtime": False,
        "ph_role": "primary_statistical_comparison_logging_only",
        "adwin_role": "secondary_diagnostic_logging_only",
        "detectors_control_runtime": False,
        "runtime_drift_injection_enabled": False,
        "model_affecting_dq_flags_all_false": True,
        "factorial_contrasts": list(CONTRASTS),
        "factorial_statistical_unit": "paired_client_summary",
        "checkpoint_proofs": checkpoint_proofs,
        "acceptance_passed": (
            all(int(condition_counts.get(condition, 0)) == expected_rows for condition in CONDITIONS)
            and all(int(detector_counts.get(condition, 0)) == detector_expected for condition in CONDITIONS)
            and bool(checkpoint_proofs["all_conditions_frozen_validation_passed"])
            and bool(checkpoint_proofs["identical_staged_checkpoint_hash_maps_across_conditions"])
        ),
    }
    write_json(results_dir / "acceptance_report.json", acceptance)
    return acceptance
