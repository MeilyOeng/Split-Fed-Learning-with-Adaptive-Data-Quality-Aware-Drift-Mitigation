"""Demand-specific labeled dataset generation for surrogate-model training."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .baseline import load_baseline_artifact
from .features import extract_client_window_features
from .io import load_yaml_config, save_json
from .pca_quality import DemandPCAQualityModel, QUALITY_DIMENSIONS
from .scoring import WideDemandQualityScorer
from .windowing import iter_client_windows, load_window_source


TARGET_COLUMNS = [
    "y_completeness",
    "y_validity",
    "y_consistency",
    "y_accuracy",
    "y_freshness",
    "y_final",
]


CORE_WINDOW_COLUMNS = [
    "client_id",
    "mode",
    "window_id",
    "window_index",
    "window_start_index",
    "window_end_index",
    "window_start",
    "window_end",
]


DIMENSION_OUTPUT_COLUMNS = [
    "quality_score",
    "dimension_completeness",
    "dimension_validity",
    "dimension_consistency",
    "dimension_accuracy",
    "dimension_freshness",
]


def _runtime_split_bounds(
    num_values: int,
    *,
    window_size_steps: int,
    forecast_horizon: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, tuple[int, int]]:
    total_windows = max(0, num_values - window_size_steps - forecast_horizon + 1)
    train_windows = int(total_windows * train_ratio)
    val_windows = int(total_windows * val_ratio)
    test_windows = total_windows - train_windows - val_windows
    train_end = train_windows
    val_end = train_end + val_windows
    test_end = val_end + test_windows
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, test_end),
    }


def _runtime_mode_for_window_start(window_start_index: int, split_bounds: dict[str, tuple[int, int]]) -> str:
    for mode in ("train", "val", "test"):
        start, end = split_bounds[mode]
        if start <= window_start_index < end:
            return mode
    return "out_of_split"


def _save_dataframe(df: pd.DataFrame, path: Path | str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
        return
    df.to_csv(out_path, index=False)


def _load_pca_model(path: Path | str) -> DemandPCAQualityModel:
    model_path = Path(path)
    with model_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return DemandPCAQualityModel.from_dict(payload)


def _ordered_output_columns(columns: Sequence[str]) -> list[str]:
    priority = [*CORE_WINDOW_COLUMNS, *DIMENSION_OUTPUT_COLUMNS]
    excluded = set(priority) | set(TARGET_COLUMNS)
    middle = [col for col in columns if col not in excluded]
    return [
        *[col for col in priority if col in columns],
        *middle,
        *[col for col in TARGET_COLUMNS if col in columns],
    ]


def _normalize_quality_score(
    raw_scores: np.ndarray,
    dimension_df: pd.DataFrame,
    model: DemandPCAQualityModel,
    epsilon: float = 1e-9,
) -> np.ndarray:
    if model.score_max_ - model.score_min_ <= epsilon:
        return np.clip(dimension_df.loc[:, QUALITY_DIMENSIONS].mean(axis=1).to_numpy(dtype=float), 0.0, 1.0)
    normalized = (raw_scores - model.score_min_) / (model.score_max_ - model.score_min_)
    return np.clip(normalized, 0.0, 1.0)


def _offline_freshness_scores(window, config: Optional[dict]) -> dict[str, float]:
    config = config or {}
    freshness_cfg = dict((config.get("freshness", {}) or {}))
    staleness_tau = pd.to_timedelta(freshness_cfg.get("staleness_tau", "48h"))
    recency_window = pd.to_timedelta(freshness_cfg.get("window_recency_span", "168h"))
    reference_time = pd.Timestamp(window.window_end)

    history_ts = pd.to_datetime(window.current_raw_history[window.timestamp_col], errors="coerce")
    history_values = pd.to_numeric(window.current_raw_history.get(window.client_id), errors="coerce")
    valid_history = history_ts.notna() & history_values.notna() & np.isfinite(history_values)
    available_start = history_ts.loc[history_ts.notna()].min() if history_ts.notna().any() else pd.NaT

    if valid_history.any() and staleness_tau > pd.Timedelta(0):
        latest_valid_ts = history_ts.loc[valid_history].max()
        age = reference_time - latest_valid_ts
        f1 = float(max(0.0, 1.0 - (age / staleness_tau)))
    else:
        f1 = 0.0

    if recency_window > pd.Timedelta(0) and pd.notna(available_start):
        recent_start = max(reference_time - recency_window, pd.Timestamp(available_start))
        expected_recent_count = int(
            len(pd.date_range(start=recent_start, end=reference_time, freq=window.frequency))
        )
        recent_mask = valid_history & (history_ts >= recent_start) & (history_ts <= reference_time)
        recent_valid_count = int(history_ts.loc[recent_mask].nunique())
        f2 = float(recent_valid_count / max(1, expected_recent_count))
        f2 = float(min(1.0, max(0.0, f2)))
    else:
        f2 = 0.0

    freshness = float(0.5 * f1 + 0.5 * f2)
    return {
        "F1_offline": f1,
        "F2_offline": f2,
        "F3_runtime_placeholder": 1.0,
        "y_freshness": freshness,
    }


def _build_window_row(
    window,
    *,
    scorer: WideDemandQualityScorer,
    score_config: dict,
    baseline_artifact: Optional[dict],
    include_features: bool,
    split_bounds: dict[str, tuple[int, int]],
) -> dict[str, float | str]:
    current = window.current_window
    clean = window.clean_window
    client_id = window.client_id
    comp = scorer.completeness.score_client(current, clean, client_id)
    val = scorer.validity.score_client(current, clean, client_id)
    cons = scorer.consistency.score_client(current, clean, client_id)
    acc = scorer.accuracy.score_client(current, clean, client_id)
    freshness = _offline_freshness_scores(window, score_config)

    row: dict[str, float | str] = {
        "client_id": client_id,
        "mode": _runtime_mode_for_window_start(window.window_start_index, split_bounds),
        "window_id": window.window_id,
        "window_index": int(window.window_index),
        "window_start_index": int(window.window_start_index),
        "window_end_index": int(window.window_end_index),
        "window_start": window.window_start.isoformat(),
        "window_end": window.window_end.isoformat(),
        "y_completeness": float(comp["completeness"]),
        "y_validity": float(val["validity"]),
        "y_consistency": float(cons["consistency"]),
        "y_accuracy": float(acc["accuracy"]),
        "y_freshness": float(freshness["y_freshness"]),
        "offline_F1": float(freshness["F1_offline"]),
        "offline_F2": float(freshness["F2_offline"]),
    }
    if include_features:
        row.update(
            extract_client_window_features(
                window,
                config=score_config,
                baseline_artifact=baseline_artifact,
            )
        )
    return row


def _write_csv_chunk(
    rows: list[dict[str, float | str]],
    *,
    path: Path,
    columns: Optional[list[str]],
    write_header: bool,
) -> list[str]:
    chunk_df = pd.DataFrame(rows)
    if chunk_df.empty:
        return columns or []
    if columns is None:
        columns = list(chunk_df.columns)
    chunk_df = chunk_df.reindex(columns=columns)
    chunk_df.to_csv(path, mode="w" if write_header else "a", header=write_header, index=False)
    return columns


def _prepare_pca_model(
    base_path: Path,
    *,
    pca_mode: str,
    pca_model_path: Optional[Path | str],
    pca_model_output_path: Optional[Path | str],
) -> DemandPCAQualityModel:
    if pca_mode == "fit":
        dims = pd.read_csv(
            base_path,
            usecols=["y_completeness", "y_validity", "y_consistency", "y_accuracy", "y_freshness"],
        )
        dimension_df = pd.DataFrame(
            {
                "completeness": dims["y_completeness"].astype(float),
                "validity": dims["y_validity"].astype(float),
                "consistency": dims["y_consistency"].astype(float),
                "accuracy": dims["y_accuracy"].astype(float),
                "freshness": dims["y_freshness"].astype(float),
            }
        )
        model = DemandPCAQualityModel.fit(dimension_df)
        if pca_model_output_path is not None:
            save_json(model.to_dict(), pca_model_output_path)
        return model
    if pca_mode == "transform":
        if pca_model_path is None:
            raise ValueError("pca_model_path is required when pca_mode='transform'.")
        return _load_pca_model(pca_model_path)
    raise ValueError("pca_mode must be 'fit' or 'transform'.")


def _finalize_csv_output(
    *,
    base_path: Path,
    output_path: Path,
    pca_mode: str,
    pca_model_path: Optional[Path | str],
    pca_model_output_path: Optional[Path | str],
    chunk_size: int,
) -> None:
    model = _prepare_pca_model(
        base_path,
        pca_mode=pca_mode,
        pca_model_path=pca_model_path,
        pca_model_output_path=pca_model_output_path,
    )

    wrote_header = False
    for chunk in pd.read_csv(base_path, chunksize=max(1, int(chunk_size))):
        dimension_df = pd.DataFrame(
            {
                "completeness": chunk["y_completeness"].astype(float),
                "validity": chunk["y_validity"].astype(float),
                "consistency": chunk["y_consistency"].astype(float),
                "accuracy": chunk["y_accuracy"].astype(float),
                "freshness": chunk["y_freshness"].astype(float),
            }
        )
        raw_scores = model.transform(dimension_df)
        normalized = _normalize_quality_score(raw_scores, dimension_df, model)
        chunk["y_final"] = normalized
        chunk["quality_score"] = normalized
        chunk["dimension_completeness"] = chunk["y_completeness"].astype(float)
        chunk["dimension_validity"] = chunk["y_validity"].astype(float)
        chunk["dimension_consistency"] = chunk["y_consistency"].astype(float)
        chunk["dimension_accuracy"] = chunk["y_accuracy"].astype(float)
        chunk["dimension_freshness"] = chunk["y_freshness"].astype(float)
        chunk = chunk.loc[:, _ordered_output_columns(chunk.columns.tolist())]
        chunk.to_csv(output_path, mode="w" if not wrote_header else "a", header=not wrote_header, index=False)
        wrote_header = True


def _build_csv_artifact(
    *,
    source,
    output_path: Path,
    scorer: WideDemandQualityScorer,
    score_config: dict,
    baseline_artifact: Optional[dict],
    include_features: bool,
    window_size_steps: int,
    stride_steps: int,
    forecast_horizon: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    max_windows: Optional[int],
    pca_mode: str,
    pca_model_path: Optional[Path | str],
    pca_model_output_path: Optional[Path | str],
    return_dataframe: bool,
    chunk_size: int,
    progress_every_windows: Optional[int],
) -> pd.DataFrame:
    total_runtime_windows = max(0, len(source.expected_grid) - window_size_steps - forecast_horizon + 1)
    split_bounds = _runtime_split_bounds(
        len(source.expected_grid),
        window_size_steps=window_size_steps,
        forecast_horizon=forecast_horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    generated = 0

    with tempfile.TemporaryDirectory(dir=output_path.parent) as tmpdir:
        base_path = Path(tmpdir) / "window_rows_base.csv"
        buffer: list[dict[str, float | str]] = []
        columns: Optional[list[str]] = None
        write_header = True

        for window in iter_client_windows(
            source,
            window_size_steps=window_size_steps,
            stride_steps=stride_steps,
        ):
            if window.window_start_index >= total_runtime_windows:
                continue
            if max_windows is not None and generated >= int(max_windows):
                break

            buffer.append(
                _build_window_row(
                    window,
                    scorer=scorer,
                    score_config=score_config,
                    baseline_artifact=baseline_artifact,
                    include_features=include_features,
                    split_bounds=split_bounds,
                )
            )
            generated += 1

            if progress_every_windows and generated % int(progress_every_windows) == 0:
                elapsed = max(1e-9, time.monotonic() - start_time)
                rate = generated / elapsed
                print(
                    f"[build_demand_label_dataset] processed={generated} "
                    f"elapsed_s={elapsed:.1f} windows_per_s={rate:.1f}",
                    file=sys.stderr,
                )

            if len(buffer) >= max(1, int(chunk_size)):
                columns = _write_csv_chunk(buffer, path=base_path, columns=columns, write_header=write_header)
                buffer.clear()
                write_header = False

        if buffer:
            _write_csv_chunk(buffer, path=base_path, columns=columns, write_header=write_header)

        if generated == 0:
            raise ValueError("No client-windows were generated. Check window size, stride, and selected clients.")

        _finalize_csv_output(
            base_path=base_path,
            output_path=output_path,
            pca_mode=pca_mode,
            pca_model_path=pca_model_path,
            pca_model_output_path=pca_model_output_path,
            chunk_size=chunk_size,
        )

    if return_dataframe:
        return pd.read_csv(output_path)
    return pd.DataFrame()


def build_demand_label_dataset(
    *,
    clean_path: Path | str,
    current_path: Path | str,
    output_path: Path | str,
    score_config: Optional[dict] = None,
    score_config_path: Optional[Path | str] = None,
    baseline_artifact_path: Optional[Path | str] = None,
    window_size_steps: int = 168,
    stride_steps: int = 24,
    forecast_horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    timestamp_col: Optional[str] = None,
    configured_freq: Optional[str] = None,
    selected_clients: Optional[Sequence[str]] = None,
    max_windows: Optional[int] = None,
    pca_mode: str = "fit",
    pca_model_path: Optional[Path | str] = None,
    pca_model_output_path: Optional[Path | str] = None,
    artifact_mode: str = "full",
    return_dataframe: bool = True,
    chunk_size: int = 5000,
    progress_every_windows: Optional[int] = 5000,
) -> pd.DataFrame:
    """Build one labeled row per client-window for surrogate-model training or runtime scoring."""
    if score_config is None and score_config_path is not None:
        score_config = load_yaml_config(score_config_path)
    score_config = score_config or {}
    baseline_artifact = load_baseline_artifact(baseline_artifact_path) if baseline_artifact_path else None

    source = load_window_source(
        clean_path,
        current_path,
        timestamp_col=timestamp_col,
        configured_freq=configured_freq or score_config.get("frequency"),
        selected_clients=selected_clients,
    )
    scorer = WideDemandQualityScorer(config=score_config)

    if artifact_mode not in {"full", "runtime_scores"}:
        raise ValueError("artifact_mode must be either 'full' or 'runtime_scores'.")

    output_path = Path(output_path)
    include_features = artifact_mode == "full"
    if output_path.suffix.lower() == ".csv":
        return _build_csv_artifact(
            source=source,
            output_path=output_path,
            scorer=scorer,
            score_config=score_config,
            baseline_artifact=baseline_artifact,
            include_features=include_features,
            window_size_steps=window_size_steps,
            stride_steps=stride_steps,
            forecast_horizon=forecast_horizon,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            max_windows=max_windows,
            pca_mode=pca_mode,
            pca_model_path=pca_model_path,
            pca_model_output_path=pca_model_output_path,
            return_dataframe=return_dataframe,
            chunk_size=chunk_size,
            progress_every_windows=progress_every_windows,
        )

    total_runtime_windows = max(0, len(source.expected_grid) - window_size_steps - forecast_horizon + 1)
    split_bounds = _runtime_split_bounds(
        len(source.expected_grid),
        window_size_steps=window_size_steps,
        forecast_horizon=forecast_horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    rows: list[dict[str, float | str]] = []
    generated = 0
    for window in iter_client_windows(
        source,
        window_size_steps=window_size_steps,
        stride_steps=stride_steps,
    ):
        if window.window_start_index >= total_runtime_windows:
            continue
        if max_windows is not None and generated >= int(max_windows):
            break
        rows.append(
            _build_window_row(
                window,
                scorer=scorer,
                score_config=score_config,
                baseline_artifact=baseline_artifact,
                include_features=include_features,
                split_bounds=split_bounds,
            )
        )
        generated += 1

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        raise ValueError("No client-windows were generated. Check window size, stride, and selected clients.")

    dimension_df = pd.DataFrame(
        {
            "completeness": dataset["y_completeness"].astype(float),
            "validity": dataset["y_validity"].astype(float),
            "consistency": dataset["y_consistency"].astype(float),
            "accuracy": dataset["y_accuracy"].astype(float),
            "freshness": dataset["y_freshness"].astype(float),
        }
    )

    if pca_mode == "fit":
        model = DemandPCAQualityModel.fit(dimension_df)
        if pca_model_output_path is not None:
            save_json(model.to_dict(), pca_model_output_path)
    elif pca_mode == "transform":
        if pca_model_path is None:
            raise ValueError("pca_model_path is required when pca_mode='transform'.")
        model = _load_pca_model(pca_model_path)
    else:
        raise ValueError("pca_mode must be 'fit' or 'transform'.")

    raw_scores = model.transform(dimension_df)
    dataset["y_final"] = _normalize_quality_score(raw_scores, dimension_df, model)
    dataset["quality_score"] = dataset["y_final"].astype(float)
    dataset["dimension_completeness"] = dataset["y_completeness"].astype(float)
    dataset["dimension_validity"] = dataset["y_validity"].astype(float)
    dataset["dimension_consistency"] = dataset["y_consistency"].astype(float)
    dataset["dimension_accuracy"] = dataset["y_accuracy"].astype(float)
    dataset["dimension_freshness"] = dataset["y_freshness"].astype(float)
    dataset = dataset.loc[:, _ordered_output_columns(dataset.columns.tolist())]
    _save_dataframe(dataset, output_path)
    return dataset
