"""Demand-specific labeled dataset generation for surrogate-model training."""

from __future__ import annotations

import json
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
    timestamp_col: Optional[str] = None,
    configured_freq: Optional[str] = None,
    selected_clients: Optional[Sequence[str]] = None,
    max_windows: Optional[int] = None,
    pca_mode: str = "fit",
    pca_model_path: Optional[Path | str] = None,
    pca_model_output_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Build one labeled row per client-window for surrogate-model training."""
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

    rows: list[dict[str, float | str]] = []
    for idx, window in enumerate(
        iter_client_windows(
            source,
            window_size_steps=window_size_steps,
            stride_steps=stride_steps,
        )
    ):
        if max_windows is not None and idx >= int(max_windows):
            break
        current = window.current_window
        clean = window.clean_window
        client_id = window.client_id
        comp = scorer.completeness.score_client(current, clean, client_id)
        val = scorer.validity.score_client(current, clean, client_id)
        cons = scorer.consistency.score_client(current, clean, client_id)
        acc = scorer.accuracy.score_client(current, clean, client_id)
        freshness = _offline_freshness_scores(window, score_config)
        features = extract_client_window_features(
            window,
            config=score_config,
            baseline_artifact=baseline_artifact,
        )

        row: dict[str, float | str] = {
            "client_id": client_id,
            "window_id": window.window_id,
            "window_start": window.window_start.isoformat(),
            "window_end": window.window_end.isoformat(),
            **features,
            "y_completeness": float(comp["completeness"]),
            "y_validity": float(val["validity"]),
            "y_consistency": float(cons["consistency"]),
            "y_accuracy": float(acc["accuracy"]),
            "y_freshness": float(freshness["y_freshness"]),
            "offline_F1": float(freshness["F1_offline"]),
            "offline_F2": float(freshness["F2_offline"]),
        }
        rows.append(row)

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

    ordered_columns = [
        "client_id",
        "window_id",
        "window_start",
        "window_end",
        *[col for col in dataset.columns if col not in {"client_id", "window_id", "window_start", "window_end", *TARGET_COLUMNS}],
        *TARGET_COLUMNS,
    ]
    dataset = dataset.loc[:, ordered_columns]
    _save_dataframe(dataset, output_path)
    return dataset
