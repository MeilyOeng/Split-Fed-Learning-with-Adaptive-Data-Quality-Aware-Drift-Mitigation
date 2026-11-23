"""Inference wrapper for demand-specific surrogate quality models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from .baseline import load_baseline_artifact
from .features import extract_client_window_features
from .io import load_yaml_config
from .train_xgboost import DIMENSION_TARGETS, FINAL_TARGET
from .windowing import ClientWindowSlice, collect_client_windows, load_window_source


def _require_xgboost():
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is required for demand surrogate inference. Install it in the active environment first."
        ) from exc
    return xgb


def _load_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _auto_find_yaml(artifact_dir: Path) -> Optional[Path]:
    candidates = sorted(list(artifact_dir.glob("*.yaml")) + list(artifact_dir.glob("*.yml")))
    return candidates[0] if candidates else None


def _auto_find_baseline(artifact_dir: Path) -> Optional[Path]:
    candidates = sorted(artifact_dir.glob("*baseline*.json"))
    return candidates[0] if candidates else None


@dataclass
class DemandQualityPredictor:
    """Load surrogate artifacts and predict scores for client-windows."""

    feature_names: list[str]
    final_feature_names: list[str]
    feature_medians: pd.Series
    dimension_models: dict[str, Any]
    final_model: Any
    scaler: Optional[Any]
    score_config: dict
    baseline_artifact: Optional[dict]

    @classmethod
    def load(
        cls,
        artifact_dir: Path | str,
        *,
        score_config_path: Optional[Path | str] = None,
        baseline_artifact_path: Optional[Path | str] = None,
    ) -> "DemandQualityPredictor":
        artifact_root = Path(artifact_dir)
        feature_names = list(_load_json(artifact_root / "feature_schema.json")["feature_names"])
        final_feature_names = list(_load_json(artifact_root / "final_feature_schema.json")["feature_names"])
        medians_payload = _load_json(artifact_root / "feature_medians.json")
        feature_medians = pd.Series(medians_payload, dtype=float)

        resolved_config_path = Path(score_config_path) if score_config_path is not None else _auto_find_yaml(artifact_root)
        score_config = load_yaml_config(resolved_config_path) if resolved_config_path is not None else {}

        resolved_baseline_path = (
            Path(baseline_artifact_path) if baseline_artifact_path is not None else _auto_find_baseline(artifact_root)
        )
        baseline_artifact = load_baseline_artifact(resolved_baseline_path) if resolved_baseline_path is not None else None

        scaler_path = artifact_root / "scaler.joblib"
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        xgb = _require_xgboost()
        dimension_models: dict[str, Any] = {}
        for target in DIMENSION_TARGETS:
            model = xgb.XGBRegressor()
            model.load_model(str(artifact_root / "models" / f"{target}.json"))
            dimension_models[target] = model
        final_model = xgb.XGBRegressor()
        final_model.load_model(str(artifact_root / "models" / f"{FINAL_TARGET}.json"))

        return cls(
            feature_names=feature_names,
            final_feature_names=final_feature_names,
            feature_medians=feature_medians,
            dimension_models=dimension_models,
            final_model=final_model,
            scaler=scaler,
            score_config=score_config,
            baseline_artifact=baseline_artifact,
        )

    def _prepare_raw_feature_row(self, features: dict[str, float]) -> pd.DataFrame:
        row = pd.DataFrame([features]).reindex(columns=self.feature_names)
        for col in row.columns:
            row[col] = pd.to_numeric(row[col], errors="coerce")
        row = row.fillna(self.feature_medians.reindex(self.feature_names))
        if self.scaler is None:
            return row
        scaled = self.scaler.transform(row)
        return pd.DataFrame(scaled, columns=self.feature_names)

    def predict_window(self, window: ClientWindowSlice) -> dict[str, float | str]:
        features = extract_client_window_features(
            window,
            config=self.score_config,
            baseline_artifact=self.baseline_artifact,
        )
        raw_row = self._prepare_raw_feature_row(features)

        outputs: dict[str, float | str] = {
            "client_id": window.client_id,
            "window_id": window.window_id,
            "window_start": window.window_start.isoformat(),
            "window_end": window.window_end.isoformat(),
        }
        final_feature_row = raw_row.copy()
        for target in DIMENSION_TARGETS:
            pred_name = f"pred_{target[2:]}"
            pred_value = float(np.clip(self.dimension_models[target].predict(raw_row)[0], 0.0, 1.0))
            outputs[target] = pred_value
            final_feature_row[pred_name] = pred_value

        final_feature_row = final_feature_row.reindex(columns=self.final_feature_names)
        outputs[FINAL_TARGET] = float(np.clip(self.final_model.predict(final_feature_row)[0], 0.0, 1.0))
        return outputs

    def predict_windows(self, windows: Sequence[ClientWindowSlice]) -> pd.DataFrame:
        return pd.DataFrame([self.predict_window(window) for window in windows])


def predict_demand_quality(
    *,
    clean_path: Path | str,
    current_path: Path | str,
    artifact_dir: Path | str,
    output_path: Optional[Path | str] = None,
    window_size_steps: int = 168,
    stride_steps: int = 24,
    timestamp_col: Optional[str] = None,
    configured_freq: Optional[str] = None,
    selected_clients: Optional[Sequence[str]] = None,
    latest_only: bool = True,
    score_config_path: Optional[Path | str] = None,
    baseline_artifact_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Run surrogate inference on demand client-windows."""
    predictor = DemandQualityPredictor.load(
        artifact_dir,
        score_config_path=score_config_path,
        baseline_artifact_path=baseline_artifact_path,
    )
    source = load_window_source(
        clean_path,
        current_path,
        timestamp_col=timestamp_col,
        configured_freq=configured_freq or predictor.score_config.get("frequency"),
        selected_clients=selected_clients,
    )
    windows = collect_client_windows(source, window_size_steps=window_size_steps, stride_steps=stride_steps)
    if latest_only:
        latest_by_client: dict[str, ClientWindowSlice] = {}
        for window in windows:
            latest_by_client[window.client_id] = window
        windows = list(latest_by_client.values())

    predictions = predictor.predict_windows(windows)
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".parquet":
            predictions.to_parquet(out_path, index=False)
        else:
            predictions.to_csv(out_path, index=False)
    return predictions
