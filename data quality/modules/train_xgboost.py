"""Demand-specific XGBoost surrogate-model training."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .label_dataset import TARGET_COLUMNS


DIMENSION_TARGETS = [
    "y_completeness",
    "y_validity",
    "y_consistency",
    "y_accuracy",
    "y_freshness",
]
FINAL_TARGET = "y_final"
METADATA_COLUMNS = ["client_id", "window_id", "window_start", "window_end"]
EXCLUDED_PREFIXES = ("offline_",)


def _save_json(payload: dict[str, Any], path: Path | str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _load_dataframe(path: Path | str) -> pd.DataFrame:
    src = Path(path)
    if src.suffix.lower() == ".parquet":
        return pd.read_parquet(src)
    return pd.read_csv(src)


def _require_xgboost():
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is required for demand surrogate training. Install it in the active environment first."
        ) from exc
    return xgb


def _prepare_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    excluded = set(METADATA_COLUMNS) | set(TARGET_COLUMNS)
    feature_cols = [
        col
        for col in df.columns
        if col not in excluded and not any(col.startswith(prefix) for prefix in EXCLUDED_PREFIXES)
    ]
    x = df.loc[:, feature_cols].copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x, feature_cols


def _split_unique_times(
    times: pd.Series,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, pd.Series]:
    sorted_times = pd.Series(pd.to_datetime(times, errors="coerce")).dropna().drop_duplicates().sort_values().reset_index(drop=True)
    if sorted_times.shape[0] < 3:
        raise ValueError("Need at least three distinct window_end timestamps for train/val/test splitting.")

    n_times = int(sorted_times.shape[0])
    train_count = max(1, int(np.floor(n_times * train_fraction)))
    val_count = max(1, int(np.floor(n_times * val_fraction)))
    if train_count + val_count >= n_times:
        val_count = max(1, n_times - train_count - 1)
    test_count = n_times - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if val_count > 1:
            val_count -= 1
        else:
            train_count = max(1, train_count - 1)

    train_times = sorted_times.iloc[:train_count]
    val_times = sorted_times.iloc[train_count : train_count + val_count]
    test_times = sorted_times.iloc[train_count + val_count :]
    return {
        "train_times": train_times,
        "val_times": val_times,
        "test_times": test_times,
    }


def _metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, pred)))
    return {
        "mae": float(mean_absolute_error(y_true_arr, pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true_arr, pred)) if y_true_arr.size > 1 else 0.0,
    }


def _tail_validation_split(
    x: pd.DataFrame,
    y: pd.Series,
    holdout_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if len(x) <= 2:
        return x, y, x, y
    split_idx = max(1, int(np.floor(len(x) * (1.0 - holdout_fraction))))
    split_idx = min(split_idx, len(x) - 1)
    return (
        x.iloc[:split_idx],
        y.iloc[:split_idx],
        x.iloc[split_idx:],
        y.iloc[split_idx:],
    )


def _default_xgb_params(seed: int) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": int(seed),
        "n_jobs": 0,
        "verbosity": 0,
    }


def _maybe_optuna_params(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
    n_trials: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_params = _default_xgb_params(seed)
    if n_trials <= 0:
        return base_params, {"enabled": False, "reason": "optuna_trials<=0"}

    try:
        import optuna
    except ModuleNotFoundError:
        return base_params, {"enabled": False, "reason": "optuna_not_installed"}

    xgb = _require_xgboost()

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "n_estimators": trial.suggest_int("n_estimators", 150, 700, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 8.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
            "tree_method": "hist",
            "random_state": int(seed),
            "n_jobs": 0,
            "verbosity": 0,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(x_train, y_train)
        pred = np.clip(model.predict(x_val), 0.0, 1.0)
        return mean_absolute_error(y_val, pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=int(n_trials))
    best_params = dict(base_params)
    best_params.update(study.best_params)
    return best_params, {
        "enabled": True,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": int(n_trials),
    }


def _fit_model(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    params: dict[str, Any],
):
    xgb = _require_xgboost()
    model = xgb.XGBRegressor(**params)
    model.fit(x, y)
    return model


def _expanding_oof_predictions(
    x_dev: pd.DataFrame,
    y_dev: pd.Series,
    dev_times: pd.Series,
    *,
    params: dict[str, Any],
    n_folds: int,
) -> tuple[np.ndarray, int]:
    unique_times = pd.Series(pd.to_datetime(dev_times, errors="coerce")).dropna().drop_duplicates().sort_values().reset_index(drop=True)
    if unique_times.shape[0] < 3:
        return np.full(len(x_dev), np.nan, dtype=float), 0

    n_blocks = min(int(n_folds) + 1, int(unique_times.shape[0]))
    blocks = [block for block in np.array_split(unique_times.to_numpy(), n_blocks) if len(block) > 0]
    preds = np.full(len(x_dev), np.nan, dtype=float)
    fold_count = 0
    for block_idx in range(1, len(blocks)):
        train_times = np.concatenate(blocks[:block_idx])
        valid_times = blocks[block_idx]
        train_mask = dev_times.isin(train_times).to_numpy()
        valid_mask = dev_times.isin(valid_times).to_numpy()
        if train_mask.sum() == 0 or valid_mask.sum() == 0:
            continue
        model = _fit_model(x_dev.loc[train_mask], y_dev.loc[train_mask], params=params)
        preds[valid_mask] = np.clip(model.predict(x_dev.loc[valid_mask]), 0.0, 1.0)
        fold_count += 1
    return preds, fold_count


def train_demand_xgboost_models(
    *,
    dataset_path: Path | str,
    output_dir: Path | str,
    dq_config_path: Optional[Path | str] = None,
    baseline_artifact_path: Optional[Path | str] = None,
    pca_model_path: Optional[Path | str] = None,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    use_scaler: bool = False,
    optuna_trials: int = 0,
    seed: int = 42,
    n_oof_folds: int = 5,
) -> dict[str, Any]:
    """Train five dimension models plus one stacked final model."""
    dataset = _load_dataframe(dataset_path)
    if "window_end" not in dataset.columns:
        raise ValueError("Training dataset must contain a window_end column.")

    dataset["window_end"] = pd.to_datetime(dataset["window_end"], errors="coerce")
    dataset = dataset.dropna(subset=["window_end"]).sort_values(["window_end", "client_id"], kind="mergesort").reset_index(drop=True)
    if dataset.empty:
        raise ValueError("Training dataset is empty after parsing window_end.")

    x_raw, original_feature_cols = _prepare_feature_frame(dataset)
    y_all = dataset.loc[:, TARGET_COLUMNS].astype(float).copy()
    split = _split_unique_times(dataset["window_end"], train_fraction=train_fraction, val_fraction=val_fraction)
    train_mask = dataset["window_end"].isin(split["train_times"]).to_numpy()
    val_mask = dataset["window_end"].isin(split["val_times"]).to_numpy()
    test_mask = dataset["window_end"].isin(split["test_times"]).to_numpy()
    dev_mask = train_mask | val_mask

    x_dev_raw = x_raw.loc[dev_mask].copy()
    y_dev = y_all.loc[dev_mask].copy()
    x_test_raw = x_raw.loc[test_mask].copy()
    y_test = y_all.loc[test_mask].copy()

    constant_cols = [col for col in x_dev_raw.columns if x_dev_raw[col].nunique(dropna=True) <= 1]
    all_nan_cols = [col for col in x_dev_raw.columns if x_dev_raw[col].isna().all()]
    drop_cols = sorted(set(constant_cols + all_nan_cols))
    if drop_cols:
        x_raw = x_raw.drop(columns=drop_cols)
        x_dev_raw = x_dev_raw.drop(columns=drop_cols)
        x_test_raw = x_test_raw.drop(columns=drop_cols)

    feature_cols = x_dev_raw.columns.tolist()
    medians = x_dev_raw.median(numeric_only=True).astype(float)
    x_raw_filled = x_raw.loc[:, feature_cols].fillna(medians)
    x_dev = x_dev_raw.loc[:, feature_cols].fillna(medians)
    x_test = x_test_raw.loc[:, feature_cols].fillna(medians)

    scaler = None
    if use_scaler:
        scaler = StandardScaler().fit(x_dev)
        x_raw_processed = pd.DataFrame(scaler.transform(x_raw), columns=feature_cols, index=x_raw.index)
        x_dev_processed = pd.DataFrame(scaler.transform(x_dev), columns=feature_cols, index=x_dev.index)
        x_test_processed = pd.DataFrame(scaler.transform(x_test), columns=feature_cols, index=x_test.index)
    else:
        x_raw_processed = x_raw_filled.copy()
        x_dev_processed = x_dev.copy()
        x_test_processed = x_test.copy()

    x_train_tune = x_raw_processed.loc[train_mask]
    y_train_tune = y_all.loc[train_mask]
    x_val_tune = x_raw_processed.loc[val_mask]
    y_val_tune = y_all.loc[val_mask]
    dev_times = dataset.loc[dev_mask, "window_end"].reset_index(drop=True)

    dimension_models: dict[str, Any] = {}
    dimension_params: dict[str, dict[str, Any]] = {}
    dimension_tuning: dict[str, dict[str, Any]] = {}
    oof_frame = pd.DataFrame(index=x_dev_processed.index)
    test_pred_frame = pd.DataFrame(index=x_test_processed.index)
    metrics: dict[str, Any] = {}

    for target in DIMENSION_TARGETS:
        params, tuning_meta = _maybe_optuna_params(
            x_train=x_train_tune,
            y_train=y_train_tune[target],
            x_val=x_val_tune,
            y_val=y_val_tune[target],
            seed=seed,
            n_trials=optuna_trials,
        )
        model = _fit_model(x_dev_processed, y_dev[target], params=params)
        oof_pred, oof_folds = _expanding_oof_predictions(
            x_dev_processed.reset_index(drop=True),
            y_dev[target].reset_index(drop=True),
            dev_times.reset_index(drop=True),
            params=params,
            n_folds=n_oof_folds,
        )
        test_pred = np.clip(model.predict(x_test_processed), 0.0, 1.0)

        pred_col = f"pred_{target[2:]}"
        oof_frame[pred_col] = oof_pred
        test_pred_frame[pred_col] = test_pred
        dimension_models[target] = model
        dimension_params[target] = params
        dimension_tuning[target] = {**tuning_meta, "oof_folds": int(oof_folds)}
        metrics[target] = {
            "test": _metrics(y_test[target], test_pred),
            "dev_oof": _metrics(y_dev[target].loc[~np.isnan(oof_pred)], oof_pred[~np.isnan(oof_pred)])
            if np.isfinite(oof_pred).any()
            else None,
        }

    stack_ready_mask = oof_frame.notna().all(axis=1).to_numpy()
    stacking_mode = "expanding_oof"
    if int(stack_ready_mask.sum()) == 0:
        stacking_mode = "fallback_full_model_predictions"
        for target in DIMENSION_TARGETS:
            pred_col = f"pred_{target[2:]}"
            oof_frame[pred_col] = np.clip(dimension_models[target].predict(x_dev_processed), 0.0, 1.0)
        stack_ready_mask = np.ones(len(oof_frame), dtype=bool)

    final_train_x = pd.concat(
        [
            x_dev_processed.reset_index(drop=True).loc[stack_ready_mask].reset_index(drop=True),
            oof_frame.reset_index(drop=True).loc[stack_ready_mask].reset_index(drop=True),
        ],
        axis=1,
    )
    final_train_y = y_dev[FINAL_TARGET].reset_index(drop=True).loc[stack_ready_mask].reset_index(drop=True)
    final_test_x = pd.concat(
        [
            x_test_processed.reset_index(drop=True),
            test_pred_frame.reset_index(drop=True),
        ],
        axis=1,
    )
    final_tune_x_train, final_tune_y_train, final_tune_x_val, final_tune_y_val = _tail_validation_split(
        final_train_x,
        final_train_y,
    )

    final_params, final_tuning = _maybe_optuna_params(
        x_train=final_tune_x_train,
        y_train=final_tune_y_train,
        x_val=final_tune_x_val,
        y_val=final_tune_y_val,
        seed=seed,
        n_trials=optuna_trials,
    )
    final_model = _fit_model(final_train_x, final_train_y, params=final_params)
    final_test_pred = np.clip(final_model.predict(final_test_x), 0.0, 1.0)
    metrics[FINAL_TARGET] = {
        "test": _metrics(y_test[FINAL_TARGET], final_test_pred),
        "stack_train_rows": int(final_train_x.shape[0]),
        "stacking_mode": stacking_mode,
    }

    artifact_dir = Path(output_dir)
    models_dir = artifact_dir / "models"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    for target, model in dimension_models.items():
        model.save_model(str(models_dir / f"{target}.json"))
    final_model.save_model(str(models_dir / f"{FINAL_TARGET}.json"))

    if scaler is not None:
        joblib.dump(scaler, artifact_dir / "scaler.joblib")

    _save_json({"feature_names": feature_cols}, artifact_dir / "feature_schema.json")
    _save_json({"feature_names": final_train_x.columns.tolist()}, artifact_dir / "final_feature_schema.json")
    _save_json({col: float(val) for col, val in medians.items()}, artifact_dir / "feature_medians.json")
    _save_json(
        {
            "dimension_targets": DIMENSION_TARGETS,
            "final_target": FINAL_TARGET,
            "metadata_columns": METADATA_COLUMNS,
            "drop_columns": drop_cols,
            "original_feature_columns": original_feature_cols,
            "use_scaler": bool(use_scaler),
            "seed": int(seed),
            "train_fraction": float(train_fraction),
            "val_fraction": float(val_fraction),
            "splits": {
                "train_windows": int(train_mask.sum()),
                "val_windows": int(val_mask.sum()),
                "test_windows": int(test_mask.sum()),
                "train_start": str(split["train_times"].iloc[0]),
                "train_end": str(split["train_times"].iloc[-1]),
                "val_start": str(split["val_times"].iloc[0]),
                "val_end": str(split["val_times"].iloc[-1]),
                "test_start": str(split["test_times"].iloc[0]),
                "test_end": str(split["test_times"].iloc[-1]),
            },
            "model_params": {**dimension_params, FINAL_TARGET: final_params},
            "tuning": {**dimension_tuning, FINAL_TARGET: final_tuning},
            "metrics": metrics,
        },
        artifact_dir / "training_metadata.json",
    )

    if dq_config_path is not None:
        shutil.copy2(dq_config_path, artifact_dir / Path(dq_config_path).name)
    if baseline_artifact_path is not None:
        shutil.copy2(baseline_artifact_path, artifact_dir / Path(baseline_artifact_path).name)
    if pca_model_path is not None:
        shutil.copy2(pca_model_path, artifact_dir / Path(pca_model_path).name)

    return {
        "artifact_dir": str(artifact_dir),
        "feature_count": int(len(feature_cols)),
        "metrics": metrics,
    }
