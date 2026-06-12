#!/usr/bin/env python3
"""CLI to train demand surrogate XGBoost models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.train_xgboost import train_demand_xgboost_models


def main() -> int:
    parser = argparse.ArgumentParser(description="Train five demand dimension models plus one final stacked model.")
    parser.add_argument("--dataset", required=True, type=Path, help="Labeled CSV or parquet produced by build_demand_label_dataset.py.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Artifact directory for saved models and metadata.")
    parser.add_argument("--dq-config", default=None, type=Path, help="Optional copy of the scoring YAML to bundle with artifacts.")
    parser.add_argument("--baseline-artifact", default=None, type=Path, help="Optional copy of the baseline JSON to bundle with artifacts.")
    parser.add_argument("--pca-model", default=None, type=Path, help="Optional copy of the PCA JSON to bundle with artifacts.")
    parser.add_argument("--train-fraction", type=float, default=0.7, help="Fraction of unique window_end times reserved for train.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Fraction of unique window_end times reserved for validation.")
    parser.add_argument("--use-scaler", action="store_true", help="Apply StandardScaler to raw features before training.")
    parser.add_argument("--optuna-trials", type=int, default=0, help="Optional Optuna trials per target when Optuna is installed.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--oof-folds", type=int, default=5, help="Number of expanding folds for stacked OOF predictions.")
    args = parser.parse_args()

    result = train_demand_xgboost_models(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        dq_config_path=args.dq_config,
        baseline_artifact_path=args.baseline_artifact,
        pca_model_path=args.pca_model,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        use_scaler=args.use_scaler,
        optuna_trials=args.optuna_trials,
        seed=args.seed,
        n_oof_folds=args.oof_folds,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
