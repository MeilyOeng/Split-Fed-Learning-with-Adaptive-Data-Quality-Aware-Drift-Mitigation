#!/usr/bin/env python3
"""CLI to build demand surrogate training labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.label_dataset import build_demand_label_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one labeled row per demand client-window.")
    parser.add_argument("--clean", required=True, type=Path, help="Clean reference demand CSV.")
    parser.add_argument("--current", required=True, type=Path, help="Current or faulty demand CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV or parquet path.")
    parser.add_argument("--score-config", required=True, type=Path, help="Data-quality scoring YAML.")
    parser.add_argument("--baseline-artifact", default=None, type=Path, help="Optional baseline JSON artifact.")
    parser.add_argument("--pca-mode", choices=["fit", "transform"], default="fit", help="Fit a new PCA combiner or reuse an existing one.")
    parser.add_argument("--pca-model-path", default=None, type=Path, help="Existing PCA JSON when --pca-mode=transform.")
    parser.add_argument("--pca-model-output", default=None, type=Path, help="Where to save the fitted PCA JSON when --pca-mode=fit.")
    parser.add_argument("--window-size-steps", type=int, default=168, help="Window size in time steps.")
    parser.add_argument("--stride-steps", type=int, default=24, help="Stride in time steps.")
    parser.add_argument("--forecast-horizon", type=int, default=1, help="Forecast horizon in time steps.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Runtime train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Runtime validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Runtime test split ratio.")
    parser.add_argument("--timestamp-col", default=None, help="Explicit timestamp column.")
    parser.add_argument("--configured-freq", default=None, help="Explicit expected sampling frequency, e.g. 1h.")
    parser.add_argument("--clients", nargs="*", default=None, help="Optional subset of client ids.")
    parser.add_argument("--max-windows", type=int, default=None, help="Optional cap for quick validation runs.")
    parser.add_argument(
        "--artifact-mode",
        choices=["full", "runtime_scores"],
        default="full",
        help="Use 'full' for surrogate-model training features, or 'runtime_scores' for lightweight SplitFed window scores.",
    )
    parser.add_argument("--chunk-size", type=int, default=5000, help="CSV chunk size for streaming output.")
    parser.add_argument(
        "--progress-every-windows",
        type=int,
        default=5000,
        help="Emit progress every N generated windows; use 0 to disable.",
    )
    args = parser.parse_args()

    build_demand_label_dataset(
        clean_path=args.clean,
        current_path=args.current,
        output_path=args.output,
        score_config_path=args.score_config,
        baseline_artifact_path=args.baseline_artifact,
        window_size_steps=args.window_size_steps,
        stride_steps=args.stride_steps,
        forecast_horizon=args.forecast_horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        timestamp_col=args.timestamp_col,
        configured_freq=args.configured_freq,
        selected_clients=args.clients,
        max_windows=args.max_windows,
        pca_mode=args.pca_mode,
        pca_model_path=args.pca_model_path,
        pca_model_output_path=args.pca_model_output,
        artifact_mode=args.artifact_mode,
        return_dataframe=False,
        chunk_size=args.chunk_size,
        progress_every_windows=(args.progress_every_windows or None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
