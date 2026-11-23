#!/usr/bin/env python3
"""CLI to run surrogate demand-quality inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.predictor import predict_demand_quality


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict demand data-quality scores from saved surrogate artifacts.")
    parser.add_argument("--clean", required=True, type=Path, help="Clean reference demand CSV.")
    parser.add_argument("--current", required=True, type=Path, help="Current demand CSV to score.")
    parser.add_argument("--artifact-dir", required=True, type=Path, help="Artifact directory produced by train_demand_xgboost.py.")
    parser.add_argument("--output", default=None, type=Path, help="Optional CSV or parquet output path.")
    parser.add_argument("--window-size-steps", type=int, default=168, help="Window size in time steps.")
    parser.add_argument("--stride-steps", type=int, default=24, help="Stride in time steps.")
    parser.add_argument("--timestamp-col", default=None, help="Explicit timestamp column.")
    parser.add_argument("--configured-freq", default=None, help="Explicit expected sampling frequency, e.g. 1h.")
    parser.add_argument("--clients", nargs="*", default=None, help="Optional subset of client ids.")
    parser.add_argument("--all-windows", action="store_true", help="Predict every generated window instead of only the latest per client.")
    parser.add_argument("--score-config", default=None, type=Path, help="Optional scoring YAML override.")
    parser.add_argument("--baseline-artifact", default=None, type=Path, help="Optional baseline JSON override.")
    args = parser.parse_args()

    predict_demand_quality(
        clean_path=args.clean,
        current_path=args.current,
        artifact_dir=args.artifact_dir,
        output_path=args.output,
        window_size_steps=args.window_size_steps,
        stride_steps=args.stride_steps,
        timestamp_col=args.timestamp_col,
        configured_freq=args.configured_freq,
        selected_clients=args.clients,
        latest_only=not args.all_windows,
        score_config_path=args.score_config,
        baseline_artifact_path=args.baseline_artifact,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
