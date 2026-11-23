#!/usr/bin/env python3
"""CLI for calibrating local historical baselines for demand quality scoring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.baseline import calibrate_demand_baseline_from_csv
from modules.io import load_yaml_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate per-client local historical baselines from clean demand data.")
    parser.add_argument("--input", required=True, type=Path, help="Path to clean historical demand CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path for the calibrated baseline artifact.")
    parser.add_argument("--timestamp-col", default="Fecha_Hora", help="Timestamp column name. Auto-detected when omitted.")
    parser.add_argument("--config", required=True, type=Path, help="YAML scoring config.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    calibrate_demand_baseline_from_csv(
        input_path=args.input,
        output_path=args.output,
        config=config,
        timestamp_col=args.timestamp_col,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
