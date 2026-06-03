#!/usr/bin/env python3
"""CLI for the end-to-end demand quality pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from demand_quality.pipeline import run_quality_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full wide-demand quality pipeline.")
    parser.add_argument("--input", required=True, type=Path, help="Clean demand CSV.")
    parser.add_argument("--faulty-output", required=True, type=Path, help="Output path for fault-injected data.")
    parser.add_argument("--fault-report", required=True, type=Path, help="Detailed fault report CSV.")
    parser.add_argument("--score-output", required=True, type=Path, help="Per-client quality score CSV.")
    parser.add_argument("--pca-output", required=True, type=Path, help="PCA and aggregation-weight CSV.")
    parser.add_argument("--fault-config", required=True, type=Path, help="Fault injection YAML config.")
    parser.add_argument("--score-config", required=True, type=Path, help="Scoring YAML config.")
    parser.add_argument("--timestamp-col", default=None, help="Timestamp column name. Auto-detected when omitted.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic injection.")
    parser.add_argument("--pca-model-output", default=None, type=Path, help="Optional JSON path for fitted PCA parameters.")
    args = parser.parse_args()

    run_quality_pipeline(
        input_path=args.input,
        faulty_output_path=args.faulty_output,
        fault_report_path=args.fault_report,
        score_output_path=args.score_output,
        pca_output_path=args.pca_output,
        fault_config_path=args.fault_config,
        score_config_path=args.score_config,
        timestamp_col=args.timestamp_col,
        seed=args.seed,
        pca_model_output_path=args.pca_model_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
