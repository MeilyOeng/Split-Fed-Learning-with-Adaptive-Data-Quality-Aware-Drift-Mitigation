#!/usr/bin/env python3
"""CLI for the end-to-end demand quality pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.pipeline import run_quality_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full wide-demand quality pipeline.")
    parser.add_argument("--input", default="dataset/dataset.csv", type=Path, help="Clean demand CSV.")
    parser.add_argument("--faulty-output", default="dataset/quality output/faulty_dataset.csv", type=Path, help="Output path for fault-injected data.")
    parser.add_argument("--fault-report", default="dataset/quality output/faulty data/fault_injection_report.csv", type=Path, help="Detailed fault report CSV.")
    parser.add_argument("--score-output", default="dataset/quality output/client_quality_scores.csv", type=Path, help="Per-client quality score CSV.")
    parser.add_argument("--pca-output", default="dataset/quality output/data score/pca.json", type=Path, help="PCA and aggregation-weight CSV.")
    parser.add_argument("--fault-config", default="data quality/configs/fault_injection_dq.yaml", type=Path, help="Fault injection YAML config.")
    parser.add_argument("--score-config", default="data quality/configs/data_quality_scoring.yaml", type=Path, help="Scoring YAML config.")
    parser.add_argument(
        "--cluster-assignment",
        default=Path("dataset/cluster-assignment.csv"),
        type=Path,
        help="Cluster assignment CSV path used for cluster-shared fault injection.",
    )
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
        cluster_assignment_path=args.cluster_assignment,
        timestamp_col=args.timestamp_col,
        seed=args.seed,
        pca_model_output_path=args.pca_model_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
