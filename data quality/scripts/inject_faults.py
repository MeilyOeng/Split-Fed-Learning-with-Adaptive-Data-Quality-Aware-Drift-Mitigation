#!/usr/bin/env python3
"""CLI for wide-format demand fault injection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.fault_injection import inject_faults_to_csv
from modules.io import load_yaml_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Inject faults into wide-format demand data.")
    parser.add_argument("--input", required=True, type=Path, help="Path to clean demand CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Path to fault-injected demand CSV.")
    parser.add_argument("--report", required=True, type=Path, help="Path to detailed fault report CSV.")
    parser.add_argument("--timestamp-col", default="Fecha_Hora", help="Timestamp column name. Auto-detected when omitted.")
    parser.add_argument("--fault-config", required=True, type=Path, help="YAML config describing fault injection.")
    parser.add_argument(
        "--cluster-assignment",
        default=Path("dataset/cluster-assignment.csv"),
        type=Path,
        help="Cluster assignment CSV path used for cluster-shared fault injection.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic fault injection.")
    args = parser.parse_args()

    config = load_yaml_config(args.fault_config)
    inject_faults_to_csv(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        config=config,
        timestamp_col=args.timestamp_col,
        seed=args.seed,
        cluster_assignment_path=args.cluster_assignment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
