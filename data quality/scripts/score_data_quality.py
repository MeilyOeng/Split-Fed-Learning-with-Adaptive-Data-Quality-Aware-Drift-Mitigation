#!/usr/bin/env python3
"""CLI for per-client data quality scoring on wide demand data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from demand_quality.io import load_yaml_config
from demand_quality.scoring import score_wide_demand_quality


def main() -> int:
    parser = argparse.ArgumentParser(description="Score wide-format demand data quality per client.")
    parser.add_argument("--input", required=True, type=Path, help="Faulty demand CSV.")
    parser.add_argument("--clean-reference", required=True, type=Path, help="Clean demand CSV used as reference.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV with subcheck and dimension scores.")
    parser.add_argument("--timestamp-col", default=None, help="Timestamp column name. Auto-detected when omitted.")
    parser.add_argument("--config", required=True, type=Path, help="YAML scoring config.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    score_wide_demand_quality(
        current_path=args.input,
        clean_reference_path=args.clean_reference,
        output_path=args.output,
        config=config,
        timestamp_col=args.timestamp_col,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
