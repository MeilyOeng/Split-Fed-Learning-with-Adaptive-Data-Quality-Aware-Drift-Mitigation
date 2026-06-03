#!/usr/bin/env python3
"""CLI for PCA-based client quality aggregation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from demand_quality.io import load_yaml_config
from demand_quality.pca_quality import compute_pca_quality_scores_from_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute PCA-based client quality scores and aggregation weights.")
    parser.add_argument("--input", required=True, type=Path, help="CSV with per-client dimension scores.")
    parser.add_argument("--output", required=True, type=Path, help="CSV with PCA quality scores and aggregation weights.")
    parser.add_argument("--config", required=True, type=Path, help="YAML config.")
    parser.add_argument("--model-output", default=None, type=Path, help="Optional JSON path for the fitted PCA model.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    compute_pca_quality_scores_from_csv(
        input_path=args.input,
        output_path=args.output,
        config=config,
        model_output_path=args.model_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
