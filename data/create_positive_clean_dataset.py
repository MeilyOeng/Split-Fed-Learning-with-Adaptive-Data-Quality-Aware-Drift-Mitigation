#!/usr/bin/env python3
"""Create the canonical positive clean dataset from the raw negative dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_INPUT_PATH = Path("dataset") / "dataset_raw_negative.csv"
DEFAULT_OUTPUT_PATH = Path("dataset") / "dataset.csv"
DEFAULT_REPORT_PATH = Path("dataset") / "dataset_positive_preprocessing_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the raw negative-demand dataset, multiply demand columns by -1 once, "
            "and write the canonical positive clean dataset plus a preprocessing report."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Raw wide CSV with negative demand values.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Canonical positive-output CSV path.")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="JSON report path with before/after summary statistics.",
    )
    parser.add_argument("--timestamp-col", default="Fecha_Hora", help="Timestamp column name to exclude from flipping.")
    parser.add_argument(
        "--value-columns",
        nargs="*",
        default=None,
        help="Optional explicit demand/client columns. Defaults to all non-timestamp columns.",
    )
    return parser.parse_args()


def _resolve_value_columns(frame: pd.DataFrame, timestamp_col: str, requested: Iterable[str] | None) -> list[str]:
    if requested:
        missing = [col for col in requested if col not in frame.columns]
        if missing:
            raise ValueError(f"Requested value columns not found in input dataset: {missing}")
        return [str(col) for col in requested]
    return [str(col) for col in frame.columns if str(col) != str(timestamp_col)]


def _clean_zero(value: float) -> float:
    return 0.0 if value == 0.0 else float(value)


def _summarize_values(frame: pd.DataFrame, value_columns: list[str]) -> dict:
    numeric = frame.loc[:, value_columns].apply(pd.to_numeric, errors="coerce")
    flat = numeric.to_numpy(dtype=float).reshape(-1)
    finite = flat[pd.notna(flat)]
    if finite.size == 0:
        return {
            "count_finite": 0,
            "min": None,
            "max": None,
            "mean": None,
            "negative_count": 0,
            "negative_fraction": 0.0,
        }
    negative_count = int((finite < 0).sum())
    return {
        "count_finite": int(finite.size),
        "min": _clean_zero(float(finite.min())),
        "max": _clean_zero(float(finite.max())),
        "mean": _clean_zero(float(finite.mean())),
        "negative_count": negative_count,
        "negative_fraction": float(negative_count / max(1, finite.size)),
    }


def create_positive_clean_dataset(
    *,
    input_path: Path | str,
    output_path: Path | str,
    report_path: Path | str,
    timestamp_col: str = "Fecha_Hora",
    value_columns: Iterable[str] | None = None,
) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)
    report_path = Path(report_path)

    frame = pd.read_csv(input_path, low_memory=False)
    if timestamp_col not in frame.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in {input_path}")

    resolved_value_columns = _resolve_value_columns(frame, timestamp_col, value_columns)
    before_summary = _summarize_values(frame, resolved_value_columns)

    output = frame.copy()
    output.loc[:, resolved_value_columns] = (
        output.loc[:, resolved_value_columns].apply(pd.to_numeric, errors="coerce") * -1.0
    )
    after_summary = _summarize_values(output, resolved_value_columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "timestamp_col": timestamp_col,
        "value_column_count": len(resolved_value_columns),
        "value_columns_preview": resolved_value_columns[:20],
        "before": before_summary,
        "after": after_summary,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    report = create_positive_clean_dataset(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        timestamp_col=args.timestamp_col,
        value_columns=args.value_columns,
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
