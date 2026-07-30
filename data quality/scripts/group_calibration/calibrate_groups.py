#!/usr/bin/env python3
"""Aggregate demand by category/subcategory and calibrate distribution plots."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


QUALITY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = QUALITY_ROOT.parent
MPL_CACHE_DIR = PROJECT_ROOT / "outputs" / "calibrations" / ".matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DEMAND_PATH = PROJECT_ROOT / "dataset" / "dataset.csv"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "dataset" / "general_categories.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "calibrations"
DEFAULT_THRESHOLD_OUTPUT_DIR = QUALITY_ROOT / "configs" / "thresholds"
DEFAULT_ACTIVE_SCORE_CONFIG = QUALITY_ROOT / "configs" / "data_quality_scoring.yaml"

REQUIRED_METADATA_COLUMNS = {"ID", "Categoria", "Subcategoria"}
SUMMARY_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate wide demand data into category and subcategory series, then generate calibration plots."
    )
    parser.add_argument("--demand", type=Path, default=DEFAULT_DEMAND_PATH, help="Wide demand CSV path.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH, help="Metadata CSV path.")
    parser.add_argument("--timestamp-col", default="Fecha_Hora", help="Timestamp column in the demand CSV.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output folder for calibration plots and summaries.",
    )
    parser.add_argument(
        "--threshold-output-dir",
        type=Path,
        default=DEFAULT_THRESHOLD_OUTPUT_DIR,
        help="Reusable output folder for generated threshold CSV/YAML config artifacts.",
    )
    parser.add_argument(
        "--active-score-config",
        type=Path,
        default=DEFAULT_ACTIVE_SCORE_CONFIG,
        help="Active data_quality_scoring.yaml to update with calibrated metadata/threshold sections.",
    )
    parser.add_argument(
        "--no-update-active-score-config",
        action="store_true",
        help="Only write threshold reference artifacts; do not update the active scoring config.",
    )
    parser.add_argument(
        "--upper-bound-value-quantile",
        type=float,
        default=0.999,
        help="Per-client demand quantile used to estimate validity upper-bound candidates.",
    )
    parser.add_argument(
        "--roc-diff-quantile",
        type=float,
        default=0.99,
        help="Per-client absolute-difference quantile used to estimate rate-of-change threshold candidates.",
    )
    parser.add_argument(
        "--group-client-quantile",
        type=float,
        default=0.95,
        help="Quantile across client-level candidates used for category/subcategory thresholds.",
    )
    parser.add_argument(
        "--threshold-margin",
        type=float,
        default=1.10,
        help="Multiplier applied to calibrated thresholds for a conservative margin.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def normalize_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return text or "group"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_group_label(series: pd.Series, fallback: str) -> pd.Series:
    cleaned = series.fillna(fallback).astype(str).str.strip()
    cleaned = cleaned.replace({"": fallback, "nan": fallback, "None": fallback})
    return cleaned


def load_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    missing = sorted(REQUIRED_METADATA_COLUMNS - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata file is missing required columns: {missing}")

    metadata = metadata.copy()
    metadata["client_id"] = metadata["ID"].map(normalize_id)
    metadata["Categoria"] = normalize_group_label(metadata["Categoria"], "UNSPECIFIED_CATEGORY")
    metadata["Subcategoria"] = normalize_group_label(metadata["Subcategoria"], "UNSPECIFIED_SUBCATEGORY")
    metadata = metadata.drop_duplicates(subset=["client_id"], keep="first").reset_index(drop=True)
    return metadata[["client_id", "Categoria", "Subcategoria"]]


def load_demand(path: Path, timestamp_col: str) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path, low_memory=False)
    if timestamp_col not in frame.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' was not found in {path}")

    frame = frame.copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="coerce")
    frame = frame.dropna(subset=[timestamp_col]).sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)

    client_cols = [str(col) for col in frame.columns if col != timestamp_col]
    rename_map = {col: normalize_id(col) for col in client_cols}
    frame = frame.rename(columns=rename_map)
    client_cols = [rename_map[col] for col in client_cols]

    for col in client_cols:
        if not is_numeric_dtype(frame[col]):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = frame[col].astype(float, copy=False)

    # Flip the whole demand table together when the source looks like the raw
    # smart-meter export: values are non-positive globally, with at least one
    # negative reading. This avoids leaving zero-heavy clients unflipped.
    demand_values = frame.loc[:, client_cols].to_numpy(dtype=float, copy=False)
    finite_mask = np.isfinite(demand_values)
    if finite_mask.any():
        finite_values = demand_values[finite_mask]
        has_negative = bool((finite_values < 0).any())
        has_positive = bool((finite_values > 0).any())
        if has_negative and not has_positive:
            frame.loc[:, client_cols] = frame.loc[:, client_cols] * -1.0

    return frame, client_cols


def aggregate_level(
    demand: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    timestamp_col: str,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    demand_cols = [col for col in demand.columns if col != timestamp_col]
    available = metadata.loc[metadata["client_id"].isin(demand_cols)].copy()
    if available.empty:
        raise ValueError(f"No metadata IDs matched the demand columns for {group_col}.")

    group_meta = (
        available.groupby(group_col)
        .agg(
            client_count=("client_id", "nunique"),
            categories=("Categoria", lambda s: " | ".join(sorted(set(s)))),
            subcategories=("Subcategoria", lambda s: " | ".join(sorted(set(s)))),
        )
        .reset_index()
        .rename(columns={group_col: "group_name"})
    )

    group_map = available.set_index("client_id")[group_col]
    common_cols = [col for col in demand_cols if col in group_map.index]
    wide = demand.loc[:, common_cols]
    aggregated_sum = wide.T.groupby(group_map.loc[common_cols]).sum(min_count=1).T
    aggregated_mean = wide.T.groupby(group_map.loc[common_cols]).mean().T
    aggregated_sum = aggregated_sum.reindex(sorted(aggregated_sum.columns), axis=1)
    aggregated_mean = aggregated_mean.reindex(sorted(aggregated_mean.columns), axis=1)
    aggregated_sum.insert(0, timestamp_col, demand[timestamp_col].to_numpy())
    aggregated_mean.insert(0, timestamp_col, demand[timestamp_col].to_numpy())

    group_meta["group_name"] = group_meta["group_name"].astype(str)
    group_meta = group_meta.sort_values("group_name", kind="mergesort").reset_index(drop=True)
    return aggregated_sum, aggregated_mean, group_meta


def summarize_series(values: np.ndarray) -> dict[str, float | int]:
    percentiles = np.percentile(values, SUMMARY_PERCENTILES)
    quantiles = dict(zip(SUMMARY_PERCENTILES, percentiles.tolist()))
    stats: dict[str, float | int] = {
        "sample_count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "range": float(np.max(values) - np.min(values)),
    }
    for pct, value in quantiles.items():
        stats[f"p{pct:02d}"] = float(value)
    stats["median"] = float(quantiles[50])
    return stats


def build_summary(
    aggregated: pd.DataFrame,
    group_meta: pd.DataFrame,
    *,
    timestamp_col: str,
    measure_name: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    meta_index = group_meta.set_index("group_name")

    for group_name in [col for col in aggregated.columns if col != timestamp_col]:
        series = pd.to_numeric(aggregated[group_name], errors="coerce").dropna()
        if series.empty:
            continue
        stats = summarize_series(series.to_numpy(dtype=np.float64))
        meta = meta_index.loc[group_name]
        record = {
            "group_name": group_name,
            "measure": measure_name,
            "client_count": int(meta["client_count"]),
            "categories": str(meta["categories"]),
            "subcategories": str(meta["subcategories"]),
        }
        record.update(stats)
        records.append(record)

    summary = pd.DataFrame(records)
    if summary.empty:
        raise ValueError("No aggregated groups contained valid demand values.")
    return summary.sort_values(["max", "group_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def _clamp_quantile(value: float, name: str) -> float:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return float(value)


def _finite_array(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    return values[np.isfinite(values)]


def build_client_threshold_stats(
    demand: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    timestamp_col: str,
    upper_bound_value_quantile: float,
    roc_diff_quantile: float,
) -> pd.DataFrame:
    value_q = _clamp_quantile(upper_bound_value_quantile, "upper_bound_value_quantile")
    diff_q = _clamp_quantile(roc_diff_quantile, "roc_diff_quantile")
    demand_cols = [col for col in demand.columns if col != timestamp_col]
    meta_index = metadata.set_index("client_id")
    records: list[dict[str, object]] = []

    for client_id in demand_cols:
        if client_id not in meta_index.index:
            continue
        values = _finite_array(demand[client_id])
        if values.size == 0:
            continue
        diffs = np.abs(np.diff(values))
        diffs = diffs[np.isfinite(diffs)]
        meta = meta_index.loc[client_id]
        records.append(
            {
                "client_id": client_id,
                "Categoria": str(meta["Categoria"]),
                "Subcategoria": str(meta["Subcategoria"]),
                "sample_count": int(values.size),
                "value_max": float(np.max(values)),
                "value_p95": float(np.quantile(values, 0.95)),
                "value_p99": float(np.quantile(values, 0.99)),
                "value_p999": float(np.quantile(values, value_q)),
                "abs_diff_sample_count": int(diffs.size),
                "abs_diff_max": float(np.max(diffs)) if diffs.size else 0.0,
                "abs_diff_p95": float(np.quantile(diffs, 0.95)) if diffs.size else 0.0,
                "abs_diff_p99": float(np.quantile(diffs, diff_q)) if diffs.size else 0.0,
            }
        )

    out = pd.DataFrame(records)
    if out.empty:
        raise ValueError("No matched clients had finite demand values for threshold calibration.")
    return out.sort_values("client_id", kind="mergesort").reset_index(drop=True)


def summarize_threshold_group(
    client_stats: pd.DataFrame,
    *,
    group_col: str | None,
    group_client_quantile: float,
    threshold_margin: float,
) -> pd.DataFrame:
    group_q = _clamp_quantile(group_client_quantile, "group_client_quantile")
    margin = float(threshold_margin)
    if margin <= 0:
        raise ValueError("threshold_margin must be positive")

    records: list[dict[str, object]] = []
    groups: list[tuple[str, pd.DataFrame]]
    if group_col is None:
        groups = [("__global__", client_stats)]
    else:
        groups = [(str(name), group.copy()) for name, group in client_stats.groupby(group_col, sort=True)]

    for group_name, group in groups:
        value_candidates = group["value_p999"].dropna().to_numpy(dtype=np.float64)
        diff_candidates = group["abs_diff_p99"].dropna().to_numpy(dtype=np.float64)
        if value_candidates.size == 0 or diff_candidates.size == 0:
            continue
        upper_base = float(np.quantile(value_candidates, group_q))
        roc_base = float(np.quantile(diff_candidates, group_q))
        records.append(
            {
                "group_name": group_name,
                "client_count": int(group["client_id"].nunique()),
                "value_candidate_min": float(np.min(value_candidates)),
                "value_candidate_median": float(np.median(value_candidates)),
                "value_candidate_p95": float(np.quantile(value_candidates, 0.95)),
                "value_candidate_max": float(np.max(value_candidates)),
                "abs_diff_candidate_min": float(np.min(diff_candidates)),
                "abs_diff_candidate_median": float(np.median(diff_candidates)),
                "abs_diff_candidate_p95": float(np.quantile(diff_candidates, 0.95)),
                "abs_diff_candidate_max": float(np.max(diff_candidates)),
                "calibrated_upper_bound": float(max(0.0, upper_base * margin)),
                "calibrated_rate_of_change_threshold": float(max(0.0, roc_base * margin)),
                "group_client_quantile": group_q,
                "threshold_margin": margin,
            }
        )

    out = pd.DataFrame(records)
    if out.empty:
        raise ValueError(f"No threshold summaries could be built for {group_col or 'global'}.")
    return out.sort_values("group_name", kind="mergesort").reset_index(drop=True)


def _yaml_scalar(value: object) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(value)):
            return "null"
        return f"{float(value):.12g}"
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _write_mapping(lines: list[str], indent: int, mapping: dict[str, float]) -> None:
    pad = " " * indent
    if not mapping:
        lines.append(f"{pad}{{}}")
        return
    for key, value in mapping.items():
        lines.append(f"{pad}{_yaml_scalar(key)}: {_yaml_scalar(value)}")


def build_threshold_config_lines(
    *,
    metadata_path: Path,
    global_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    subcategory_summary: pd.DataFrame,
    include_header: bool = True,
) -> list[str]:
    global_row = global_summary.iloc[0]
    cat_upper = dict(zip(category_summary["group_name"], category_summary["calibrated_upper_bound"]))
    cat_roc = dict(zip(category_summary["group_name"], category_summary["calibrated_rate_of_change_threshold"]))
    sub_upper = dict(zip(subcategory_summary["group_name"], subcategory_summary["calibrated_upper_bound"]))
    sub_roc = dict(zip(subcategory_summary["group_name"], subcategory_summary["calibrated_rate_of_change_threshold"]))

    lines: list[str] = []
    if include_header:
        lines.extend(
            [
                "# Generated by data quality/scripts/group_calibration/calibrate_groups.py",
                "# Reference copy of the threshold sections written into data_quality_scoring.yaml.",
            ]
        )
    lines.extend(
        [
            "metadata:",
            f"  path: {_yaml_scalar(str(metadata_path))}",
            "  client_id_column: ID",
            "  category_column: Categoria",
            "  subcategory_column: Subcategoria",
            "",
            "thresholds:",
            "  allow_clean_reference_fallback: false",
            "",
            "validity:",
            "  lower_bound: 0.0",
            f"  upper_bound: {_yaml_scalar(global_row['calibrated_upper_bound'])}",
            "  upper_bound_by_category:",
        ]
    )
    _write_mapping(lines, 4, cat_upper)
    lines.append("  upper_bound_by_subcategory:")
    _write_mapping(lines, 4, sub_upper)
    lines.extend([
        "",
        "consistency:",
        "  freeze_min_len: 6",
        "  freeze_atol: 0.0",
        "  allow_constant_series: false",
        f"  rate_of_change_threshold: {_yaml_scalar(global_row['calibrated_rate_of_change_threshold'])}",
        "  rate_of_change_threshold_by_category:",
    ])
    _write_mapping(lines, 4, cat_roc)
    lines.append("  rate_of_change_threshold_by_subcategory:")
    _write_mapping(lines, 4, sub_roc)
    lines.append("  roc_multiplier: 1.5")
    return lines


def write_threshold_config_yaml(
    path: Path,
    *,
    metadata_path: Path,
    global_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    subcategory_summary: pd.DataFrame,
) -> list[str]:
    lines = build_threshold_config_lines(
        metadata_path=metadata_path,
        global_summary=global_summary,
        category_summary=category_summary,
        subcategory_summary=subcategory_summary,
        include_header=True,
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def _top_level_key(line: str) -> str | None:
    if not line or line.startswith(" ") or line.startswith("#") or ":" not in line:
        return None
    return line.split(":", 1)[0].strip() or None


def _split_top_level_blocks(lines: list[str]) -> dict[str, tuple[int, int]]:
    starts: list[tuple[str, int]] = []
    for idx, line in enumerate(lines):
        key = _top_level_key(line)
        if key is not None:
            starts.append((key, idx))
    blocks: dict[str, tuple[int, int]] = {}
    for pos, (key, start) in enumerate(starts):
        end = starts[pos + 1][1] if pos + 1 < len(starts) else len(lines)
        blocks[key] = (start, end)
    return blocks


def _extract_top_level_section(lines: list[str], section: str) -> list[str]:
    blocks = _split_top_level_blocks(lines)
    if section not in blocks:
        raise ValueError(f"Generated threshold config did not contain section {section!r}.")
    start, end = blocks[section]
    section_lines = list(lines[start:end])
    while section_lines and not section_lines[-1].strip():
        section_lines.pop()
    return section_lines


def update_active_score_config(
    path: Path,
    *,
    threshold_lines: list[str],
    sections: tuple[str, ...] = ("metadata", "thresholds", "validity", "consistency"),
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Active scoring config does not exist: {path}")

    target_lines = path.read_text(encoding="utf-8").splitlines()
    replacement_sections = {section: _extract_top_level_section(threshold_lines, section) for section in sections}

    for section in sections:
        blocks = _split_top_level_blocks(target_lines)
        replacement = replacement_sections[section]
        if section in blocks:
            start, end = blocks[section]
            target_lines = target_lines[:start] + replacement + [""] + target_lines[end:]
        else:
            target_lines.extend(["", *replacement])

    compacted: list[str] = []
    blank_seen = False
    for line in target_lines:
        if line.strip():
            compacted.append(line)
            blank_seen = False
        elif not blank_seen:
            compacted.append("")
            blank_seen = True
    path.write_text("\n".join(compacted).strip() + "\n", encoding="utf-8")


def write_threshold_calibration_outputs(
    *,
    demand: pd.DataFrame,
    metadata: pd.DataFrame,
    timestamp_col: str,
    metadata_path: Path,
    output_dir: Path,
    upper_bound_value_quantile: float,
    roc_diff_quantile: float,
    group_client_quantile: float,
    threshold_margin: float,
    active_score_config_path: Path | None = None,
) -> dict[str, str]:
    threshold_dir = output_dir
    ensure_dir(threshold_dir)
    client_stats = build_client_threshold_stats(
        demand,
        metadata,
        timestamp_col=timestamp_col,
        upper_bound_value_quantile=upper_bound_value_quantile,
        roc_diff_quantile=roc_diff_quantile,
    )
    global_summary = summarize_threshold_group(
        client_stats,
        group_col=None,
        group_client_quantile=group_client_quantile,
        threshold_margin=threshold_margin,
    )
    category_summary = summarize_threshold_group(
        client_stats,
        group_col="Categoria",
        group_client_quantile=group_client_quantile,
        threshold_margin=threshold_margin,
    )
    subcategory_summary = summarize_threshold_group(
        client_stats,
        group_col="Subcategoria",
        group_client_quantile=group_client_quantile,
        threshold_margin=threshold_margin,
    )

    client_path = threshold_dir / "client_threshold_stats.csv"
    global_path = threshold_dir / "global_thresholds.csv"
    category_path = threshold_dir / "category_thresholds.csv"
    subcategory_path = threshold_dir / "subcategory_thresholds.csv"
    yaml_path = threshold_dir / "data_quality_thresholds.yaml"
    client_stats.to_csv(client_path, index=False)
    global_summary.to_csv(global_path, index=False)
    category_summary.to_csv(category_path, index=False)
    subcategory_summary.to_csv(subcategory_path, index=False)
    threshold_lines = write_threshold_config_yaml(
        yaml_path,
        metadata_path=metadata_path,
        global_summary=global_summary,
        category_summary=category_summary,
        subcategory_summary=subcategory_summary,
    )
    if active_score_config_path is not None:
        update_active_score_config(active_score_config_path, threshold_lines=threshold_lines)
    return {
        "client_threshold_stats": str(client_path),
        "global_thresholds": str(global_path),
        "category_thresholds": str(category_path),
        "subcategory_thresholds": str(subcategory_path),
        "threshold_config_yaml": str(yaml_path),
        "active_score_config": str(active_score_config_path) if active_score_config_path is not None else "",
    }


def save_matching_report(
    path: Path,
    *,
    metadata: pd.DataFrame,
    demand_client_cols: Iterable[str],
    matched: pd.DataFrame,
) -> None:
    demand_ids = set(demand_client_cols)
    metadata_ids = set(metadata["client_id"])
    payload = {
        "metadata_clients": int(len(metadata_ids)),
        "demand_clients": int(len(demand_ids)),
        "matched_clients": int(matched["client_id"].nunique()),
        "metadata_without_demand": int(len(metadata_ids - demand_ids)),
        "demand_without_metadata": int(len(demand_ids - metadata_ids)),
        "categories": int(metadata["Categoria"].nunique()),
        "subcategories": int(metadata["Subcategoria"].nunique()),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_group_distribution(
    *,
    group_name: str,
    level_name: str,
    measure_name: str,
    value_label: str,
    values: np.ndarray,
    client_count: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    fig.suptitle(f"{level_name.title()} {measure_name.replace('_', ' ').title()}: {group_name}", fontsize=14)

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    median = float(np.percentile(values, 50))
    p05 = float(np.percentile(values, 5))
    p95 = float(np.percentile(values, 95))
    p99 = float(np.percentile(values, 99))

    bins = min(60, max(20, int(np.sqrt(values.size))))
    axes[0].hist(values, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.9)
    for marker, color, label in (
        (vmin, "#6B7280", "min"),
        (p05, "#0F766E", "p05"),
        (median, "#C2410C", "median"),
        (p95, "#B91C1C", "p95"),
        (vmax, "#111827", "max"),
    ):
        axes[0].axvline(marker, color=color, linestyle="--", linewidth=1.5, label=f"{label}={marker:.2f}")
    axes[0].set_title("Histogram")
    axes[0].set_xlabel(value_label)
    axes[0].set_ylabel("Frequency")
    axes[0].legend(fontsize=8)

    sorted_values = np.sort(values)
    survival = np.arange(sorted_values.size, 0, -1, dtype=np.float64) / float(sorted_values.size)
    survival = np.clip(survival, 1.0 / float(sorted_values.size), 1.0)
    axes[1].plot(sorted_values, survival, color="#D64F4F", linewidth=2)
    axes[1].axvline(p95, color="#0F766E", linestyle="--", linewidth=1.2, label=f"p95={p95:.2f}")
    axes[1].axvline(p99, color="#B91C1C", linestyle="--", linewidth=1.2, label=f"p99={p99:.2f}")
    axes[1].axvline(vmax, color="#111827", linestyle=":", linewidth=1.4, label=f"max={vmax:.2f}")
    axes[1].set_yscale("log")
    axes[1].set_title("Upper Tail")
    axes[1].set_xlabel(value_label)
    axes[1].set_ylabel("P(X >= x)")
    axes[1].legend(fontsize=8)

    percentile_axis = np.linspace(0, 100, 201)
    percentile_values = np.percentile(values, percentile_axis)
    axes[2].plot(percentile_axis, percentile_values, color="#2E8B57", linewidth=2)
    axes[2].scatter([0, 50, 100], [vmin, median, vmax], color="#111827", s=18, zorder=3)
    axes[2].axhline(vmin, color="#6B7280", linestyle="--", linewidth=1)
    axes[2].axhline(vmax, color="#111827", linestyle="--", linewidth=1)
    axes[2].fill_between(percentile_axis, p05, p95, color="#A7D3C5", alpha=0.2)
    axes[2].set_title("Percentile Curve")
    axes[2].set_xlabel("Percentile")
    axes[2].set_ylabel(value_label)
    axes[2].text(
        0.02,
        0.98,
        (
            f"clients={client_count}\n"
            f"min={vmin:.2f}\n"
            f"median={median:.2f}\n"
            f"p95={p95:.2f}\n"
            f"max={vmax:.2f}"
        ),
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#D1D5DB"},
    )

    for ax in axes:
        ax.grid(alpha=0.2, linewidth=0.6)

    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_range_overview(
    summary: pd.DataFrame,
    *,
    level_name: str,
    measure_name: str,
    value_label: str,
    out_path: Path,
) -> None:
    ordered = summary.sort_values(["p95", "median"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
    y_positions = np.arange(len(ordered))
    figure_height = max(5.5, 0.33 * len(ordered) + 1.5)

    fig, ax = plt.subplots(figsize=(14, figure_height))
    ax.hlines(y_positions, ordered["min"], ordered["max"], color="#94A3B8", linewidth=1.8, label="min-max")
    ax.hlines(y_positions, ordered["p05"], ordered["p95"], color="#2563EB", linewidth=4.5, label="p05-p95")
    ax.scatter(ordered["median"], y_positions, color="#DC2626", s=24, label="median", zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered["group_name"])
    ax.invert_yaxis()
    ax.set_xlabel(value_label)
    ax.set_title(f"{level_name.title()} {measure_name.replace('_', ' ').title()} min-max and percentile ranges")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_measure_outputs(
    *,
    aggregated: pd.DataFrame,
    summary: pd.DataFrame,
    level_name: str,
    measure_name: str,
    value_label: str,
    timestamp_col: str,
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)
    group_plot_dir = output_dir / "groups"
    ensure_dir(group_plot_dir)

    aggregated.to_csv(output_dir / f"{level_name}_{measure_name}_timeseries.csv", index=False)
    summary.to_csv(output_dir / f"{level_name}_{measure_name}_summary.csv", index=False)
    plot_range_overview(
        summary,
        level_name=level_name,
        measure_name=measure_name,
        value_label=value_label,
        out_path=output_dir / f"{level_name}_{measure_name}_range_overview.png",
    )

    summary_index = summary.set_index("group_name")
    for group_name in [col for col in aggregated.columns if col != timestamp_col]:
        series = pd.to_numeric(aggregated[group_name], errors="coerce").dropna()
        if series.empty:
            continue
        safe_name = slugify(group_name)
        plot_group_distribution(
            group_name=group_name,
            level_name=level_name,
            measure_name=measure_name,
            value_label=value_label,
            values=series.to_numpy(dtype=np.float64),
            client_count=int(summary_index.loc[group_name, "client_count"]),
            out_path=group_plot_dir / f"{safe_name}.png",
        )


def write_level_outputs(
    *,
    aggregated_sum: pd.DataFrame,
    aggregated_mean: pd.DataFrame,
    sum_summary: pd.DataFrame,
    mean_summary: pd.DataFrame,
    level_name: str,
    timestamp_col: str,
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)
    write_measure_outputs(
        aggregated=aggregated_sum,
        summary=sum_summary,
        level_name=level_name,
        measure_name="aggregated",
        value_label="Aggregated demand",
        timestamp_col=timestamp_col,
        output_dir=output_dir / "aggregated",
    )
    write_measure_outputs(
        aggregated=aggregated_mean,
        summary=mean_summary,
        level_name=level_name,
        measure_name="per_client_average",
        value_label="Average demand per client",
        timestamp_col=timestamp_col,
        output_dir=output_dir / "per_client_average",
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_root
    ensure_dir(output_dir)

    log(f"Loading metadata from {args.metadata}")
    metadata = load_metadata(args.metadata)

    log(f"Loading demand data from {args.demand}")
    demand, client_cols = load_demand(args.demand, args.timestamp_col)
    matched = metadata.loc[metadata["client_id"].isin(client_cols)].copy()
    if matched.empty:
        raise ValueError("No client IDs matched between metadata and demand data.")

    save_matching_report(
        output_dir / "matching_report.json",
        metadata=metadata,
        demand_client_cols=client_cols,
        matched=matched,
    )

    log("Calibrating DQ validity/rate-of-change thresholds")
    threshold_outputs = write_threshold_calibration_outputs(
        demand=demand,
        metadata=matched,
        timestamp_col=args.timestamp_col,
        metadata_path=args.metadata,
        output_dir=args.threshold_output_dir,
        upper_bound_value_quantile=args.upper_bound_value_quantile,
        roc_diff_quantile=args.roc_diff_quantile,
        group_client_quantile=args.group_client_quantile,
        threshold_margin=args.threshold_margin,
        active_score_config_path=None if args.no_update_active_score_config else args.active_score_config,
    )

    for level_name, group_col in (("category", "Categoria"), ("subcategory", "Subcategoria")):
        log(f"Aggregating {level_name} demand series")
        aggregated_sum, aggregated_mean, group_meta = aggregate_level(
            demand,
            matched,
            timestamp_col=args.timestamp_col,
            group_col=group_col,
        )
        sum_summary = build_summary(
            aggregated_sum,
            group_meta,
            timestamp_col=args.timestamp_col,
            measure_name="aggregated",
        )
        mean_summary = build_summary(
            aggregated_mean,
            group_meta,
            timestamp_col=args.timestamp_col,
            measure_name="per_client_average",
        )
        write_level_outputs(
            aggregated_sum=aggregated_sum,
            aggregated_mean=aggregated_mean,
            sum_summary=sum_summary,
            mean_summary=mean_summary,
            level_name=level_name,
            timestamp_col=args.timestamp_col,
            output_dir=output_dir / level_name,
        )

    run_summary = {
        "demand_path": str(args.demand),
        "metadata_path": str(args.metadata),
        "timestamp_col": args.timestamp_col,
        "output_dir": str(output_dir),
        "threshold_output_dir": str(args.threshold_output_dir),
        "active_score_config": str(args.active_score_config),
        "updated_active_score_config": bool(not args.no_update_active_score_config),
        "rows": int(len(demand)),
        "matched_clients": int(matched["client_id"].nunique()),
        "measures": ["aggregated", "per_client_average"],
        "threshold_outputs": threshold_outputs,
        "upper_bound_value_quantile": args.upper_bound_value_quantile,
        "roc_diff_quantile": args.roc_diff_quantile,
        "group_client_quantile": args.group_client_quantile,
        "threshold_margin": args.threshold_margin,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    log(f"Finished. Outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
