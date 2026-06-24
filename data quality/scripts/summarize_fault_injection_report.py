#!/usr/bin/env python3
"""Summarize a fault-injection report and, optionally, exact row/window impact."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TIMESTAMP_META_COLS = {
    "__dq_missing_timestamp__",
    "__dq_duplicate_timestamp_count__",
}


def normalize_client_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a fault-injection report and optional exact row/window impact."
    )
    parser.add_argument("--report", required=True, type=Path, help="Fault injection report CSV.")
    parser.add_argument("--clean", type=Path, default=None, help="Optional clean wide CSV for exact diffing.")
    parser.add_argument("--faulty", type=Path, default=None, help="Optional faulty wide CSV for exact diffing.")
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown/text summary output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON summary output path.")
    parser.add_argument("--per-client-output", type=Path, default=None, help="Optional per-client impact CSV output path.")
    parser.add_argument("--timestamp-col", default="Fecha_Hora", help="Timestamp column name.")
    parser.add_argument("--window-size-steps", type=int, default=24, help="Runtime window size.")
    parser.add_argument("--forecast-horizon", type=int, default=1, help="Runtime forecast horizon.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--top-clients", type=int, default=15, help="Number of top impacted clients to show.")
    return parser.parse_args()


def parse_parameters(value: Any) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    return loaded if isinstance(loaded, dict) else {"value": loaded}


def ensure_report_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, default in (
        ("cluster_id", None),
        ("scope", "client"),
        ("shared_timestamp_event", False),
    ):
        if col not in out.columns:
            out[col] = default
    out["client_id"] = out["client_id"].map(normalize_client_id)
    out["dimension"] = out["dimension"].astype(str)
    out["subcheck"] = out["subcheck"].astype(str)
    out["fault_type"] = out["fault_type"].astype(str)
    out["scope"] = out["scope"].fillna("client").astype(str)
    out["cluster_id"] = out["cluster_id"].where(out["cluster_id"].notna(), None)
    out["affected_rows"] = pd.to_numeric(out["affected_rows"], errors="coerce").fillna(0).astype(int)
    out["affected_fraction"] = pd.to_numeric(out["affected_fraction"], errors="coerce").fillna(0.0).astype(float)
    out["shared_timestamp_event"] = out["shared_timestamp_event"].fillna(False).astype(bool)
    out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce")
    out["end_time"] = pd.to_datetime(out["end_time"], errors="coerce")
    out["parameters_dict"] = out["parameters_used"].map(parse_parameters)
    return out


def summarize_report(df: pd.DataFrame) -> dict[str, Any]:
    event_df = df.loc[df["dimension"] != "summary"].copy()
    summary_df = df.loc[df["dimension"] == "summary"].copy()
    real_client_event_df = event_df.loc[~event_df["client_id"].astype(str).str.startswith("__")].copy()
    per_client_final = summary_df.loc[
        (summary_df["fault_type"] == "final_net_corruption")
        & (summary_df["subcheck"] == "final_rows_changed")
        & (summary_df["client_id"] != "__all_selected_clients__")
    ].copy()
    overall_final = summary_df.loc[
        (summary_df["fault_type"] == "final_net_corruption")
        & (summary_df["subcheck"] == "final_rows_changed")
        & (summary_df["client_id"] == "__all_selected_clients__")
    ].copy()

    selected_clients_from_summary: list[str] = []
    overall_final_row: dict[str, Any] | None = None
    if not overall_final.empty:
        overall_final_row = overall_final.iloc[0].to_dict()
        params = dict(overall_final_row.get("parameters_dict") or {})
        selected_clients_from_summary = [normalize_client_id(x) for x in params.get("selected_clients", [])]

    event_clients = sorted(
        {
            normalize_client_id(cid)
            for cid in real_client_event_df["client_id"].dropna().astype(str)
            if cid and cid != "__all_selected_clients__"
        }
    )
    selected_clients = selected_clients_from_summary or event_clients

    dimension_summary = (
        event_df.groupby("dimension", dropna=False)
        .agg(
            event_count=("fault_type", "size"),
            affected_rows_sum=("affected_rows", "sum"),
            affected_rows_mean=("affected_rows", "mean"),
            clients_impacted=("client_id", lambda s: s.astype(str).nunique()),
        )
        .reset_index()
        .sort_values(["affected_rows_sum", "event_count", "dimension"], ascending=[False, False, True], kind="mergesort")
    )

    fault_type_summary = (
        event_df.groupby(["dimension", "fault_type"], dropna=False)
        .agg(
            event_count=("fault_type", "size"),
            affected_rows_sum=("affected_rows", "sum"),
            affected_rows_mean=("affected_rows", "mean"),
            clients_impacted=("client_id", lambda s: s.astype(str).nunique()),
        )
        .reset_index()
        .sort_values(["affected_rows_sum", "event_count", "dimension", "fault_type"], ascending=[False, False, True, True], kind="mergesort")
    )

    subcheck_summary = (
        event_df.groupby(["dimension", "subcheck"], dropna=False)
        .agg(
            event_count=("subcheck", "size"),
            affected_rows_sum=("affected_rows", "sum"),
            affected_rows_mean=("affected_rows", "mean"),
            clients_impacted=("client_id", lambda s: s.astype(str).nunique()),
        )
        .reset_index()
        .sort_values(["affected_rows_sum", "event_count", "dimension", "subcheck"], ascending=[False, False, True, True], kind="mergesort")
    )

    scope_summary = (
        event_df.groupby("scope", dropna=False)
        .agg(
            event_count=("scope", "size"),
            affected_rows_sum=("affected_rows", "sum"),
        )
        .reset_index()
        .sort_values(["affected_rows_sum", "event_count", "scope"], ascending=[False, False, True], kind="mergesort")
    )

    cluster_summary = pd.DataFrame()
    if event_df["cluster_id"].notna().any():
        cluster_summary = (
            event_df.loc[event_df["cluster_id"].notna()]
            .groupby("cluster_id", dropna=False)
            .agg(
                event_count=("cluster_id", "size"),
                affected_rows_sum=("affected_rows", "sum"),
                clients_impacted=("client_id", lambda s: s.astype(str).nunique()),
            )
            .reset_index()
            .sort_values(["affected_rows_sum", "event_count", "cluster_id"], ascending=[False, False, True], kind="mergesort")
        )

    client_event_summary = (
        real_client_event_df.groupby("client_id", dropna=False)
        .agg(
            event_count=("client_id", "size"),
            affected_rows_event_sum=("affected_rows", "sum"),
            first_fault_time=("start_time", "min"),
            last_fault_time=("end_time", "max"),
        )
        .reset_index()
    )
    if not per_client_final.empty:
        client_event_summary = client_event_summary.merge(
            per_client_final[["client_id", "affected_rows", "affected_fraction"]].rename(
                columns={
                    "affected_rows": "final_net_changed_rows_report",
                    "affected_fraction": "final_net_changed_fraction_report",
                }
            ),
            on="client_id",
            how="left",
        )
    client_event_summary = client_event_summary.sort_values(
        ["affected_rows_event_sum", "event_count", "client_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    report_stats = {
        "report_rows_total": int(len(df)),
        "event_rows_total": int(len(event_df)),
        "summary_rows_total": int(len(summary_df)),
        "selected_clients_count": int(len(selected_clients)),
        "event_clients_count": int(len(event_clients)),
        "shared_timestamp_event_count": int(event_df["shared_timestamp_event"].sum()),
        "event_affected_rows_sum": int(event_df["affected_rows"].sum()),
        "event_affected_rows_mean": float(event_df["affected_rows"].mean()) if not event_df.empty else 0.0,
        "first_event_time": None if event_df["start_time"].dropna().empty else str(event_df["start_time"].min()),
        "last_event_time": None if event_df["end_time"].dropna().empty else str(event_df["end_time"].max()),
        "overall_final_net_changed_rows_report": int(overall_final_row["affected_rows"]) if overall_final_row else None,
        "overall_final_net_changed_fraction_report": float(overall_final_row["affected_fraction"]) if overall_final_row else None,
        "per_client_final_net_changed_rows_sum_report": int(per_client_final["affected_rows"].sum()) if not per_client_final.empty else None,
    }

    return {
        "stats": report_stats,
        "selected_clients": selected_clients,
        "dimension_summary": dimension_summary,
        "fault_type_summary": fault_type_summary,
        "subcheck_summary": subcheck_summary,
        "scope_summary": scope_summary,
        "cluster_summary": cluster_summary,
        "client_event_summary": client_event_summary,
    }


def read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def load_wide_subset(
    path: Path,
    timestamp_col: str,
    client_cols: list[str],
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    header = read_header(path)
    available = set(header)
    usecols = [timestamp_col]
    for col in client_cols:
        if col in available:
            usecols.append(col)
    for col in extra_cols or []:
        if col in available and col not in usecols:
            usecols.append(col)
    if timestamp_col not in usecols:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in {path}")
    df = pd.read_csv(path, usecols=usecols)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    return df


def values_equal_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, equal_nan=False)


def num_windows(num_rows: int, window_size: int, forecast_horizon: int) -> int:
    return max(0, num_rows - window_size - forecast_horizon + 1)


def split_bounds(
    total_rows: int,
    window_size: int,
    forecast_horizon: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, tuple[int, int]]:
    total_windows = num_windows(total_rows, window_size, forecast_horizon)
    train_windows = int(total_windows * train_ratio)
    val_windows = int(total_windows * val_ratio)
    test_windows = total_windows - train_windows - val_windows
    train_end = train_windows
    val_end = train_end + val_windows
    test_end = val_end + test_windows
    return {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, test_end)}


def affected_window_mask(
    changed_rows_mask: np.ndarray,
    total_rows: int,
    window_size: int,
    forecast_horizon: int,
) -> np.ndarray:
    total_windows = num_windows(total_rows, window_size, forecast_horizon)
    if total_windows <= 0:
        return np.zeros(0, dtype=bool)

    coverage = np.zeros(total_windows + 1, dtype=int)
    max_offset = window_size + forecast_horizon - 1
    changed_idxs = np.flatnonzero(changed_rows_mask)
    for row_idx in changed_idxs:
        start = max(0, int(row_idx) - max_offset)
        end = min(int(row_idx), total_windows - 1)
        if start <= end:
            coverage[start] += 1
            coverage[end + 1] -= 1
    return np.cumsum(coverage[:-1]) > 0


def count_windows_by_mode(mask: np.ndarray, bounds: dict[str, tuple[int, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for mode, (start, end) in bounds.items():
        out[mode] = int(mask[start:end].sum()) if end > start else 0
    return out


def compute_exact_impact(
    clean_path: Path,
    faulty_path: Path,
    selected_clients: list[str],
    *,
    timestamp_col: str,
    window_size: int,
    forecast_horizon: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, Any]:
    faulty_header = read_header(faulty_path)
    faulty_clients = [
        str(col)
        for col in faulty_header
        if col != timestamp_col and col not in TIMESTAMP_META_COLS
    ]
    if selected_clients:
        faulty_clients = [client for client in faulty_clients if client in set(selected_clients)]
    if not faulty_clients:
        raise ValueError("No selected client columns found in faulty dataset for exact analysis.")

    clean_df = load_wide_subset(clean_path, timestamp_col, faulty_clients)
    faulty_df = load_wide_subset(faulty_path, timestamp_col, faulty_clients, extra_cols=sorted(TIMESTAMP_META_COLS))

    common_clients = [client for client in faulty_clients if client in clean_df.columns and client in faulty_df.columns]
    if not common_clients:
        raise ValueError("No overlapping client columns were found between clean and faulty datasets.")

    clean_df = clean_df[[timestamp_col] + common_clients].copy()
    extra_faulty_cols = [col for col in sorted(TIMESTAMP_META_COLS) if col in faulty_df.columns]
    faulty_df = faulty_df[[timestamp_col] + common_clients + extra_faulty_cols].copy()

    if len(clean_df) != len(faulty_df) or not clean_df[timestamp_col].equals(faulty_df[timestamp_col]):
        raise ValueError(
            "Clean and faulty datasets do not share the same restored timestamp grid. "
            "Exact row/window analysis requires aligned timestamps."
        )

    total_rows = int(len(clean_df))
    bounds = split_bounds(total_rows, window_size, forecast_horizon, train_ratio, val_ratio, test_ratio)
    total_windows = num_windows(total_rows, window_size, forecast_horizon)

    clean_values = clean_df[common_clients].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    faulty_values = faulty_df[common_clients].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    value_changed = ~values_equal_matrix(clean_values, faulty_values)

    missing_rows = (
        pd.to_numeric(faulty_df["__dq_missing_timestamp__"], errors="coerce").fillna(0).to_numpy(dtype=int) > 0
        if "__dq_missing_timestamp__" in faulty_df.columns
        else np.zeros(total_rows, dtype=bool)
    )
    duplicate_rows = (
        pd.to_numeric(faulty_df["__dq_duplicate_timestamp_count__"], errors="coerce").fillna(0).to_numpy(dtype=int) > 0
        if "__dq_duplicate_timestamp_count__" in faulty_df.columns
        else np.zeros(total_rows, dtype=bool)
    )
    duplicate_extra_rows_total = (
        int(pd.to_numeric(faulty_df["__dq_duplicate_timestamp_count__"], errors="coerce").fillna(0).sum())
        if "__dq_duplicate_timestamp_count__" in faulty_df.columns
        else 0
    )
    structural_changed = missing_rows | duplicate_rows
    overall_changed = structural_changed | value_changed.any(axis=1)

    overall_window_mask = affected_window_mask(overall_changed, total_rows, window_size, forecast_horizon)
    overall_windows_by_mode = count_windows_by_mode(overall_window_mask, bounds)

    per_client_records: list[dict[str, Any]] = []
    union_client_window_count_sum = {"train": 0, "val": 0, "test": 0}

    for client_idx, client_id in enumerate(common_clients):
        client_value_changed = value_changed[:, client_idx]
        client_final_changed = structural_changed | client_value_changed
        client_window_mask = affected_window_mask(client_final_changed, total_rows, window_size, forecast_horizon)
        client_windows_by_mode = count_windows_by_mode(client_window_mask, bounds)
        for mode in ("train", "val", "test"):
            union_client_window_count_sum[mode] += client_windows_by_mode[mode]

        per_client_records.append(
            {
                "client_id": client_id,
                "exact_value_changed_rows": int(client_value_changed.sum()),
                "exact_final_changed_rows": int(client_final_changed.sum()),
                "exact_final_changed_fraction": float(client_final_changed.mean()) if total_rows else 0.0,
                "affected_train_windows": client_windows_by_mode["train"],
                "affected_val_windows": client_windows_by_mode["val"],
                "affected_test_windows": client_windows_by_mode["test"],
                "affected_total_windows": int(client_window_mask.sum()),
            }
        )

    per_client_df = pd.DataFrame(per_client_records).sort_values(
        ["exact_final_changed_rows", "affected_total_windows", "client_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    exact_stats = {
        "aligned_rows": total_rows,
        "clients_compared": int(len(common_clients)),
        "value_changed_cell_count": int(value_changed.sum()),
        "rows_with_any_value_change": int(value_changed.any(axis=1).sum()),
        "rows_with_structural_change": int(structural_changed.sum()),
        "restored_missing_timestamp_rows": int(missing_rows.sum()),
        "rows_with_duplicate_timestamps": int(duplicate_rows.sum()),
        "duplicate_timestamp_extra_rows_total": duplicate_extra_rows_total,
        "rows_with_any_final_change": int(overall_changed.sum()),
        "rows_with_any_final_change_fraction": float(overall_changed.mean()) if total_rows else 0.0,
        "window_size": int(window_size),
        "forecast_horizon": int(forecast_horizon),
        "total_windows": int(total_windows),
        "train_windows_total": int(bounds["train"][1] - bounds["train"][0]),
        "val_windows_total": int(bounds["val"][1] - bounds["val"][0]),
        "test_windows_total": int(bounds["test"][1] - bounds["test"][0]),
        "any_client_affected_train_windows": overall_windows_by_mode["train"],
        "any_client_affected_val_windows": overall_windows_by_mode["val"],
        "any_client_affected_test_windows": overall_windows_by_mode["test"],
        "any_client_affected_total_windows": int(overall_window_mask.sum()),
        "client_window_exposure_train_sum": int(union_client_window_count_sum["train"]),
        "client_window_exposure_val_sum": int(union_client_window_count_sum["val"]),
        "client_window_exposure_test_sum": int(union_client_window_count_sum["test"]),
        "client_window_exposure_total_sum": int(sum(union_client_window_count_sum.values())),
    }

    return {
        "stats": exact_stats,
        "per_client": per_client_df,
    }


def format_kv_lines(mapping: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in mapping.items():
        lines.append(f"- {key}: {value}")
    return lines


def format_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "(none)"
    view = df.head(max_rows) if max_rows is not None else df
    return view.to_string(index=False)


def build_text_report(
    report_summary: dict[str, Any],
    exact_summary: dict[str, Any] | None,
    *,
    top_clients: int,
) -> str:
    parts: list[str] = []

    parts.append("# Fault Injection Summary")
    parts.append("")
    parts.append("## Report Overview")
    parts.extend(format_kv_lines(report_summary["stats"]))
    parts.append("")

    parts.append("## Event Summary By Dimension")
    parts.append("```text")
    parts.append(format_table(report_summary["dimension_summary"]))
    parts.append("```")
    parts.append("")

    parts.append("## Event Summary By Fault Type")
    parts.append("```text")
    parts.append(format_table(report_summary["fault_type_summary"], max_rows=20))
    parts.append("```")
    parts.append("")

    parts.append("## Event Summary By Subcheck")
    parts.append("```text")
    parts.append(format_table(report_summary["subcheck_summary"], max_rows=20))
    parts.append("```")
    parts.append("")

    parts.append("## Event Summary By Scope")
    parts.append("```text")
    parts.append(format_table(report_summary["scope_summary"]))
    parts.append("```")
    parts.append("")

    if not report_summary["cluster_summary"].empty:
        parts.append("## Event Summary By Cluster")
        parts.append("```text")
        parts.append(format_table(report_summary["cluster_summary"]))
        parts.append("```")
        parts.append("")

    parts.append("## Top Clients By Event Impact")
    parts.append("```text")
    parts.append(format_table(report_summary["client_event_summary"], max_rows=top_clients))
    parts.append("```")
    parts.append("")

    if exact_summary is not None:
        parts.append("## Exact Final Corruption Against Clean/Faulty CSVs")
        parts.extend(format_kv_lines(exact_summary["stats"]))
        parts.append("")
        parts.append("## Top Clients By Exact Row/Window Impact")
        parts.append("```text")
        parts.append(format_table(exact_summary["per_client"], max_rows=top_clients))
        parts.append("```")
        parts.append("")

    return "\n".join(parts).strip() + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def main() -> int:
    args = parse_args()
    ratios_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratios_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    report_df = ensure_report_columns(pd.read_csv(args.report))
    report_summary = summarize_report(report_df)

    exact_summary: dict[str, Any] | None = None
    if (args.clean is None) ^ (args.faulty is None):
        raise ValueError("Provide both --clean and --faulty together for exact row/window analysis.")
    if args.clean is not None and args.faulty is not None:
        exact_summary = compute_exact_impact(
            clean_path=args.clean,
            faulty_path=args.faulty,
            selected_clients=report_summary["selected_clients"],
            timestamp_col=args.timestamp_col,
            window_size=args.window_size_steps,
            forecast_horizon=args.forecast_horizon,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        merged_client = report_summary["client_event_summary"].merge(
            exact_summary["per_client"],
            on="client_id",
            how="outer",
        ).sort_values(
            ["exact_final_changed_rows", "affected_rows_event_sum", "client_id"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        if args.per_client_output is not None:
            args.per_client_output.parent.mkdir(parents=True, exist_ok=True)
            merged_client.to_csv(args.per_client_output, index=False)
        exact_summary["merged_per_client"] = merged_client
    elif args.per_client_output is not None:
        args.per_client_output.parent.mkdir(parents=True, exist_ok=True)
        report_summary["client_event_summary"].to_csv(args.per_client_output, index=False)

    text_report = build_text_report(report_summary, exact_summary, top_clients=args.top_clients)
    print(text_report, end="")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text_report, encoding="utf-8")

    if args.json_output is not None:
        payload = {
            "report_stats": report_summary["stats"],
            "selected_clients": report_summary["selected_clients"],
            "dimension_summary": report_summary["dimension_summary"].to_dict(orient="records"),
            "fault_type_summary": report_summary["fault_type_summary"].to_dict(orient="records"),
            "subcheck_summary": report_summary["subcheck_summary"].to_dict(orient="records"),
            "scope_summary": report_summary["scope_summary"].to_dict(orient="records"),
            "cluster_summary": report_summary["cluster_summary"].to_dict(orient="records"),
            "client_event_summary": report_summary["client_event_summary"].to_dict(orient="records"),
        }
        if exact_summary is not None:
            payload["exact_stats"] = exact_summary["stats"]
            payload["exact_per_client"] = exact_summary["per_client"].to_dict(orient="records")
        write_json(args.json_output, payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
