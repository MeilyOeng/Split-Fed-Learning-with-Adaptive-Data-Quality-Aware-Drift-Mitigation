#!/usr/bin/env python3
"""Offline anchor-purity analysis; fault labels never enter runtime training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def normalize_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _column(fieldnames: Sequence[str] | None, *candidates: str) -> str:
    lookup = {str(name).strip().lower(): str(name) for name in (fieldnames or [])}
    for candidate in candidates:
        actual = lookup.get(candidate.lower())
        if actual is not None:
            return actual
    raise ValueError(
        f"Missing required column; expected one of {list(candidates)}, got {list(fieldnames or [])}."
    )


def load_cluster_assignments(path: Path) -> dict[str, set[str]]:
    clusters: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        client_column = _column(reader.fieldnames, "client_id", "id")
        cluster_column = _column(reader.fieldnames, "cluster_id", "cluster")
        for row in reader:
            client_id = normalize_id(row.get(client_column, ""))
            cluster_id = normalize_id(row.get(cluster_column, ""))
            if client_id and cluster_id:
                clusters[cluster_id].add(client_id)
    if not clusters:
        raise ValueError(f"No client-to-cluster assignments found in {path}.")
    return dict(clusters)


def load_faulty_client_ids(path: Path) -> set[str]:
    faulty: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        client_column = _column(reader.fieldnames, "client_id", "id")
        fault_column = None
        try:
            fault_column = _column(reader.fieldnames, "fault_type", "fault")
        except ValueError:
            pass
        for row in reader:
            client_id = normalize_id(row.get(client_column, ""))
            fault_name = str(row.get(fault_column, "injected") if fault_column else "injected").strip().lower()
            if client_id and fault_name not in {"", "none", "clean", "no_fault", "no-fault"}:
                faulty.add(client_id)
    return faulty


def load_anchor_rounds(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rounds_by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
                if record.get("event") != "quality_anchor_round":
                    continue
                record = dict(record)
                record["source_log"] = str(path)
                identity_values = (
                    record.get("cluster_id"),
                    record.get("scheduler_epoch"),
                    record.get("global_round"),
                    record.get("logical_round"),
                    record.get("round_id"),
                )
                identity = (
                    identity_values
                    if all(value is not None for value in identity_values)
                    else (str(path), line_number)
                )
                rounds_by_identity[identity] = record
    if not rounds_by_identity:
        raise ValueError("No quality_anchor_round events found in the supplied aggregation logs.")
    return list(rounds_by_identity.values())


def calculate_anchor_purity_rows(
    rounds: Sequence[Mapping[str, Any]],
    cluster_clients: Mapping[str, set[str]],
    faulty_clients: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in rounds:
        cluster_id = normalize_id(record.get("cluster_id", ""))
        if cluster_id not in cluster_clients:
            raise ValueError(
                f"Aggregation record references cluster {cluster_id!r}, which is absent from assignments."
            )
        assigned = set(cluster_clients[cluster_id])
        clean = assigned - faulty_clients
        anchors = {normalize_id(value) for value in record.get("anchor_client_ids", [])}
        anchors.discard("")
        unknown_anchors = sorted(anchors - assigned)
        if unknown_anchors:
            raise ValueError(
                f"Anchor IDs are absent from cluster {cluster_id} assignments: {unknown_anchors}."
            )
        if not anchors:
            raise ValueError("Anchor round contains no anchor_client_ids.")

        clean_anchors = anchors & clean
        anchor_purity = len(clean_anchors) / len(anchors)
        anchor_clean_recall = len(clean_anchors) / len(clean) if clean else math.nan
        overall_clean_proportion = len(clean) / len(assigned) if assigned else math.nan
        rows.append(
            {
                "source_log": record.get("source_log"),
                "aggregation_mode": record.get("aggregation_mode"),
                "cluster_id": cluster_id,
                "global_round": record.get("global_round"),
                "scheduler_epoch": record.get("scheduler_epoch"),
                "logical_round": record.get("logical_round"),
                "round_id": record.get("round_id"),
                "assigned_client_count": len(assigned),
                "faulty_client_count": len(assigned & faulty_clients),
                "clean_client_count": len(clean),
                "anchor_client_count": len(anchors),
                "clean_anchor_count": len(clean_anchors),
                "anchor_purity": anchor_purity,
                "anchor_clean_recall": anchor_clean_recall,
                "overall_clean_client_proportion": overall_clean_proportion,
                "anchor_purity_minus_overall_clean_proportion": (
                    anchor_purity - overall_clean_proportion
                ),
                "anchor_client_ids": ",".join(sorted(anchors)),
                "clean_anchor_client_ids": ",".join(sorted(clean_anchors)),
            }
        )
    return rows


def write_report(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    purities = [float(row["anchor_purity"]) for row in rows]
    recalls = [float(row["anchor_clean_recall"]) for row in rows]
    baselines = [float(row["overall_clean_client_proportion"]) for row in rows]
    finite_recalls = [value for value in recalls if math.isfinite(value)]
    return {
        "analysis_only": True,
        "round_count": len(rows),
        "mean_anchor_purity": sum(purities) / len(purities),
        "mean_anchor_clean_recall": (
            sum(finite_recalls) / len(finite_recalls) if finite_recalls else None
        ),
        "mean_overall_clean_client_proportion": sum(baselines) / len(baselines),
        "mean_purity_advantage": (
            sum(purity - baseline for purity, baseline in zip(purities, baselines)) / len(rows)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare offline anchor purity and clean recall against injected-fault labels."
    )
    parser.add_argument(
        "--aggregation-log",
        type=Path,
        nargs="+",
        required=True,
        help="One or more server quality_anchor_aggregation.jsonl files.",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("dataset/cluster-assignment.csv"),
        help="Client-to-cluster assignment CSV.",
    )
    parser.add_argument(
        "--fault-report",
        type=Path,
        required=True,
        help="Offline injected-fault report CSV containing client_id.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/anchor_purity_report.csv"),
        help="Destination CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assignments = load_cluster_assignments(args.assignments)
    faulty_clients = load_faulty_client_ids(args.fault_report)
    rounds = load_anchor_rounds(args.aggregation_log)
    rows = calculate_anchor_purity_rows(rounds, assignments, faulty_clients)
    write_report(rows, args.output)
    print(json.dumps({"output": str(args.output), **summarize(rows)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
