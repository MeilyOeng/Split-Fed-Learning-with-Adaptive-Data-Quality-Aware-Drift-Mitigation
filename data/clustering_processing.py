 #!/usr/bin/env python3
"""Preprocess cluster data into pickle payloads with metadata features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

DQ_METADATA_PREFIX = "__dq_"


DEFAULT_DATASET_PATH = Path("dataset") / "dataset.csv"
DEFAULT_ASSIGNMENT_PATH = Path("dataset") / "cluster-assignment.csv"
DEFAULT_METADATA_PATH = Path("dataset") / "general_categories.csv"
DEFAULT_OUTPUT_DIR = Path("dataset") / "output"


def log(message: str) -> None:
    print(message, flush=True)


def normalize_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text or "field"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cluster pickle files with metadata features.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Wide demand CSV path.")
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENT_PATH, help="Cluster assignment CSV path.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH, help="Client metadata CSV path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for cluster pickle outputs.")
    parser.add_argument("--timestamp-col", default="Fecha_Hora", help="Timestamp column name.")
    return parser.parse_args()


def _normalize_group_text(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("UNKNOWN").astype(str).str.strip()
    return cleaned.replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})


def _prepare_metadata(metadata_path: Path) -> tuple[pd.DataFrame, list[str], dict]:
    if not metadata_path.exists():
        log(f"Metadata file not found at {metadata_path}; continuing without metadata features.")
        return pd.DataFrame(columns=["id"]), [], {"metadata_found": False, "static_feature_columns": [], "input_size": 1}

    metadata = pd.read_csv(metadata_path).copy()
    if "ID" not in metadata.columns:
        raise ValueError(f"Metadata file must contain an ID column: {metadata_path}")

    metadata["id"] = metadata["ID"].map(normalize_id)
    metadata = metadata.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    raw_feature_frame = pd.DataFrame({"id": metadata["id"]})
    numeric_feature_columns: list[str] = []
    categorical_feature_columns: list[str] = []
    numeric_stats: dict[str, dict[str, float]] = {}

    metadata_cols = [col for col in metadata.columns if col not in {"ID", "id"}]
    for col in metadata_cols:
        if pd.api.types.is_numeric_dtype(metadata[col]):
            numeric = pd.to_numeric(metadata[col], errors="coerce")
            fill_value = float(numeric.median()) if numeric.notna().any() else 0.0
            filled = numeric.fillna(fill_value).astype(float)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            if std == 0.0:
                std = 1.0
            feature_col = f"meta_num_{slugify(col)}"
            raw_feature_frame[feature_col] = (filled - mean) / std
            numeric_feature_columns.append(feature_col)
            numeric_stats[col] = {"fill": fill_value, "mean": mean, "std": std}
        else:
            cleaned = _normalize_group_text(metadata[col])
            dummies = pd.get_dummies(cleaned, prefix=f"meta_cat_{slugify(col)}", dtype=float)
            dummies = dummies.rename(columns={name: slugify(name) for name in dummies.columns})
            raw_feature_frame = pd.concat([raw_feature_frame, dummies], axis=1)
            categorical_feature_columns.extend(dummies.columns.tolist())

    metadata_export = metadata.copy()
    for col in metadata_cols:
        if not pd.api.types.is_numeric_dtype(metadata_export[col]):
            metadata_export[col] = _normalize_group_text(metadata_export[col])
    metadata_export = metadata_export.rename(columns={"ID": "source_id"})
    metadata_export = metadata_export.merge(raw_feature_frame, on="id", how="left")

    static_feature_columns = numeric_feature_columns + categorical_feature_columns
    manifest = {
        "metadata_found": True,
        "raw_metadata_columns": metadata_cols,
        "numeric_feature_columns": numeric_feature_columns,
        "categorical_feature_columns": categorical_feature_columns,
        "static_feature_columns": static_feature_columns,
        "numeric_feature_stats": numeric_stats,
        "input_feature_columns": ["value"] + static_feature_columns,
        "input_size": 1 + len(static_feature_columns),
    }
    return metadata_export, static_feature_columns, manifest


def _iter_client_columns(frame: pd.DataFrame, timestamp_col: str) -> Iterable[str]:
    for col in frame.columns:
        if col != timestamp_col and not str(col).startswith(DQ_METADATA_PREFIX):
            yield str(col)


def preprocess_clusters(
    dataset_path: str | Path,
    assignment_path: str | Path,
    output_dir: str | Path,
    *,
    metadata_path: str | Path | None = DEFAULT_METADATA_PATH,
    timestamp_col: str = "Fecha_Hora",
) -> dict:
    dataset_path = Path(dataset_path)
    assignment_path = Path(assignment_path)
    output_dir = Path(output_dir)
    metadata_path = None if metadata_path is None else Path(metadata_path)

    log(f"Reading dataset from {dataset_path}")
    dataset = pd.read_csv(dataset_path, low_memory=False)
    if timestamp_col not in dataset.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in {dataset_path}")
    dataset[timestamp_col] = pd.to_datetime(dataset[timestamp_col], errors="coerce")
    dataset = dataset.dropna(subset=[timestamp_col]).sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)

    client_cols = list(_iter_client_columns(dataset, timestamp_col))
    rename_map = {col: normalize_id(col) for col in client_cols}
    dataset = dataset.rename(columns=rename_map)
    client_cols = [rename_map[col] for col in client_cols]

    for client_id in client_cols:
        dataset[client_id] = pd.to_numeric(dataset[client_id], errors="coerce")

    log(f"Reading cluster assignments from {assignment_path}")
    assignments = pd.read_csv(assignment_path).copy()
    required_assignment_cols = {"ID", "Cluster"}
    if not required_assignment_cols.issubset(assignments.columns):
        missing = sorted(required_assignment_cols - set(assignments.columns))
        raise ValueError(f"Assignment file is missing columns: {missing}")
    assignments["id"] = assignments["ID"].map(normalize_id)
    assignments = assignments.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    assignments["Cluster"] = assignments["Cluster"].astype(str)

    metadata_export, static_feature_columns, manifest = _prepare_metadata(metadata_path) if metadata_path else (
        pd.DataFrame({"id": assignments["id"]}),
        [],
        {"metadata_found": False, "static_feature_columns": [], "input_feature_columns": ["value"], "input_size": 1},
    )

    metadata_export = assignments[["id", "Cluster"]].merge(metadata_export, on="id", how="left")
    for feature_col in static_feature_columns:
        if feature_col not in metadata_export.columns:
            metadata_export[feature_col] = 0.0
    metadata_export[static_feature_columns] = metadata_export[static_feature_columns].fillna(0.0)

    available_ids = set(client_cols)
    matched_assignments = assignments.loc[assignments["id"].isin(available_ids)].copy()
    missing_from_dataset = sorted(set(assignments["id"]) - available_ids)
    if matched_assignments.empty:
        raise ValueError("No cluster assignment IDs matched the dataset columns.")

    ensure_dir(output_dir)

    cluster_summaries: list[dict[str, object]] = []
    for cluster_id, cluster_rows in matched_assignments.groupby("Cluster", sort=True):
        cluster_client_ids = [cid for cid in cluster_rows["id"].tolist() if cid in dataset.columns]
        if not cluster_client_ids:
            continue
        log(f"Building cluster {cluster_id} with {len(cluster_client_ids)} clients")

        cluster_wide = dataset[[timestamp_col] + cluster_client_ids].copy()
        cluster_long = cluster_wide.melt(id_vars=[timestamp_col], var_name="id", value_name="value")
        cluster_long["id"] = cluster_long["id"].map(normalize_id)
        cluster_long["value"] = pd.to_numeric(cluster_long["value"], errors="coerce")
        cluster_long["id"] = cluster_long["id"].astype(int)

        cluster_metadata = metadata_export.loc[metadata_export["Cluster"] == cluster_id].copy()
        cluster_metadata["id"] = cluster_metadata["id"].astype(int)

        output_path = output_dir / f"cluster_{cluster_id}.pkl"
        payload = {
            "timeseries": cluster_long,
            "client_metadata": cluster_metadata,
            "timestamp_col": timestamp_col,
            "value_col": "value",
            "static_feature_columns": static_feature_columns,
            "input_feature_columns": ["value"] + static_feature_columns,
            "input_size": 1 + len(static_feature_columns),
        }
        pd.to_pickle(payload, output_path)
        cluster_summaries.append(
            {
                "cluster_id": cluster_id,
                "client_count": int(len(cluster_client_ids)),
                "rows": int(len(cluster_long)),
                "output_path": str(output_path),
            }
        )
        log(f"Saved cluster {cluster_id} to {output_path}")

    feature_manifest = {
        "dataset_path": str(dataset_path),
        "assignment_path": str(assignment_path),
        "metadata_path": str(metadata_path) if metadata_path else None,
        "timestamp_col": timestamp_col,
        "num_rows": int(len(dataset)),
        "num_clients": int(len(client_cols)),
        "matched_assignment_clients": int(len(matched_assignments)),
        "assignment_clients_missing_from_dataset": int(len(missing_from_dataset)),
        "assignment_client_ids_missing_from_dataset": missing_from_dataset[:50],
        "clusters": cluster_summaries,
    }
    feature_manifest.update(manifest)

    manifest_path = output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    log(f"Saved feature manifest to {manifest_path}")
    log("Preprocessing complete")
    return feature_manifest


if __name__ == "__main__":
    args = parse_args()
    preprocess_clusters(
        dataset_path=args.dataset,
        assignment_path=args.assignments,
        output_dir=args.output_dir,
        metadata_path=args.metadata,
        timestamp_col=args.timestamp_col,
    )
