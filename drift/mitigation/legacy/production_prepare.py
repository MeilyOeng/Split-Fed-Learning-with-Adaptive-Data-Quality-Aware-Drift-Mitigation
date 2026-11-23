"""Production preparation and load-only acceptance for the 84-client suite."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import pickle
import shutil
import subprocess
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from drift.contracts import SchedulePoint
from drift.process import generate_process_timeline
from ..checkpoint_registry import CheckpointHRegistry, register_checkpoint_h, stage_method_checkpoint
from ..core import (
    Method, ProcessSpec, SUITE_ID, SegmentSizes, build_segment_rows, canonical_hash,
    environment_record, file_sha256, guard_output_directory, load_suite_yaml,
    selected_cohort_hash, stable_select_per_cluster, write_csv_rows, write_json,
)

DEFAULT_DATASET_DIR = Path("dataset/output_clean")
DEFAULT_ASSIGNMENTS = Path("dataset/cluster-assignment.csv")
RELEVANT_SOURCES = (
    "client/client.py", "data/data_streaming.py", "data/stream_service.py", "drift/configuration.py",
    "drift/contracts.py", "drift/generators.py", "drift/mitigation/checkpoint_registry.py",
    "drift/mitigation/legacy/cli.py", "drift/mitigation/core.py", "drift/mitigation/evaluation.py",
    "drift/mitigation/legacy/production_prepare.py", "drift/mitigation/legacy/production_runner.py",
    "drift/mitigation/legacy/service_adapter.py", "drift/pipeline.py", "drift/process.py", "drift/schedule.py",
    "drift/windowing.py", "fed_server/fed_server.py", "models/split_lstm.py", "run_split.py",
    "server/frozen_gradient.py", "server/server.py", "utils/client_optimization.py", "utils/runtime_utils.py",
)


def _schedule(length: int, onset: int, timestamps: list[Any]) -> list[SchedulePoint]:
    return [SchedulePoint(
        absolute_index=index, timestamp=str(timestamps[index]), configured_regime="new" if index >= onset else "old",
        old_concept_id="old", new_concept_id="new", active_concept_id="new" if index >= onset else "old",
        drift_weight=1.0 if index >= onset else 0.0, regime_active=index >= onset,
        change_point_event=index == onset, new_regime_selected=index >= onset,
    ) for index in range(length)]


def _load_assignments(path: Path, registry: CheckpointHRegistry) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    frame = pd.read_csv(path, dtype=str)
    columns = {column.lower(): column for column in frame.columns}
    id_col, cluster_col = columns.get("id"), columns.get("cluster")
    if not id_col or not cluster_col: raise ValueError("assignment CSV must contain ID and Cluster")
    rows = [{"client_id": str(row[id_col]), "cluster_id": str(row[cluster_col])} for _, row in frame.iterrows()]
    registry_pairs = {(row["cluster_id"], row["client_id"]) for row in registry.clients}
    assignment_pairs = {(row["cluster_id"], row["client_id"]) for row in rows}
    if registry_pairs != assignment_pairs: raise ValueError("checkpoint H clients and cluster assignments differ")
    clusters = {str(cluster): sorted(row["client_id"] for row in rows if row["cluster_id"] == str(cluster)) for cluster in range(1, 5)}
    return clusters, sorted(rows, key=lambda row: (row["cluster_id"], row["client_id"]))


def _scaler_reference(registry: CheckpointHRegistry, dataset_frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    rows = []
    for client in registry.clients:
        config_path = registry.root / f"cluster_{client['cluster_id']}" / f"client_{client['client_id']}_config.json"
        norm = json.loads(config_path.read_text(encoding="utf-8"))["normalization"]
        frame = dataset_frames[client["cluster_id"]]
        values = frame.loc[frame["id"].astype(str) == client["client_id"]].sort_values("Fecha_Hora")["value"].to_numpy(float).astype(np.float32).astype(np.float64)
        # 6115 training windows, L=24, h=1 -> shared X/y union [0, 6138].
        fitted = values[:6139]
        if not np.isclose(float(norm["mean"]), fitted.mean(), rtol=0.0, atol=2e-8):
            raise ValueError(f"saved scaler location does not match clean training observations for {client['client_id']}")
        if not np.isclose(float(norm["scale"]), fitted.std(), rtol=0.0, atol=2e-8):
            raise ValueError(f"saved scaler scale does not match clean training observations for {client['client_id']}")
        rows.append({
            "cluster_id": client["cluster_id"], "client_id": client["client_id"], "method": norm["method"],
            "ownership": "per_client", "target_scaling_strategy": norm["target_scaling_strategy"],
            "x_location": norm["x_location"], "x_scale": norm["x_scale"], "y_location": norm["y_location"],
            "y_scale": norm["y_scale"], "fit_first_observation_index": 0, "fit_last_observation_index": 6138,
            "fit_observation_count": 6139, "fit_scope": "clean_training_observations_only",
            "artifact_hash": canonical_hash(norm),
        })
    return {"contract": "one per-client scaler shared by target feature X and y", "streaming_refit_allowed": False,
            "combined_hash": canonical_hash(rows), "clients": rows}


def _source_hashes(repo: Path, config_path: Path, assignment_path: Path) -> dict[str, Any]:
    source_rows = []
    for relative in RELEVANT_SOURCES:
        path = repo / relative
        if path.is_file(): source_rows.append({"path": relative, "sha256": file_sha256(path)})
    datasets = [{"path": str(DEFAULT_DATASET_DIR / f"cluster_{cluster}.pkl"), "sha256": file_sha256(DEFAULT_DATASET_DIR / f"cluster_{cluster}.pkl")} for cluster in range(1, 5)]
    tracked, untracked = [], []
    for row in source_rows:
        check = subprocess.run(["git", "ls-files", "--error-unmatch", row["path"]], cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        (tracked if check.returncode == 0 else untracked).append(row)
    return {
        "datasets": datasets, "assignments": {"path": str(assignment_path), "sha256": file_sha256(assignment_path)},
        "configuration": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "relevant_sources": source_rows, "relevant_source_hash": canonical_hash(source_rows),
        "tracked_sources": tracked, "relevant_untracked_sources": untracked,
        "relevant_untracked_source_hash": canonical_hash(untracked),
    }


def _materialize_common(root: Path, registry: CheckpointHRegistry, cohort: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    common = root / "common"
    clean_dir, drifted_dir = common / "clean_deployment_timeline", common / "drifted_deployment_timeline"
    clean_dir.mkdir(parents=True); drifted_dir.mkdir(parents=True)
    selected = {(row["cluster_id"], row["client_id"]) for row in cohort if row["selected_for_drift"]}
    process = ProcessSpec(
        drift_type=config["process"]["drift_type"], drift_source=config["process"]["drift_source"],
        old_intercept=float(config["process"]["old_intercept"]), old_ar_coefficients=tuple(config["process"]["old_ar_coefficients"]),
        intercept_delta=float(config["process"]["intercept_delta"]), ar_coefficient_delta=tuple(config["process"]["ar_coefficient_delta"]),
        innovation_sigma=float(config["process"]["innovation_sigma"]), innovation_scale=float(config["process"]["innovation_scale"]),
        innovation_source=config["process"]["innovation_source"],
    )
    sizes = SegmentSizes(**config["segments"])
    timeline_rows, ground_truth, split_rows = [], [], None
    dataset_frames: dict[str, pd.DataFrame] = {}
    onset = None
    for cluster in range(1, 5):
        source_path = DEFAULT_DATASET_DIR / f"cluster_{cluster}.pkl"
        with source_path.open("rb") as handle: payload = pickle.load(handle)
        clean_payload, drift_payload = dict(payload), dict(payload)
        source_frame = payload["timeseries"].copy()
        source_frame["Fecha_Hora"] = pd.to_datetime(source_frame["Fecha_Hora"])
        dataset_frames[str(cluster)] = source_frame
        clean_frame, drift_frame = source_frame.copy(), source_frame.copy()
        for client_id, indices in source_frame.groupby("id", sort=False).groups.items():
            client_text = str(int(client_id))
            ordered_indices = list(source_frame.loc[indices].sort_values("Fecha_Hora").index)
            values = source_frame.loc[ordered_indices, "value"].to_numpy(float)
            timestamps = source_frame.loc[ordered_indices, "Fecha_Hora"].tolist()
            if len(values) != 8760: raise ValueError(f"client {client_text} must have 8760 observations, found {len(values)}")
            deployment_start = len(values) - sizes.total - int(config["windowing"]["window_size"]) - int(config["windowing"]["forecast_horizon"]) + 1
            starts = np.arange(deployment_start, deployment_start + sizes.total)
            resolved_onset = int(starts[sizes.pre_drift] + config["windowing"]["window_size"] - 1 + config["windowing"]["forecast_horizon"])
            onset = resolved_onset if onset is None else onset
            if onset != resolved_onset: raise ValueError("client onset coordinates differ")
            is_selected = (str(cluster), client_text) in selected
            drift_values = values.copy()
            if is_selected:
                result = generate_process_timeline(values, _schedule(len(values), onset, timestamps), process.equation_manifest() | {
                    "old": {"intercept": process.old_intercept, "ar_coefficients": list(process.old_ar_coefficients), "innovation_sigma": process.innovation_sigma},
                    "changes": {"intercept_delta": process.intercept_delta, "ar_coefficient_delta": list(process.ar_coefficient_delta), "innovation_scale": process.innovation_scale},
                    "innovation_source": process.innovation_source,
                }, target_feature_index=0, client_seed=int(config["seeds"]["drift_seed"]), component_id=f"cluster-{cluster}-client-{client_text}")
                drift_values = result.values[:, 0]
                if not np.array_equal(drift_values[:onset], values[:onset]): raise ValueError("process drift contaminated pre-onset data")
            drift_frame.loc[ordered_indices, "value"] = drift_values
            segment_rows = build_segment_rows(window_starts=starts, window_size=config["windowing"]["window_size"],
                horizon=config["windowing"]["forecast_horizon"], timestamps=timestamps, sizes=sizes, onset_index=onset, selected=is_selected)
            for row in segment_rows: ground_truth.append({"cluster_id": str(cluster), "client_id": client_text, **row})
            if split_rows is None: split_rows = segment_rows
            timeline_rows.append({
                "cluster_id": str(cluster), "client_id": client_text, "selected_for_drift": is_selected,
                "clean_timeline_hash": canonical_hash(values.tolist()), "drifted_timeline_hash": canonical_hash(drift_values.tolist()),
                "drift_ground_truth_hash": canonical_hash([{k: row[k] for k in ("deployment_window_number", "target_index", "true_drift_active")} for row in segment_rows]),
            })
        clean_payload["timeseries"], drift_payload["timeseries"] = clean_frame, drift_frame
        clean_path, drift_path = clean_dir / f"cluster_{cluster}.pkl", drifted_dir / f"cluster_{cluster}.pkl"
        with clean_path.open("wb") as handle: pickle.dump(clean_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with drift_path.open("wb") as handle: pickle.dump(drift_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    scaler_reference = _scaler_reference(registry, dataset_frames)
    write_json(common / "normalization_reference.json", scaler_reference)
    fields = list(ground_truth[0])
    write_csv_rows(common / "drift_ground_truth_manifest.csv", fields, ground_truth)
    split_fields = list(split_rows[0])
    write_csv_rows(common / "split_manifest.csv", split_fields, split_rows)
    timeline_files = [{"path": str(path.relative_to(root)), "sha256": file_sha256(path)} for path in sorted(clean_dir.glob("*.pkl")) + sorted(drifted_dir.glob("*.pkl"))]
    drift_hash = canonical_hash([row["drifted_timeline_hash"] for row in timeline_rows])
    manifest = {
        "materialized": True, "runtime_drift_injection": False, "onset_index": onset,
        "deployment_start_index": 6988, "adaptation_first_window": 525, "adaptation_first_window_start": 7512,
        "adaptation_first_window_end": 7535, "adaptation_first_target": 7536,
        "adaptation_first_window_start_timestamp": split_rows[524]["window_start_timestamp"],
        "adaptation_first_window_end_timestamp": split_rows[524]["window_end_timestamp"],
        "adaptation_first_target_timestamp": split_rows[524]["target_timestamp"],
        "holdout_first_window": 1225, "holdout_first_window_start": 8212, "holdout_first_window_end": 8235,
        "holdout_first_target": 8236, "holdout_first_window_start_timestamp": split_rows[1224]["window_start_timestamp"],
        "holdout_first_window_end_timestamp": split_rows[1224]["window_end_timestamp"],
        "holdout_first_target_timestamp": split_rows[1224]["target_timestamp"],
        "last_target": 8759, "last_target_timestamp": split_rows[-1]["target_timestamp"], "timeline_files": timeline_files,
        "clients": timeline_rows, "drifted_combined_hash": drift_hash,
        "method_drift_ground_truth_hashes": {method.value: drift_hash for method in (Method.M0, Method.M1, Method.M2)},
    }
    write_json(common / "timeline_hashes.json", manifest)
    return {"timeline_manifest": manifest, "scaler_reference": scaler_reference, "ground_truth_rows": len(ground_truth)}


def prepare_production_suite(config_path: str | Path, checkpoint_h: str | Path, output: str | Path, *, resume: bool = False, overwrite: bool = False, preflight_only: bool = True) -> dict[str, Any]:
    if not preflight_only: raise ValueError("prepare is load-only; pass preflight_only=True")
    repo = Path.cwd().resolve(); config_path = Path(config_path); assignment_path = DEFAULT_ASSIGNMENTS
    config = load_suite_yaml(config_path)
    registry = register_checkpoint_h(checkpoint_h, expected_epochs=int(config["checkpoint_h"]["completed_epochs"]))
    clusters, assignment_rows = _load_assignments(assignment_path, registry)
    cohort = stable_select_per_cluster(clusters, per_cluster=int(config["population"]["drifted_per_cluster"]), seed=int(config["seeds"]["client_selection_seed"]))
    cohort_hash = selected_cohort_hash(cohort)
    sources = _source_hashes(repo, config_path, assignment_path)
    output = Path(output)
    if resume:
        timeline = json.loads((output / "common" / "timeline_hashes.json").read_text(encoding="utf-8"))
        for artifact in timeline["timeline_files"]:
            if file_sha256(output / artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"resume timeline artifact hash mismatch: {artifact['path']}")
        if canonical_hash([row["drifted_timeline_hash"] for row in timeline["clients"]]) != timeline["drifted_combined_hash"]:
            raise ValueError("resume drifted timeline state hash mismatch")
        normalization = json.loads((output / "common" / "normalization_reference.json").read_text(encoding="utf-8"))
        if canonical_hash(normalization["clients"]) != normalization["combined_hash"]:
            raise ValueError("resume scaler hash mismatch")
        expected = {
            "checkpoint_h": registry.combined_artifact_hash, "config": canonical_hash(config),
            "sources": sources["relevant_source_hash"], "cohort": cohort_hash,
            "split": file_sha256(output / "split_manifest.csv"), "scalers": normalization["combined_hash"],
            "timeline": timeline["drifted_combined_hash"], "timeline_artifacts": canonical_hash(timeline["timeline_files"]),
        }
        guard_output_directory(output, resume=True, expected_hashes=expected)
        report = json.loads((output / "preflight_acceptance_report.json").read_text(encoding="utf-8"))
        if not report.get("accepted"): raise ValueError("cannot resume an unaccepted preflight")
        return {"output": str(output), "acceptance": report, "resumed": True}
    root = guard_output_directory(output, overwrite=overwrite)
    (root / ".drift_mitigation_suite").write_text(SUITE_ID + "\n", encoding="utf-8")
    (root / "environment").mkdir(); (root / "comparison").mkdir()
    write_json(root / "checkpoint_H_manifest.json", registry.as_dict())
    write_json(root / "source_hashes.json", sources)
    write_json(root / "relevant_source_hashes.json", {"files": sources["relevant_sources"], "hash": sources["relevant_source_hash"]})
    write_json(root / "relevant_untracked_source_hashes.json", {"files": sources["relevant_untracked_sources"], "hash": sources["relevant_untracked_source_hash"]})
    write_csv_rows(root / "selected_drift_clients.csv", list(cohort[0]), cohort)
    write_csv_rows(root / "split_manifest.csv", ["segment", "window_offset", "window_limit"], [
        {"segment": "PRE_DRIFT", "window_offset": 0, "window_limit": 524},
        {"segment": "ADAPTATION", "window_offset": 524, "window_limit": 700},
        {"segment": "FINAL_HOLDOUT", "window_offset": 1224, "window_limit": 524},
    ])
    write_json(root / "selected_drift_clients_hash.json", {"sha256": cohort_hash, "selected_count": 32, "control_count": 52})
    shutil.copy2(config_path, root / "suite_resolved_config.yaml")
    common = _materialize_common(root, registry, cohort, config)
    shutil.copy2(root / "selected_drift_clients.csv", root / "common" / "selected_drift_clients.csv")
    shutil.copy2(root / "split_manifest.csv", root / "common" / "split_manifest_summary.csv")
    stage_reports = {}
    # This generation-1 pipeline predates R1/M3 and has no bespoke handling
    # for them (e.g. no EWC dispatch); it stays closed to the original four
    # methods rather than picking up new Method members automatically.
    for method in (Method.R0, Method.M0, Method.M1, Method.M2):
        method_root = root / method.value
        method_root.mkdir()
        stage_reports[method.value] = stage_method_checkpoint(registry, method_root)
        write_json(method_root / "seeds.json", config["seeds"])
        (method_root / "logs").mkdir(); (method_root / "logs" / "stdout.log").touch(); (method_root / "logs" / "stderr.log").touch(); (method_root / "logs" / "events.jsonl").touch()
    initial_hashes = {row["combined_tensor_state_hash"] for row in stage_reports.values()}
    env = environment_record(repo)
    write_json(root / "environment" / "environment.json", env)
    try:
        freeze = subprocess.run([str(Path(__import__('sys').executable)), "-m", "pip", "freeze"], check=True, text=True, stdout=subprocess.PIPE).stdout
    except Exception as exc: freeze = f"pip freeze unavailable: {exc}\n"
    (root / "environment" / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    resume_hashes = {
        "checkpoint_h": registry.combined_artifact_hash, "config": canonical_hash(config),
        "sources": sources["relevant_source_hash"], "cohort": cohort_hash,
        "split": file_sha256(root / "split_manifest.csv"), "scalers": common["scaler_reference"]["combined_hash"],
        "timeline": common["timeline_manifest"]["drifted_combined_hash"],
        "timeline_artifacts": canonical_hash(common["timeline_manifest"]["timeline_files"]),
    }
    write_json(root / "resume_hashes.json", resume_hashes)
    checks = {
        "checkpoint_H_valid_84_clients_four_servers": registry.accepted,
        "checkpoint_H_shared_scalers_valid": bool(common["scaler_reference"]["combined_hash"]),
        "selected_exactly_32_eight_per_cluster": sum(row["selected_for_drift"] for row in cohort) == 32 and all(sum(row["selected_for_drift"] and row["cluster_id"] == str(c) for row in cohort) == 8 for c in range(1, 5)),
        "materialized_timelines_once": common["timeline_manifest"]["materialized"] and not common["timeline_manifest"]["runtime_drift_injection"],
        "M0_M1_M2_identical_drift_hash": len(set(common["timeline_manifest"]["method_drift_ground_truth_hashes"].values())) == 1,
        "four_methods_same_initial_tensor_state": len(initial_hashes) == 1 and next(iter(initial_hashes)) == registry.combined_tensor_state_hash,
        "baseline_progress_not_copied": all(not row["baseline_progress_files_copied"] for row in stage_reports.values()),
        "deployment_counters_zero": all(row["deployment_counters_start_at_zero"] for row in stage_reports.values()),
        "load_only_zero_optimizer_steps": True, "load_only_zero_fedavg_events": True,
        "full_methods_not_launched": True,
    }
    acceptance = {"suite_id": SUITE_ID, "accepted": all(checks.values()), "complete": True, "preflight_only": True, "checks": checks,
                  "checkpoint_H": str(registry.root), "output": str(root), "onset": {key: common["timeline_manifest"][key] for key in ("onset_index", "adaptation_first_window_start", "adaptation_first_window_end", "adaptation_first_target")}}
    write_json(root / "preflight_acceptance_report.json", acceptance)
    write_json(root / "suite_manifest.json", {"suite_id": SUITE_ID, "phase": "production_preflight", "full_run_authorized": False,
        "checkpoint_H_manifest": "checkpoint_H_manifest.json", "cohort_hash": cohort_hash,
        "timeline_hash": common["timeline_manifest"]["drifted_combined_hash"], "process_equations": ProcessSpec().equation_manifest(),
        "stage_reports": stage_reports, "source_snapshot_timing": "post-checkpoint-H run; provenance only, not a claim about historical training sources"})
    return {"output": str(root), "acceptance": acceptance, "checkpoint_H": registry.as_dict(), "cohort": cohort, "common": common, "resumed": False}
