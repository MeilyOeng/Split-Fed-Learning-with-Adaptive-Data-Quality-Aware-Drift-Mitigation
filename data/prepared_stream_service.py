#!/usr/bin/env python3
"""HTTP stream over immutable offline-prepared X/y samples only."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
import torch
from flask import Flask, Response, jsonify, request

try:
    from data.prepared_window_store import PreparedClientSamples, PreparedWindowStore
except ModuleNotFoundError:  # launched as ``python data/prepared_stream_service.py``
    from prepared_window_store import PreparedClientSamples, PreparedWindowStore

try:
    from utils.runtime_utils import build_quality_anchor_config, derive_dq_mode, parse_bool, quality_scores_needed
except ModuleNotFoundError:  # launched as ``python data/prepared_stream_service.py``
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.runtime_utils import build_quality_anchor_config, derive_dq_mode, parse_bool, quality_scores_needed


app = Flask(__name__)
PREPARATION_ROOT = Path(os.environ.get("PREPARATION_ROOT", "")).resolve()
STREAM_CLUSTER_ID = str(os.environ.get("STREAM_CLUSTER_ID", ""))
STREAM_PORT = int(os.environ.get("STREAM_PORT", "5001"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
ROLE_MAP = json.loads(os.environ.get("PREPARED_ROLE_MAP", "{}"))
ALLOWED_ROLES = tuple(item.strip() for item in os.environ.get("PREPARED_ALLOWED_ROLES", "").split(",") if item.strip())
EXPECTED_COORDINATE_HASH = str(os.environ.get("PREPARED_EXPECTED_COORDINATE_HASH", "")).strip()
EXPECTED_DATASET_HASHES = json.loads(os.environ.get("PREPARED_EXPECTED_DATASET_HASHES", "{}"))
EXPECTED_SCALER_HASH = str(os.environ.get("PREPARED_EXPECTED_SCALER_HASH", "")).strip()

# Derived identically to client.py's own ClusteredClientRunner.__init__, from
# the SAME env vars runner.py's build_method_environment sets for this whole
# phase subprocess tree -- never independently guessed -- so client.py's own
# dq_mode cross-check against /stream/norm_stats can never mismatch.
_USE_QUALITY_AS_INPUT = parse_bool(os.environ.get("USE_QUALITY_AS_INPUT", "false"))
_USE_QUALITY_WEIGHTS = parse_bool(os.environ.get("USE_QUALITY_WEIGHTS", "false"))
_USE_QUALITY_AGGREGATION_WEIGHTS = parse_bool(os.environ.get("USE_QUALITY_AGGREGATION_WEIGHTS", "false"))
_QUALITY_ANCHOR_CONFIG = build_quality_anchor_config(os.environ)
_DQ_MODE = derive_dq_mode(
    _USE_QUALITY_WEIGHTS, _USE_QUALITY_AS_INPUT, _USE_QUALITY_AGGREGATION_WEIGHTS,
    _QUALITY_ANCHOR_CONFIG.use_quality_anchor_aggregation, _QUALITY_ANCHOR_CONFIG.use_gradient_alignment_filter,
    _QUALITY_ANCHOR_CONFIG.use_high_quality_validation_guardrail,
)
_QUALITY_NEEDED = quality_scores_needed(
    _USE_QUALITY_WEIGHTS, _USE_QUALITY_AS_INPUT, _USE_QUALITY_AGGREGATION_WEIGHTS,
    _QUALITY_ANCHOR_CONFIG.use_quality_anchor_aggregation,
)


def _enabled(name: str) -> bool:
    return str(os.environ.get(name, "false")).strip().lower() in {"1", "true", "yes", "y"}


def _validate_fail_closed_environment() -> None:
    if not _enabled("PREPARED_DATASET_BACKEND"):
        raise ValueError("prepared stream service requires PREPARED_DATASET_BACKEND=true")
    forbidden_true = [name for name in (
        "NORMALIZE", "SHARED_X_Y_SCALER", "DRIFT_INJECTION_ENABLED",
        "RUNTIME_WINDOW_RECONSTRUCTION", "RUNTIME_TARGET_RECONSTRUCTION", "RUNTIME_SPLIT_RECALCULATION",
    ) if _enabled(name)]
    if forbidden_true:
        raise ValueError(f"prepared backend rejects enabled runtime transform(s): {forbidden_true}")
    if str(os.environ.get("DRIFT_CONFIG_PATH", "")).strip():
        raise ValueError("prepared backend rejects non-empty DRIFT_CONFIG_PATH")
    if not PREPARATION_ROOT.is_dir():
        raise FileNotFoundError(f"PREPARATION_ROOT does not exist: {PREPARATION_ROOT}")
    if not ALLOWED_ROLES or not ROLE_MAP:
        raise ValueError("prepared backend requires explicit allowed roles and mode-to-role mapping")
    if any(role not in ALLOWED_ROLES for role in ROLE_MAP.values()):
        raise ValueError("prepared role map contains a role outside PREPARED_ALLOWED_ROLES")
    if not EXPECTED_SCALER_HASH:
        raise ValueError("prepared backend requires PREPARED_EXPECTED_SCALER_HASH")


_validate_fail_closed_environment()
store = PreparedWindowStore(
    PREPARATION_ROOT,
    allowed_roles=ALLOWED_ROLES,
    expected_coordinate_manifest_hash=EXPECTED_COORDINATE_HASH or None,
)
if str(store.manifest.get("scaler_manifest_hash", "")) != EXPECTED_SCALER_HASH:
    raise ValueError("prepared scaler-manifest hash mismatch")
for role in set(ROLE_MAP.values()):
    expected = EXPECTED_DATASET_HASHES.get(role)
    actual = store.manifest["roles"][role]["combined_payload_hash"]
    if not expected or str(expected) != str(actual):
        raise ValueError(f"prepared combined dataset hash mismatch for role={role}")

scaler_manifest = json.loads((PREPARATION_ROOT / "manifests" / "scaler_parameters.json").read_text(encoding="utf-8"))
scalers = {(str(row["cluster_id"]), str(row["client_id"])): row for row in scaler_manifest["clients"]}
cache: dict[tuple[str, str, str], PreparedClientSamples] = {}
streams: dict[tuple[str, str, str], tuple[np.ndarray, int]] = {}
lock = Lock()


def _role(mode: str) -> str:
    try:
        return str(ROLE_MAP[str(mode)])
    except KeyError as exc:
        raise PermissionError(f"stream mode is not approved for this prepared process: {mode}") from exc


def _samples(mode: str, cluster_id: str, client_id: str) -> PreparedClientSamples:
    role = _role(mode)
    key = (role, str(cluster_id), str(client_id))
    if STREAM_CLUSTER_ID and str(cluster_id) != STREAM_CLUSTER_ID:
        raise KeyError(f"cluster {cluster_id} is not loaded by this stream process")
    if key not in cache:
        cache[key] = store.load_client(*key)
    return cache[key]


def _positions(mode: str, samples: PreparedClientSamples) -> np.ndarray:
    mode = str(mode)
    if mode in {"train", "val", "test"}:
        return np.arange(samples.size, dtype=np.int64)
    segments = np.asarray(samples.metadata["segment"]).astype(str)
    requested = {
        "deployment_pre_drift": "PRE_DRIFT",
        "deployment_adaptation": "ADAPTATION",
        "deployment_final_holdout": "FINAL_HOLDOUT",
        "deployment_counterfactual_clean_holdout": "FINAL_HOLDOUT",
    }.get(mode)
    if requested is None:
        raise PermissionError(f"unsupported prepared deployment mode: {mode}")
    result = np.flatnonzero(segments == requested).astype(np.int64)
    if result.size == 0:
        raise ValueError(f"prepared role contains no samples for segment={requested}")
    return result


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy", "backend": "prepared_fail_closed", "cluster_id": STREAM_CLUSTER_ID,
        "allowed_roles": sorted(ALLOWED_ROLES), "role_map": ROLE_MAP,
        "runtime_normalization": False, "runtime_scaler_fitting": False,
        "runtime_drift_generation": False, "runtime_window_construction": False,
        "runtime_target_construction": False, "runtime_split_calculation": False,
        "coordinate_manifest_hash": store.coordinate_manifest_hash,
    })


@app.get("/stream/norm_stats")
def norm_stats():
    cluster_id = str(request.args["cluster_id"])
    client_id = str(request.args["client_id"])
    try:
        row = scalers[(cluster_id, client_id)]
    except KeyError:
        return jsonify({"error": "unknown prepared scaler"}), 404
    # `normalize=true` here tells the client how to invert predictions. It does
    # not authorize normalization in this service; X/y are already normalized.
    return jsonify({
        "cluster_id": cluster_id, "client_id": client_id, "normalize": True,
        "method": row["method"], "mean": row["location"], "scale": row["scale"],
        "x_location": [row["location"]], "x_scale": [row["scale"]],
        "y_location": row["location"], "y_scale": row["scale"],
        "target_scaling_strategy": "shared", "input_size": 1, "base_input_size": 1,
        "stream_input_size": 1, "model_input_size": 1,
        "num_train_windows": 6115 if "clean_train" in ALLOWED_ROLES else 0,
        "num_val_windows": 873 if "clean_validation" in ALLOWED_ROLES else 0,
        "num_test_windows": 1748 if any(role.endswith("deployment") for role in ALLOWED_ROLES) else 0,
        "representation": "normalized", "normalization_applied_count": 1,
        "scaler_hash": row["scaler_hash"], "coordinate_manifest_hash": store.coordinate_manifest_hash,
        "dq_mode": _DQ_MODE, "quality_needed": _QUALITY_NEEDED, "quality_input_enabled": _USE_QUALITY_AS_INPUT,
        "quality_weights_enabled": _USE_QUALITY_WEIGHTS, "use_static_features": False,
    })


@app.post("/stream/reset")
def reset():
    body = request.get_json() or {}
    cluster_id, client_id = str(body["cluster_id"]), str(body["client_id"])
    mode = str(body.get("mode", "train"))
    try:
        samples = _samples(mode, cluster_id, client_id)
        positions = _positions(mode, samples)
    except (PermissionError, KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    requested_offset = int(body.get("window_offset", 0) or 0)
    requested_limit = body.get("window_limit")
    # Deployment offsets are accepted only as assertions against the stored
    # segment coordinates; they never determine the segment.
    if mode.startswith("deployment_"):
        stored_first = int(positions[0]) if samples.role in {"clean_deployment", "drifted_deployment"} else 0
        if requested_offset not in {0, stored_first}:
            return jsonify({"error": "runtime offset disagrees with stored prepared segment"}), 400
    else:
        if requested_offset < 0 or requested_offset > len(positions):
            return jsonify({"error": "prepared stream offset is outside the approved role"}), 400
        positions = positions[requested_offset:]
    if requested_limit is not None:
        requested_limit = int(requested_limit)
        if requested_limit != len(positions):
            return jsonify({"error": "runtime window_limit disagrees with stored prepared role/segment"}), 400
    key = (cluster_id, client_id, mode)
    with lock:
        streams[key] = (positions, 0)
    return jsonify({
        "status": "reset", "cluster_id": cluster_id, "client_id": client_id, "mode": mode,
        "dataset_role": samples.role, "windows_available": len(positions), "representation": "normalized",
        "normalization_applied_count": 1, "runtime_processing": False,
    })


@app.get("/stream/status")
def status():
    cluster_id, client_id = str(request.args["cluster_id"]), str(request.args["client_id"])
    mode = str(request.args.get("mode", "train"))
    key = (cluster_id, client_id, mode)
    if key not in streams:
        return jsonify({"error": "prepared stream has not been reset"}), 400
    positions, cursor = streams[key]
    return jsonify({"cluster_id": cluster_id, "client_id": client_id, "mode": mode, "current_offset": cursor, "remaining_windows": len(positions) - cursor})


@app.post("/stream/next")
def next_batch():
    body = request.get_json() or {}
    cluster_id, client_id = str(body["cluster_id"]), str(body["client_id"])
    mode, batch_size = str(body.get("mode", "train")), int(body.get("batch_size", BATCH_SIZE))
    key = (cluster_id, client_id, mode)
    with lock:
        if key not in streams:
            return jsonify({"error": "prepared stream has not been reset"}), 400
        positions, cursor = streams[key]
        if cursor >= len(positions):
            return Response(status=204)
        end = min(cursor + batch_size, len(positions))
        batch_positions = positions[cursor:end]
        streams[key] = (positions, end)
    try:
        samples = _samples(mode, cluster_id, client_id)
    except (PermissionError, KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    metadata = samples.metadata
    return jsonify({
        "cluster_id": cluster_id, "client_id": client_id, "mode": mode,
        "X": samples.X[batch_positions].tolist(), "y": samples.y[batch_positions].tolist(),
        "window_start_indices": metadata["window_start_index"][batch_positions].tolist(),
        "window_end_indices": metadata["window_end_index"][batch_positions].tolist(),
        "target_indices": metadata["target_index"][batch_positions].tolist(),
        "window_start_timestamps": metadata["window_start_timestamp_ns"][batch_positions].tolist(),
        "window_end_timestamps": metadata["window_end_timestamp_ns"][batch_positions].tolist(),
        "target_timestamps": metadata["target_timestamp_ns"][batch_positions].tolist(),
        "dataset_role": samples.role, "representation": "normalized",
        "normalization_applied_count": 1, "runtime_processing": False,
        "quality_scores": None,
        "quality_weight_scores": (
            metadata["quality_weight_score"][batch_positions].tolist()
            if "quality_weight_score" in metadata else None
        ),
        "quality_anchor_scores": None,
        "quality_score_metadata": (
            {"granularity": "window", "source": "offline_prepared_dq_scoring"}
            if "quality_weight_score" in metadata else {}
        ),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=STREAM_PORT, debug=False, threaded=True)
