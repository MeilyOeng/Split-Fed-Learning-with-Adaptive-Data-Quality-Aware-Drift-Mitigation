"""Registration, validation, hashing, and isolated staging for checkpoint H."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping
import torch

from .core import canonical_hash, file_sha256, tensor_mapping_hash, write_json

_CLIENT_RE = re.compile(r"client_(?P<client_id>\d+)_final\.pt$")
_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")

@dataclass(frozen=True)
class CheckpointHRegistry:
    root: Path
    clients: tuple[dict[str, Any], ...]
    servers: tuple[dict[str, Any], ...]
    global_server: dict[str, Any]
    combined_tensor_state_hash: str
    combined_artifact_hash: str
    normalization_hash: str
    accepted: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root), "clients": list(self.clients), "servers": list(self.servers),
            "global_server": self.global_server, "combined_tensor_state_hash": self.combined_tensor_state_hash,
            "combined_artifact_hash": self.combined_artifact_hash, "normalization_hash": self.normalization_hash,
            "accepted": self.accepted,
        }

def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise ValueError(f"checkpoint payload is missing model_state_dict: {path}")
    return dict(payload)

def _hash_nested(value: Any) -> str:
    digest = hashlib.sha256()
    def update(prefix: str, item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(prefix.encode() + b"\0")
            digest.update(str(tuple(tensor.shape)).encode() + b"\0")
            digest.update(str(tensor.dtype).encode() + b"\0")
            digest.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str): update(f"{prefix}/{key}", item[key])
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item): update(f"{prefix}/{index}", nested)
        else:
            digest.update(prefix.encode() + b"\0")
            digest.update(json.dumps(item, sort_keys=True, default=str).encode() + b"\0")
    update("payload", value)
    return digest.hexdigest()

def hash_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    payload = load_checkpoint_payload(target)
    state = dict(payload["model_state_dict"])
    buffers = {name: tensor for name, tensor in state.items() if name.endswith(_BUFFER_SUFFIXES)}
    parameters = {name: tensor for name, tensor in state.items() if name not in buffers}
    counters = {key: value for key, value in payload.items() if key not in {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict"}}
    return {
        "path": str(target), "artifact_file": file_sha256(target),
        "parameters": tensor_mapping_hash(parameters), "buffers": tensor_mapping_hash(buffers),
        "tensor_state": tensor_mapping_hash(state),
        "optimizer_state": _hash_nested(payload.get("optimizer_state_dict")),
        "scheduler_state": _hash_nested(payload.get("scheduler_state_dict")),
        "training_counters": canonical_hash(counters), "counter_values": counters,
    }

def _normalization_record(
    config_path: Path,
    *,
    expected_num_test_windows: int = 1748,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    normalization = dict(config.get("normalization", {}))
    required = {
        "normalize": True, "method": "zscore", "target_scaling_strategy": "shared",
        "num_train_windows": 6115, "num_val_windows": 873,
        "num_test_windows": int(expected_num_test_windows),
    }
    for key, expected in required.items():
        if normalization.get(key) != expected:
            raise ValueError(f"checkpoint H normalization mismatch in {config_path}: {key}={normalization.get(key)!r}")
    if normalization.get("x_location", [None])[0] != normalization.get("y_location"):
        raise ValueError(f"X/y scaler location mismatch in {config_path}")
    if normalization.get("x_scale", [None])[0] != normalization.get("y_scale"):
        raise ValueError(f"X/y scaler scale mismatch in {config_path}")
    return normalization

def register_checkpoint_h(
    root: str | Path,
    *,
    expected_epochs: int = 130,
    expected_num_test_windows: int = 1748,
) -> CheckpointHRegistry:
    root = Path(root).resolve()
    if not root.is_dir(): raise FileNotFoundError(f"checkpoint H directory does not exist: {root}")
    clients, servers, normalization = [], [], []
    for cluster_dir in sorted(root.glob("cluster_*"), key=lambda item: item.name):
        cluster_id = cluster_dir.name.removeprefix("cluster_")
        for checkpoint in sorted(cluster_dir.glob("client_*_final.pt")):
            match = _CLIENT_RE.match(checkpoint.name)
            if not match: continue
            client_id = match.group("client_id")
            progress_path = cluster_dir / f"client_{client_id}_progress.json"
            config_path = cluster_dir / f"client_{client_id}_config.json"
            if not progress_path.is_file() or not config_path.is_file():
                raise ValueError(f"checkpoint H client {cluster_id}/{client_id} lacks progress or config metadata")
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if int(progress.get("completed_epochs", -1)) != expected_epochs or progress.get("train_stream_active") is True:
                raise ValueError(f"checkpoint H client {cluster_id}/{client_id} is incomplete")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model = config.get("model", {})
            expected_model = {"input_size": 1, "hidden_size": 128, "num_layers": 3, "split_layer": 2, "output_size": 1}
            if any(model.get(key) != value for key, value in expected_model.items()):
                raise ValueError(f"checkpoint H architecture mismatch in {config_path}")
            if config.get("data_quality", {}).get("dq_mode") != "none":
                raise ValueError(f"checkpoint H unexpectedly enables DQ behavior in {config_path}")
            clients.append({"cluster_id": cluster_id, "client_id": client_id, "relative_path": str(checkpoint.relative_to(root)), **hash_checkpoint_payload(checkpoint)})
            normalization.append({
                "cluster_id": cluster_id,
                "client_id": client_id,
                **_normalization_record(
                    config_path,
                    expected_num_test_windows=expected_num_test_windows,
                ),
            })
        server_path = cluster_dir / "server_final.pt"
        if server_path.is_file():
            servers.append({"cluster_id": cluster_id, "relative_path": str(server_path.relative_to(root)), **hash_checkpoint_payload(server_path)})
    if len(clients) != 84: raise ValueError(f"checkpoint H must contain exactly 84 final clients, found {len(clients)}")
    counts = {cluster: sum(row["cluster_id"] == cluster for row in clients) for cluster in sorted({row["cluster_id"] for row in clients})}
    if counts != {"1": 21, "2": 21, "3": 21, "4": 21}: raise ValueError(f"checkpoint H population mismatch: {counts}")
    if len(servers) != 4: raise ValueError(f"checkpoint H must contain four final servers, found {len(servers)}")
    if len({row["tensor_state"] for row in servers}) != 1: raise ValueError("checkpoint H final server tensor states are not identical")
    global_path = root / "global" / "global_server_latest.pt"
    if not global_path.is_file(): raise ValueError("checkpoint H is missing global/global_server_latest.pt")
    global_server = {"relative_path": str(global_path.relative_to(root)), **hash_checkpoint_payload(global_path)}
    ordered = clients + servers + [global_server]
    return CheckpointHRegistry(root, tuple(clients), tuple(servers), global_server,
        canonical_hash([row["tensor_state"] for row in ordered]),
        canonical_hash([row["artifact_file"] for row in ordered]), canonical_hash(normalization))

def validate_checkpoint_h(
    root: str | Path,
    *,
    expected_epochs: int = 130,
    expected_num_test_windows: int = 1748,
) -> dict[str, Any]:
    return register_checkpoint_h(
        root, expected_epochs=expected_epochs, expected_num_test_windows=expected_num_test_windows,
    ).as_dict()

def classify_storage_mode(policy: Any) -> str:
    """Pure reporting label derived from UpdatePolicy fields -- never
    consulted by the checkpoint loader/resolver, which needs no method or
    policy knowledge at all (see utils.runtime_utils' local-first, declared
    -base-reference-fallback resolution). Used only for run_manifest.json
    provenance/human-readability.
    """
    if not policy.selected_clients_update and not policy.control_clients_update:
        return "reference_base"
    if policy.selected_clients_update and not policy.control_clients_update and not policy.server_updates:
        return "changed_only"
    return "full_final"


BASE_REFERENCE_SCHEMA_VERSION = 1


def _write_base_reference(registry: CheckpointHRegistry, method_root: Path) -> dict[str, Any]:
    """``auto`` mode: never physically duplicate H at method-init time, for
    any client/server/global artifact -- write a lightweight, self-
    contained declaration instead, from hashes register_checkpoint_h already
    computed (no re-reading, no copying). checkpoints/base_reference.json is
    the sole authority the resolver consults for whether H-fallback is
    allowed (see utils.runtime_utils.load_checkpoint_base_reference) --
    run_manifest.json's storage-mode fields are provenance only and are
    never read by any resolution logic.
    """
    reference_dir = method_root / "checkpoints"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_path = reference_dir / "base_reference.json"
    if reference_path.exists():
        raise FileExistsError(f"checkpoint base reference already exists: {reference_path}")
    try:
        checkpoint_h_root = os.path.relpath(registry.root, start=method_root)
        path_type = "relative_to_method_root"
    except ValueError:
        # Only possible on platforms where the two paths can be on different
        # drives (not POSIX) -- kept as a documented, exercised escape hatch
        # rather than an assumption that relative always succeeds.
        checkpoint_h_root = str(registry.root)
        path_type = "absolute"
    payload = {
        "base_reference_schema_version": BASE_REFERENCE_SCHEMA_VERSION,
        "path_type": path_type,
        "checkpoint_h_root": checkpoint_h_root,
        "checkpoint_h_combined_tensor_state_hash": registry.combined_tensor_state_hash,
        "written_by": "stage_method_checkpoint",
        "clients": [
            {"cluster_id": row["cluster_id"], "client_id": row["client_id"], "artifact_file": row["artifact_file"], "tensor_state": row["tensor_state"]}
            for row in registry.clients
        ],
        "servers": [
            {"cluster_id": row["cluster_id"], "artifact_file": row["artifact_file"], "tensor_state": row["tensor_state"]}
            for row in registry.servers
        ],
        "global_server": {"artifact_file": registry.global_server["artifact_file"], "tensor_state": registry.global_server["tensor_state"]},
    }
    write_json(reference_path, payload)
    return {
        "source_root": str(registry.root), "base_reference_path": str(reference_path), "checkpoint_count": 0,
        "combined_tensor_state_hash": registry.combined_tensor_state_hash,
        "combined_artifact_hash": registry.combined_artifact_hash,
        "baseline_progress_files_copied": False, "deployment_counters_start_at_zero": True, "checkpoints": [],
    }


def stage_method_checkpoint(registry: CheckpointHRegistry, method_root: str | Path, *, mode: str = "auto") -> dict[str, Any]:
    if mode not in ("auto", "legacy_full"):
        raise ValueError(f"unsupported checkpoint staging mode: {mode!r}")
    method_root = Path(method_root)
    progress_dir = method_root / "checkpoints" / "deployment_progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    for client in registry.clients:
        write_json(progress_dir / f"cluster_{client['cluster_id']}_client_{client['client_id']}.json", {
            "adaptation_pass": 0, "deployment_windows_seen": 0, "adaptation_windows_seen": 0,
            "client_optimizer_steps": 0, "server_optimizer_steps": 0, "fedavg_events": 0, "complete": False,
        })

    if mode == "auto":
        return _write_base_reference(registry, method_root)

    # legacy_full: today's exact copy-everything-up-front behavior, byte for
    # byte -- kept as an explicit debug/A-B-comparison override, never the
    # default.
    initial = method_root / "checkpoints" / "initial"
    if initial.exists(): raise FileExistsError(f"staged checkpoint already exists: {initial}")
    copied = []
    for source in [*registry.clients, *registry.servers, registry.global_server]:
        relative = Path(source["relative_path"])
        destination = initial / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(registry.root / relative, destination)
        staged = hash_checkpoint_payload(destination)
        for field in ("artifact_file", "parameters", "buffers", "tensor_state", "optimizer_state", "scheduler_state", "training_counters"):
            if staged[field] != source[field]: raise ValueError(f"staged checkpoint hash mismatch for {relative}: {field}")
        copied.append({"relative_path": str(relative), **staged})
    result = {
        "source_root": str(registry.root), "initial_root": str(initial), "checkpoint_count": len(copied),
        "combined_tensor_state_hash": canonical_hash([row["tensor_state"] for row in copied]),
        "combined_artifact_hash": canonical_hash([row["artifact_file"] for row in copied]),
        "baseline_progress_files_copied": False, "deployment_counters_start_at_zero": True, "checkpoints": copied,
    }
    write_json(method_root / "checkpoint_hashes_before.json", {"checkpoint_H_combined_artifact_hash": registry.combined_artifact_hash, "staged_combined_artifact_hash": result["combined_artifact_hash"], "checkpoints": copied})
    write_json(method_root / "tensor_state_hashes_before.json", {"checkpoint_H_combined_tensor_state_hash": registry.combined_tensor_state_hash, "staged_combined_tensor_state_hash": result["combined_tensor_state_hash"], "checkpoints": copied})
    return result
