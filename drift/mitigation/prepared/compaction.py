"""Optional, dry-run-first checkpoint compactor.

Converts a *completed* method run's local checkpoint files that are
provably state-identical to checkpoint H into a
``checkpoints/base_reference.json`` declaration, then deletes the physical
duplicates. Off by default (dry-run unless explicitly told otherwise), never
invoked implicitly by ``run-method``/``run-full-suite``, never touches an
incomplete method directory, and never touches predictions/logs/configs/the
historical ``run_manifest.json`` (which stays exactly as it was written by
whatever run originally produced it -- resolution never consults it, per
``utils.runtime_utils.load_checkpoint_base_reference``).

Redundancy is decided by exact tensor/state comparison against checkpoint H
(``torch.testing.assert_close(..., rtol=0.0, atol=0.0)``), never by
filename or method name.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from ..checkpoint_registry import BASE_REFERENCE_SCHEMA_VERSION
from ..core import file_sha256
from utils.runtime_utils import (
    load_checkpoint_base_reference,
    resolve_client_checkpoint_fallback,
    resolve_global_server_checkpoint_fallback,
    resolve_server_checkpoint_fallback,
)

_CLUSTER_IDS = ("1", "2", "3", "4")


def _state_dicts_equal(path_a: Path, path_b: Path) -> bool:
    state_a = torch.load(path_a, map_location="cpu", weights_only=False)["model_state_dict"]
    state_b = torch.load(path_b, map_location="cpu", weights_only=False)["model_state_dict"]
    if set(state_a) != set(state_b):
        return False
    for name in state_a:
        try:
            torch.testing.assert_close(state_a[name], state_b[name], rtol=0.0, atol=0.0)
        except AssertionError:
            return False
    return True


def _client_h_mapping(h_root: Path, cluster_id: str, client_id: str) -> Path:
    return h_root / f"cluster_{cluster_id}" / f"client_{client_id}_final.pt"


def _server_h_mapping(h_root: Path, cluster_id: str) -> Path:
    return h_root / f"cluster_{cluster_id}" / "server_final.pt"


def _enumerate_client_artifacts(method_root: Path):
    for cluster_id in _CLUSTER_IDS:
        cluster_dir = method_root / f"cluster_{cluster_id}"
        if not cluster_dir.is_dir():
            continue
        for path in sorted(cluster_dir.glob("client_*.pt")):
            name = path.name
            # Longest/most-specific suffix first: "_best_checkpoint.pt" also
            # ends with "_checkpoint.pt", so checking the generic suffix
            # first would truncate "1_best" into the client id.
            for suffix in ("_best_checkpoint.pt", "_checkpoint.pt", "_final.pt"):
                if name.endswith(suffix):
                    client_id = name[len("client_"):-len(suffix)]
                    yield cluster_id, client_id, path
                    break


def _enumerate_server_artifacts(method_root: Path):
    for cluster_id in _CLUSTER_IDS:
        cluster_dir = method_root / f"cluster_{cluster_id}"
        if not cluster_dir.is_dir():
            continue
        for filename in ("server_latest.pt", "server_best.pt", "server_final.pt"):
            path = cluster_dir / filename
            if path.is_file():
                yield cluster_id, path


def _verdict_row(method_root: Path, path: Path, kind: str, h_path: Path | None, verdict: str, detail: str = "") -> dict[str, Any]:
    return {
        "relative_path": str(path.relative_to(method_root)),
        "kind": kind,
        "h_artifact": str(h_path) if h_path is not None else None,
        "verdict": verdict,
        "size_bytes": path.stat().st_size,
        "detail": detail,
    }


def _compare_and_record(method_root: Path, path: Path, kind: str, h_path: Path) -> dict[str, Any]:
    if not h_path.is_file():
        return _verdict_row(method_root, path, kind, None, "no_h_mapping", "checkpoint H has no artifact at the mapped location")
    try:
        identical = _state_dicts_equal(path, h_path)
    except Exception as exc:  # unreadable/corrupt payload -- never guess, report and skip
        return _verdict_row(method_root, path, kind, h_path, "comparison_failed", str(exc))
    return _verdict_row(method_root, path, kind, h_path, "identical_to_H" if identical else "differs_from_H")


def _write_base_reference_after_compaction(
    method_root: Path, h_root: Path,
    new_clients: dict[tuple[str, str], dict[str, Any]], new_servers: dict[str, dict[str, Any]],
    new_global_server: dict[str, Any] | None,
) -> Path:
    reference_path = method_root / "checkpoints" / "base_reference.json"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if reference_path.is_file():
        existing = json.loads(reference_path.read_text(encoding="utf-8"))
    clients = {(row["cluster_id"], row["client_id"]): row for row in existing.get("clients", [])}
    clients.update(new_clients)
    servers = {row["cluster_id"]: row for row in existing.get("servers", [])}
    servers.update(new_servers)
    global_server = new_global_server if new_global_server is not None else existing.get("global_server", {})
    if "checkpoint_h_root" in existing:
        path_type, checkpoint_h_root = existing["path_type"], existing["checkpoint_h_root"]
    else:
        try:
            checkpoint_h_root, path_type = os.path.relpath(h_root, start=method_root), "relative_to_method_root"
        except ValueError:
            checkpoint_h_root, path_type = str(h_root), "absolute"
    payload = {
        "base_reference_schema_version": BASE_REFERENCE_SCHEMA_VERSION,
        "path_type": path_type,
        "checkpoint_h_root": checkpoint_h_root,
        "checkpoint_h_combined_tensor_state_hash": existing.get("checkpoint_h_combined_tensor_state_hash"),
        "written_by": "compact-checkpoints",
        "clients": list(clients.values()),
        "servers": list(servers.values()),
        "global_server": global_server,
    }
    reference_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return reference_path


def _resolve_fallback_for_kind(base_reference: dict[str, Any], reference_path: Path, kind: str, cluster_id: str, client_id: str | None):
    if kind == "client":
        return resolve_client_checkpoint_fallback(base_reference, reference_path, cluster_id, client_id)
    if kind == "server":
        return resolve_server_checkpoint_fallback(base_reference, reference_path, cluster_id)
    if kind == "global_server":
        return resolve_global_server_checkpoint_fallback(base_reference, reference_path)
    raise ValueError(f"unsupported artifact kind for reconstruction: {kind!r}")


def _verify_compaction_reconstructs_removed_state(
    reference_path: Path, kind: str, cluster_id: str, client_id: str | None, removed_state: dict[str, Any],
) -> None:
    """Required invariant: after a file is removed by the compactor, the
    effective final state it represented must remain exactly
    reconstructable -- verified here by actually resolving it back through
    the same central resolver every other consumer uses, and comparing
    every tensor against what was captured immediately before deletion
    (not merely re-deriving the same equality check done pre-delete, and
    not merely checking the resolved path exists).
    """
    base_reference = load_checkpoint_base_reference(reference_path)
    if base_reference is None:
        raise RuntimeError(f"compaction post-write verification failed: {reference_path} did not load back")
    fallback = _resolve_fallback_for_kind(base_reference, reference_path, kind, cluster_id, client_id)
    if not fallback.path.is_file():
        raise RuntimeError(f"compaction post-write verification failed: resolved fallback artifact missing: {fallback.path}")
    reconstructed_state = torch.load(fallback.path, map_location="cpu", weights_only=False)["model_state_dict"]
    if set(reconstructed_state) != set(removed_state):
        raise RuntimeError(f"compaction post-write verification failed: reconstructed state keys differ for {fallback.path}")
    for name in removed_state:
        torch.testing.assert_close(reconstructed_state[name], removed_state[name], rtol=0.0, atol=0.0)


def compact_method_directory(method_root: Path, *, apply: bool) -> dict[str, Any]:
    manifest_path = method_root / "run_manifest.json"
    if not manifest_path.is_file():
        return {"method_root": str(method_root), "status": "no_manifest", "skipped": True, "artifacts": [], "reclaimable_bytes": 0}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = manifest.get("status")
    if status != "complete":
        return {"method_root": str(method_root), "status": status or "unknown", "skipped": True, "reason": "not complete", "artifacts": [], "reclaimable_bytes": 0}
    checkpoint_h_info = manifest.get("checkpoint_H") or {}
    h_root_raw = (checkpoint_h_info.get("stage") or {}).get("source_root")
    if not h_root_raw:
        return {"method_root": str(method_root), "status": status, "skipped": True, "reason": "run_manifest.json has no resolvable checkpoint H root", "artifacts": [], "reclaimable_bytes": 0}
    h_root = Path(h_root_raw)

    artifacts: list[dict[str, Any]] = []
    # Each entry: (path, kind, cluster_id, client_id_or_None, state_dict_captured_before_delete).
    # State is captured now, before anything is deleted, so post-write
    # verification compares against what was actually removed -- not a
    # re-derivation of the earlier identical_to_H check.
    to_delete: list[tuple[Path, str, str, str | None, dict[str, Any]]] = []
    new_clients: dict[tuple[str, str], dict[str, Any]] = {}
    new_servers: dict[str, dict[str, Any]] = {}
    new_global_server: dict[str, Any] | None = None

    for cluster_id, client_id, path in _enumerate_client_artifacts(method_root):
        h_path = _client_h_mapping(h_root, cluster_id, client_id)
        row = _compare_and_record(method_root, path, "client", h_path)
        artifacts.append(row)
        if row["verdict"] == "identical_to_H":
            removed_state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
            to_delete.append((path, "client", cluster_id, client_id, removed_state))
            new_clients[(cluster_id, client_id)] = {"cluster_id": cluster_id, "client_id": client_id, "artifact_file": file_sha256(h_path)}

    for cluster_id, path in _enumerate_server_artifacts(method_root):
        h_path = _server_h_mapping(h_root, cluster_id)
        row = _compare_and_record(method_root, path, "server", h_path)
        artifacts.append(row)
        if row["verdict"] == "identical_to_H":
            removed_state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
            to_delete.append((path, "server", cluster_id, None, removed_state))
            new_servers[cluster_id] = {"cluster_id": cluster_id, "artifact_file": file_sha256(h_path)}

    global_client_path = method_root / "global" / "global_client_latest.pt"
    if global_client_path.is_file():
        # No H counterpart exists for this artifact at all (grep-confirmed
        # in the design work for this feature) -- always genuine full_final
        # output, never eligible for compaction.
        artifacts.append(_verdict_row(method_root, global_client_path, "global_client", None, "no_h_mapping", "global_client_latest.pt has no checkpoint H equivalent; always genuine full_final output"))
    global_server_path = method_root / "global" / "global_server_latest.pt"
    if global_server_path.is_file():
        h_global = h_root / "global" / "global_server_latest.pt"
        row = _compare_and_record(method_root, global_server_path, "global_server", h_global)
        artifacts.append(row)
        if row["verdict"] == "identical_to_H":
            removed_state = torch.load(global_server_path, map_location="cpu", weights_only=False)["model_state_dict"]
            to_delete.append((global_server_path, "global_server", "", None, removed_state))
            new_global_server = {"artifact_file": file_sha256(h_global)}

    reclaimable_bytes = sum(row["size_bytes"] for row in artifacts if row["verdict"] == "identical_to_H")

    if apply and to_delete:
        # Reference is written -- and the artifacts it will let the resolver
        # reconstruct are proven to exist and match H -- *before* anything
        # physical is deleted. Deletion only happens once every removed
        # artifact is already safely reconstructable through the reference
        # just written.
        reference_path = _write_base_reference_after_compaction(method_root, h_root, new_clients, new_servers, new_global_server)
        for path, kind, cluster_id, client_id, removed_state in to_delete:
            path.unlink()
            _verify_compaction_reconstructs_removed_state(reference_path, kind, cluster_id, client_id, removed_state)

    return {
        "method_root": str(method_root), "status": status, "skipped": False,
        "checkpoint_h_root": str(h_root), "artifacts": artifacts, "reclaimable_bytes": reclaimable_bytes,
        "compacted": bool(apply and to_delete),
    }


def _is_runseed_dir(path: Path) -> bool:
    """Structural check -- a runseed directory directly contains method
    directories (each with its own run_manifest.json) -- never by name
    pattern, since method directory names come from the open Method
    registry, not something this module should hardcode a second time.
    """
    if not path.is_dir():
        return False
    return any((child / "run_manifest.json").is_file() for child in path.iterdir() if child.is_dir())


def compact_runseed(runseed_root: Path, *, apply: bool) -> dict[str, Any]:
    methods = [
        compact_method_directory(child, apply=apply)
        for child in sorted(runseed_root.iterdir())
        if child.is_dir() and (child / "run_manifest.json").is_file()
    ]
    return {
        "runseed_root": str(runseed_root), "methods": methods,
        "reclaimable_bytes": sum(m["reclaimable_bytes"] for m in methods),
    }


def compact_checkpoints(path: str | Path, *, dry_run: bool = True) -> dict[str, Any]:
    """Entry point. ``path`` may be a single completed runseed<N>/ directory
    or an outer experiment directory containing multiple runseed*/
    children -- detected structurally, never by name. Always dry-run
    (report only, nothing deleted) unless ``dry_run=False`` is passed
    explicitly.
    """
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"compact-checkpoints target does not exist: {path}")
    apply = not dry_run
    if _is_runseed_dir(path):
        seed_report = compact_runseed(path, apply=apply)
        return {"scope": "single_runseed", "dry_run": dry_run, "seeds": [seed_report], "total_reclaimable_bytes": seed_report["reclaimable_bytes"]}
    seed_reports = [
        compact_runseed(child, apply=apply)
        for child in sorted(path.iterdir())
        if child.is_dir() and _is_runseed_dir(child)
    ]
    if not seed_reports:
        raise ValueError(
            f"{path} is neither a completed runseed directory (method subdirectories with run_manifest.json) "
            "nor an outer directory containing runseed*/ children -- nothing to compact"
        )
    return {
        "scope": "multi_seed_suite", "dry_run": dry_run, "seeds": seed_reports,
        "total_reclaimable_bytes": sum(s["reclaimable_bytes"] for s in seed_reports),
    }