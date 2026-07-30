"""Hash-only utilities for proving that frozen validation did not mutate checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch

from .artifacts import write_json


def state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_parameter_hash(path: str | Path) -> str:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint has no model_state_dict: {path}")
    return state_dict_hash(checkpoint["model_state_dict"])


def snapshot_checkpoint_hashes(root: str | Path) -> dict[str, str]:
    base = Path(root)
    return {
        str(path.relative_to(base)): checkpoint_parameter_hash(path)
        for path in sorted(base.rglob("*.pt"))
        if path.is_file()
    }


def write_frozen_validation_report(
    output: str | Path,
    *,
    before: Mapping[str, str],
    after: Mapping[str, str],
    server_runtime_hashes: Mapping[str, str],
    round_start_events: int,
    fedavg_events: int,
) -> Path:
    changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
    payload = {
        "checkpoint_hashes_before": dict(before),
        "checkpoint_hashes_after": dict(after),
        "server_runtime_hashes": dict(server_runtime_hashes),
        "changed_checkpoint_parameters": changed,
        "parameters_unchanged": not changed,
        "round_start_events": int(round_start_events),
        "fedavg_events": int(fedavg_events),
        "frozen_validation_passed": not changed and round_start_events == 0 and fedavg_events == 0,
    }
    return write_json(output, payload)
