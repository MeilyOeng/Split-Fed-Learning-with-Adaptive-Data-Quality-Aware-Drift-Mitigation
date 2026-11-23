"""Offline derivation of M3's EWC reference: theta_star and a diagonal
empirical-Fisher-style parameter-importance estimate for each client,
computed once from checkpoint H's own clean_validation data.

This never touches checkpoint H itself (read-only), never runs a live
client/server subprocess (a pure in-process forward/backward over frozen H
weights), and never uses drifted, deployment, oracle, or counterfactual
data -- only the clean_validation role H's own training was legitimately
allowed to see.

Terminology: the importance estimate here is a *diagonal squared-gradient
importance approximation* (sometimes called "empirical Fisher"), not an
exact Fisher information matrix. It is computed from squared *per-batch*
gradients (not per-sample gradients), averaged over batches -- a coarser
but tractable estimator; see ``_diagonal_importance`` below.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from data.prepared_window_store import PreparedWindowStore, file_sha256
from models.split_lstm import LSTMConfig, build_split_lstm
from utils.runtime_utils import (
    load_checkpoint_base_reference,
    resolve_client_checkpoint_fallback,
    try_load_torch_checkpoint,
)

from ..checkpoint_registry import hash_checkpoint_payload
from ..core import EWCConfig, ModelArchitecture, canonical_hash, ewc_config_from_mapping, write_json

FISHER_BATCH_SIZE = 128
_CLIENT_FINAL_RE = re.compile(r"client_(?P<client_id>\w+)_final\.pt$")


def _discover_checkpoint_h(checkpoint_h: Path) -> dict[str, Any]:
    """Minimal, population-size-agnostic checkpoint H discovery.

    Unlike checkpoint_registry.register_checkpoint_h (which enforces the
    production 84-client/4-cluster population and a fixed epoch count),
    this only needs a stable tensor/artifact identity hash and a listing
    of client/server final checkpoints -- population and epoch validation
    already happen upstream (checkpoint-H acceptance, prepared.runner's
    _activate_h) before any method, including M3, is ever allowed to run
    against a given checkpoint H.
    """
    clients: list[dict[str, Any]] = []
    for path in sorted(checkpoint_h.glob("cluster_*/client_*_final.pt")):
        match = _CLIENT_FINAL_RE.match(path.name)
        if not match:
            continue
        clients.append({
            "cluster_id": path.parent.name.removeprefix("cluster_"), "client_id": match.group("client_id"),
            "relative_path": str(path.relative_to(checkpoint_h)), **hash_checkpoint_payload(path),
        })
    servers: list[dict[str, Any]] = []
    for path in sorted(checkpoint_h.glob("cluster_*/server_final.pt")):
        servers.append({
            "cluster_id": path.parent.name.removeprefix("cluster_"),
            "relative_path": str(path.relative_to(checkpoint_h)), **hash_checkpoint_payload(path),
        })
    if not clients or not servers:
        raise ValueError(f"checkpoint H has no client/server final checkpoints under {checkpoint_h}")
    ordered = clients + servers
    return {
        "clients": clients, "servers": servers,
        "combined_tensor_state_hash": canonical_hash([row["tensor_state"] for row in ordered]),
        "combined_artifact_hash": canonical_hash([row["artifact_file"] for row in ordered]),
    }


def _load_h_clean_validation_store(checkpoint_h: Path, preparation_root: Path) -> tuple[PreparedWindowStore, dict[str, Any]]:
    """Resolve checkpoint H's clean_validation identity and a store to read it from.

    Prefers ``<checkpoint_h>/h_training_view/``: a physically-preserved
    copy made once, at H-training time, by
    ``prepared.training.build_h_training_view`` -- it never changes again,
    so its clean_validation identity cannot drift even if the *current*
    preparation is later regenerated (checkpoint_H_epoch130_manifest.json's
    own clean_validation_hash field is NOT reliable for this: it is
    overwritten every time validate-checkpoint-h runs, against whichever
    preparation was passed that time).

    Falls back to the current preparation if h_training_view/ is absent
    *or* physically present but unreadable by the current prepared-data
    reader (e.g. a payload-hash-scheme change since H was trained), and
    only after verifying the fallback's clean_validation hash matches
    what checkpoint H's own acceptance manifest last recorded -- fail
    closed on mismatch, since there is otherwise no way to know the
    fallback data is really what H was trained against.
    """
    training_view = checkpoint_h / "h_training_view"
    if training_view.is_dir():
        try:
            manifest = json.loads((training_view / "manifests" / "prepared_dataset_hashes.json").read_text(encoding="utf-8"))
            clean_validation_hash = str(manifest["roles"]["clean_validation"]["combined_payload_hash"])
            store = PreparedWindowStore(training_view, allowed_roles=("clean_validation",))
            # Prove the view is actually readable by the *current* reader,
            # not merely present: a prepared-payload-hash-scheme change
            # since H was trained (see data/prepared_window_store.py) can
            # leave a real, physically intact h_training_view/ that the
            # current code can no longer verify. One real client is enough
            # to detect that systemically -- fall through to the verified
            # fallback rather than raising here.
            probe_clients = manifest["roles"]["clean_validation"]["clients"]
            probe_cluster, probe_client = next(iter(probe_clients)).split(":", 1)
            store.load_client("clean_validation", probe_cluster, probe_client)
            return store, {"source": "h_training_view", "clean_validation_hash": clean_validation_hash}
        except (ValueError, KeyError, StopIteration, OSError, json.JSONDecodeError):
            pass
    h_manifest_path = checkpoint_h / "checkpoint_H_epoch130_manifest.json"
    if not h_manifest_path.is_file():
        raise ValueError(
            f"checkpoint H has neither h_training_view/ nor checkpoint_H_epoch130_manifest.json "
            f"to verify clean_validation identity against: {checkpoint_h}"
        )
    h_manifest = json.loads(h_manifest_path.read_text(encoding="utf-8"))
    prepared = json.loads((preparation_root / "manifests" / "prepared_dataset_hashes.json").read_text(encoding="utf-8"))
    current_hash = str(prepared["roles"]["clean_validation"]["combined_payload_hash"])
    recorded_hash = str(h_manifest.get("clean_validation_hash", ""))
    if current_hash != recorded_hash:
        raise ValueError(
            "EWC reference clean_validation identity mismatch (fail-closed): checkpoint H has "
            "no h_training_view/ to read clean_validation from directly, and the current "
            f"preparation's clean_validation hash ({current_hash!r}) does not match the hash "
            f"checkpoint H's own acceptance manifest last recorded ({recorded_hash!r}). Refusing "
            "to build an EWC reference from unverified data."
        )
    store = PreparedWindowStore(preparation_root, allowed_roles=("clean_validation",))
    return store, {"source": "current_preparation_verified", "clean_validation_hash": current_hash}


def _build_client_server(model: ModelArchitecture) -> tuple[nn.Module, nn.Module]:
    cfg = LSTMConfig(
        input_size=model.input_size, hidden_size=model.hidden_size, num_layers=model.num_layers,
        dropout=model.dropout, bidirectional=model.bidirectional, output_size=model.output_size,
    )
    return build_split_lstm(cfg, model.split_layer)


def _diagonal_importance(
    client_model: nn.Module, server_model: nn.Module, samples, *, fisher_max_batches: int | None,
) -> dict[str, Any]:
    """Diagonal squared-gradient importance from squared *per-batch*
    forecasting-loss gradients, averaged over batches (not per-sample
    gradients averaged over samples -- a coarser, more tractable
    empirical-Fisher approximation; documented here per its actual
    construction rather than claimed as an exact Fisher matrix).

    Reproduces the exact production forward pass (client backbone -> last
    timestep -> unsqueeze to a length-1 sequence -> server backbone+head)
    so the loss landscape being probed matches what M3's client update
    will actually be adapting under.
    """
    X = torch.as_tensor(samples.X, dtype=torch.float32)
    y = torch.as_tensor(samples.y, dtype=torch.float32).reshape(-1)
    names = [name for name, p in client_model.named_parameters() if p.requires_grad]
    params = [p for p in client_model.parameters() if p.requires_grad]
    accum = [torch.zeros_like(p) for p in params]
    criterion = nn.MSELoss()
    batch_count = 0
    sample_count = 0
    n = X.shape[0]
    for start in range(0, n, FISHER_BATCH_SIZE):
        if fisher_max_batches is not None and batch_count >= fisher_max_batches:
            break
        end = min(start + FISHER_BATCH_SIZE, n)
        x_batch, y_batch = X[start:end], y[start:end]
        activations = client_model(x_batch)[:, -1, :]
        # Active split contract: server receives only the last hidden
        # state and treats it as a length-1 sequence (server.py's
        # _ensure_lstm_input_3d) -- reproduced explicitly here since this
        # runs offline, without that server code path.
        predictions = server_model(activations.unsqueeze(1))
        loss = criterion(predictions, y_batch.unsqueeze(1))
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
        for acc, grad in zip(accum, grads):
            acc += grad.detach() ** 2
        batch_count += 1
        sample_count += int(x_batch.shape[0])
    if batch_count == 0:
        raise ValueError("no clean_validation batches available for EWC importance estimation")
    diagonal = {name: (acc / batch_count).clone() for name, acc in zip(names, accum)}
    return {"diagonal": diagonal, "sample_count": sample_count, "batch_count": batch_count}


def _ewc_reference_identity(
    registry: dict[str, Any], clean_validation_hash: str, model: ModelArchitecture, ewc: EWCConfig,
) -> dict[str, Any]:
    """Identity fields deliberately exclude drift cohort, client_selection_seed,
    drift process parameters, and lambda -- none of these affect how the
    reference is *constructed* (only checkpoint H, clean_validation,
    architecture, and Fisher configuration do), so a future cross-suite
    cache could reuse this same identity format without redesign, even
    though this implementation stores references suite-locally.
    """
    return {
        "checkpoint_h_combined_tensor_state_hash": registry["combined_tensor_state_hash"],
        "checkpoint_h_combined_artifact_hash": registry["combined_artifact_hash"],
        "clean_validation_hash": clean_validation_hash,
        "model_architecture": asdict(model),
        "fisher_source": ewc.fisher_source,
        "fisher_type": ewc.fisher_type,
        "fisher_batch_size": FISHER_BATCH_SIZE,
        "fisher_max_batches": ewc.fisher_max_batches,
    }


def build_or_load_ewc_reference(
    checkpoint_h: str | Path, preparation_root: str | Path, config: Mapping[str, Any], suite_root: str | Path,
) -> dict[str, Any]:
    """Return the manifest for this suite's EWC reference, building it if
    absent, reusing it if identity matches, and failing closed if an
    existing reference's identity does not match (never silently
    recomputed or silently trusted).
    """
    checkpoint_h = Path(checkpoint_h).resolve()
    preparation_root = Path(preparation_root).resolve()
    suite_root = Path(suite_root).resolve()
    model = ModelArchitecture(**dict(config.get("model") or {}))
    model.validate()
    ewc = ewc_config_from_mapping(config.get("ewc"))
    ewc.validate()

    registry = _discover_checkpoint_h(checkpoint_h)
    store, clean_validation_info = _load_h_clean_validation_store(checkpoint_h, preparation_root)
    identity = _ewc_reference_identity(registry, clean_validation_info["clean_validation_hash"], model, ewc)
    identity_hash = canonical_hash(identity)

    reference_root = suite_root / "ewc_reference"
    manifest_path = reference_root / "ewc_reference_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity_hash") == identity_hash:
            return existing
        raise ValueError(
            f"existing EWC reference at {reference_root} has a different identity than the "
            "current checkpoint H / clean_validation / architecture / Fisher configuration; "
            "refusing to silently rebuild or overwrite it (fail-closed). Use a new suite root "
            "if the reference is genuinely meant to change."
        )

    reference_root.mkdir(parents=True, exist_ok=True)
    client_entries: list[dict[str, Any]] = []
    cluster_ids = sorted({row["cluster_id"] for row in registry["clients"]}, key=int)
    for cluster_id in cluster_ids:
        server_row = next(row for row in registry["servers"] if row["cluster_id"] == cluster_id)
        _, server_model = _build_client_server(model)
        try_load_torch_checkpoint(server_model, None, checkpoint_h / server_row["relative_path"])
        server_model.eval()
        for client_row in [row for row in registry["clients"] if row["cluster_id"] == cluster_id]:
            client_id = client_row["client_id"]
            client_model, _ = _build_client_server(model)
            try_load_torch_checkpoint(client_model, None, checkpoint_h / client_row["relative_path"])
            client_model.eval()
            theta_star = {name: tensor.detach().clone() for name, tensor in client_model.state_dict().items()}
            samples = store.load_client("clean_validation", cluster_id, client_id)
            importance = _diagonal_importance(client_model, server_model, samples, fisher_max_batches=ewc.fisher_max_batches)
            cluster_dir = reference_root / f"cluster_{cluster_id}"
            cluster_dir.mkdir(exist_ok=True)
            reference_path = cluster_dir / f"client_{client_id}_ewc_reference.pt"
            torch.save({
                "client_id": client_id, "cluster_id": cluster_id,
                "checkpoint_h": str(checkpoint_h),
                "checkpoint_h_client_tensor_state_hash": client_row["tensor_state"],
                "theta_star": theta_star,
                "fisher_diagonal": importance["diagonal"],
                "fisher_source_role": "clean_validation",
                "fisher_type": "empirical_diagonal_squared_batch_gradient",
                "fisher_sample_count": importance["sample_count"],
                "fisher_batch_count": importance["batch_count"],
                "fisher_batch_size": FISHER_BATCH_SIZE,
                "loss_definition": "nn.MSELoss(reduction='mean') on server(client(X)[:, -1, :].unsqueeze(1))",
                "model_architecture": asdict(model),
            }, reference_path)
            client_entries.append({
                "cluster_id": cluster_id, "client_id": client_id,
                "relative_path": str(reference_path.relative_to(reference_root)),
                "reference_artifact_sha256": file_sha256(reference_path),
                "fisher_sample_count": importance["sample_count"], "fisher_batch_count": importance["batch_count"],
            })
    manifest = {
        "identity": identity, "identity_hash": identity_hash,
        "checkpoint_h": str(checkpoint_h),
        "clean_validation_source": clean_validation_info["source"],
        "clean_validation_hash": clean_validation_info["clean_validation_hash"],
        "fisher_source": "clean_validation",
        "fisher_type": "empirical_diagonal_squared_batch_gradient",
        "fisher_batch_size": FISHER_BATCH_SIZE,
        "fisher_max_batches": ewc.fisher_max_batches,
        "clients": client_entries,
    }
    write_json(manifest_path, manifest)
    return manifest


def load_client_ewc_reference(suite_root: str | Path, cluster_id: str, client_id: str) -> dict[str, Any]:
    """Load one client's theta_star/fisher_diagonal for use during M3 adaptation."""
    path = Path(suite_root).resolve() / "ewc_reference" / f"cluster_{cluster_id}" / f"client_{client_id}_ewc_reference.pt"
    if not path.is_file():
        raise FileNotFoundError(f"no EWC reference for cluster={cluster_id} client={client_id}: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def summarize_fisher_importance(suite_root: str | Path) -> list[dict[str, Any]]:
    """Optional diagnostic (Part E): per-client Fisher-importance summary
    statistics, read entirely from the already-built EWC reference -- no
    new inference. Returns one row per client; writes nothing itself.
    Empty list if this suite has no EWC reference (e.g. it never ran M3).
    """
    suite_root = Path(suite_root).resolve()
    manifest_path = suite_root / "ewc_reference" / "ewc_reference_manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["clients"]:
        reference = load_client_ewc_reference(suite_root, entry["cluster_id"], entry["client_id"])
        values = torch.cat([tensor.flatten() for tensor in reference["fisher_diagonal"].values()])
        rows.append({
            "cluster_id": entry["cluster_id"], "client_id": entry["client_id"],
            "fisher_mean": float(values.mean()), "fisher_std": float(values.std(unbiased=False)),
            "fisher_max": float(values.max()), "fisher_p95": float(torch.quantile(values, 0.95)),
            "fraction_near_zero": float((values < 1e-8).float().mean()),
        })
    return rows


def _parameter_movement_l2(theta_after: dict[str, Any], theta_star: dict[str, Any]) -> float:
    squared_norm = sum(
        float(((theta_after[name] - theta_star[name]).float() ** 2).sum()) for name in theta_star
    )
    return squared_norm ** 0.5


def _summarize_adaptation_parameter_movement_from_initial(method_root: Path, initial_root: Path) -> list[dict[str, Any]]:
    """checkpoints/initial/ present (legacy_full runs, or a hand-built
    fixture) -- theta_star comes from that physical copy, exactly as
    before this function had any other source for it.
    """
    rows: list[dict[str, Any]] = []
    for initial_path in sorted(initial_root.glob("cluster_*/client_*_final.pt")):
        cluster_id = initial_path.parent.name.removeprefix("cluster_")
        client_id = _CLIENT_FINAL_RE.match(initial_path.name).group("client_id")
        after_path = method_root / f"cluster_{cluster_id}" / f"client_{client_id}_final.pt"
        if not after_path.is_file():
            continue
        theta_star = torch.load(initial_path, map_location="cpu", weights_only=False)["model_state_dict"]
        theta_after = torch.load(after_path, map_location="cpu", weights_only=False)["model_state_dict"]
        rows.append({"cluster_id": cluster_id, "client_id": client_id, "parameter_movement_l2": _parameter_movement_l2(theta_after, theta_star)})
    return rows


def _summarize_adaptation_parameter_movement_from_base_reference(method_root: Path, reference_path: Path) -> list[dict[str, Any]]:
    """checkpoints/initial/ absent (auto-mode runs never create it) --
    theta_star is resolved through the same local-first/declared-base
    -reference-fallback contract every other checkpoint consumer uses
    (utils.runtime_utils), never a hardcoded relative path.
    """
    base_reference = load_checkpoint_base_reference(reference_path if reference_path.is_file() else None)
    if base_reference is None:
        return []
    rows: list[dict[str, Any]] = []
    for after_path in sorted(method_root.glob("cluster_*/client_*_final.pt")):
        cluster_id = after_path.parent.name.removeprefix("cluster_")
        client_id = _CLIENT_FINAL_RE.match(after_path.name).group("client_id")
        fallback = resolve_client_checkpoint_fallback(base_reference, reference_path, cluster_id, client_id)
        if not fallback.path.is_file():
            continue
        theta_star = torch.load(fallback.path, map_location="cpu", weights_only=False)["model_state_dict"]
        theta_after = torch.load(after_path, map_location="cpu", weights_only=False)["model_state_dict"]
        rows.append({"cluster_id": cluster_id, "client_id": client_id, "parameter_movement_l2": _parameter_movement_l2(theta_after, theta_star)})
    return rows


def summarize_adaptation_parameter_movement(method_root: str | Path) -> list[dict[str, Any]]:
    """Optional diagnostic (Part E): ||theta_after - theta_star||_2 for
    every client this method actually adapted, read entirely from already
    -saved checkpoints -- no new inference, and works identically for M2
    and M3 (or any other selected-clients-update method) without needing an
    EWC reference at all.
    """
    method_root = Path(method_root).resolve()
    initial_root = method_root / "checkpoints" / "initial"
    if initial_root.is_dir():
        return _summarize_adaptation_parameter_movement_from_initial(method_root, initial_root)
    return _summarize_adaptation_parameter_movement_from_base_reference(method_root, method_root / "checkpoints" / "base_reference.json")
