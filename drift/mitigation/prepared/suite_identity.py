"""Immutable identity registration for a prepared production run root."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Any

from ..core import BaselineHContract, ModelArchitecture, canonical_hash, load_suite_yaml, write_json
from .preparation import validate_offline_preparation
from .training import baseline_h_contract_from_config


def _derive_model_architecture_from_h(checkpoint_h: Path) -> ModelArchitecture:
    """Best-effort, read-only architecture recovery for an H that predates
    both the explicit BaselineHContract stamp and the ``model:`` config
    block.

    Every real checkpoint H in this repository was trained by
    ``client/client.py``, which writes each client's resolved architecture
    into its own ``client_ID_config.json`` (`"model": {...}`) regardless
    of whether the mitigation config schema knew about ``model:`` yet.
    Reading that file back is therefore safely derivable for any real H.
    Falls back to the documented default architecture -- the only one any
    H in this repository has ever actually used -- when no such file is
    present (e.g. a minimal/synthetic test fixture), rather than failing
    read-only derivation outright.
    """
    required = ("input_size", "hidden_size", "num_layers", "split_layer", "output_size", "dropout")
    for config_path in sorted(checkpoint_h.glob("cluster_*/client_*_config.json")):
        try:
            client_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model = client_config.get("model", {})
        if all(key in model for key in required):
            return ModelArchitecture(
                input_size=int(model["input_size"]), hidden_size=int(model["hidden_size"]),
                num_layers=int(model["num_layers"]), split_layer=int(model["split_layer"]),
                output_size=int(model["output_size"]), dropout=float(model["dropout"]),
                bidirectional=bool(model.get("bidirectional", False)),
            )
    return ModelArchitecture()


def _resolve_h_baseline_contract(checkpoint_h: Path, config: dict[str, Any], h_manifest: dict[str, Any]) -> dict[str, Any]:
    """Read the explicit contract a freshly trained H is stamped with, or
    fall back to deriving one, read-only, from an existing H's own
    checkpoint_H_epoch130_manifest.json, per-client config.json (for
    architecture), plus the current config. Any H trained before this
    contract existed (including every H currently in this repository)
    takes the fallback path; it is never rewritten.
    """
    stamped_path = checkpoint_h / "checkpoint_H_baseline_contract.json"
    if stamped_path.is_file():
        stamped = json.loads(stamped_path.read_text(encoding="utf-8"))
        return {"source": "explicit", "contract": dict(stamped["contract"]), "contract_hash": str(stamped["contract_hash"])}
    model = _derive_model_architecture_from_h(checkpoint_h)
    contract = BaselineHContract(
        window_size=int(config["windowing"]["window_size"]),
        forecast_horizon=int(config["windowing"]["forecast_horizon"]),
        h_training_seed=int(config.get("seeds", {}).get("h_training_seed", 42)),
        normalization_method=str(config["normalization"]["method"]),
        scaler_manifest_hash=str(h_manifest["scaler_manifest_hash"]),
        coordinate_manifest_hash=str(h_manifest["coordinate_manifest_hash"]),
        clean_train_hash=str(h_manifest["clean_train_hash"]),
        clean_validation_hash=str(h_manifest["clean_validation_hash"]),
        model_input_size=model.input_size, model_hidden_size=model.hidden_size,
        model_num_layers=model.num_layers, model_split_layer=model.split_layer,
        model_output_size=model.output_size, model_dropout=model.dropout,
        model_bidirectional=model.bidirectional,
    )
    return {"source": "derived_legacy_read_only", "contract": asdict(contract), "contract_hash": contract.content_hash()}


def _assert_baseline_h_contract_compatible(h_contract: dict[str, Any], scenario_contract: dict[str, Any]) -> None:
    """Gate H reuse on data/architecture compatibility only.

    h_training_seed is deliberately excluded: it describes how H's weights
    were originally produced, not whether an already-trained H remains
    valid for a new scenario. A scenario config that later declares a
    different h_training_seed (e.g. because it separately plans to train a
    fresh H) must not block reusing this already-trained one.

    Model architecture fields ARE included: a different hidden size,
    layer count, split layer, dropout, or input/output size changes what
    H's own weights mean and shape-checks against, so it must require a
    new H rather than being silently reused.
    """
    reuse_fields = (
        "window_size", "forecast_horizon", "normalization_method",
        "scaler_manifest_hash", "coordinate_manifest_hash",
        "clean_train_hash", "clean_validation_hash",
        "model_input_size", "model_hidden_size", "model_num_layers",
        "model_split_layer", "model_output_size", "model_dropout", "model_bidirectional",
    )
    mismatched = [field for field in reuse_fields if h_contract[field] != scenario_contract[field]]
    if mismatched:
        raise ValueError(f"checkpoint H is not reusable for this scenario; baseline contract disagrees on {mismatched}")


def initialize_prepared_suite_root(
    suite_root: str | Path,
    preparation_root: str | Path,
    checkpoint_h: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Create or verify the top-level identities required by evaluation."""
    suite_root = Path(suite_root).resolve()
    preparation_root = Path(preparation_root).resolve()
    checkpoint_h = Path(checkpoint_h).resolve()
    config_path = Path(config_path).resolve()
    validate_offline_preparation(preparation_root)
    prepared = json.loads(
        (preparation_root / "manifests" / "prepared_dataset_hashes.json").read_text(encoding="utf-8")
    )
    config = load_suite_yaml(config_path)
    h_manifest_path = checkpoint_h / "checkpoint_H_epoch130_manifest.json"
    h_acceptance_path = checkpoint_h / "checkpoint_H_acceptance.json"
    if not h_manifest_path.is_file() or not h_acceptance_path.is_file():
        raise ValueError("checkpoint H must pass validate-checkpoint-h before method execution")
    h_manifest = json.loads(h_manifest_path.read_text(encoding="utf-8"))
    h_acceptance = json.loads(h_acceptance_path.read_text(encoding="utf-8"))
    if not h_acceptance.get("accepted") or int(h_manifest.get("epoch", -1)) != 130:
        raise ValueError("checkpoint H acceptance is false or not epoch 130")
    h_baseline = _resolve_h_baseline_contract(checkpoint_h, config, h_manifest)
    scenario_contract = asdict(baseline_h_contract_from_config(config, prepared))
    _assert_baseline_h_contract_compatible(h_baseline["contract"], scenario_contract)
    identity = {
        "prepared_root": str(preparation_root),
        "prepared_dataset_manifest_hash": canonical_hash(prepared),
        "drifted_deployment_hash": prepared["roles"]["drifted_deployment"]["combined_payload_hash"],
        "selected_cohort_hash": prepared["selected_cohort_hash"],
        "checkpoint_H": str(checkpoint_h),
        "checkpoint_H_tensor_hash": h_manifest["combined_tensor_hash"],
        "checkpoint_H_optimizer_hash": h_manifest["combined_optimizer_hash"],
        "baseline_h_contract_source": h_baseline["source"],
        "baseline_h_contract_hash": h_baseline["contract_hash"],
        "config_hash": canonical_hash(config),
    }
    suite_root.mkdir(parents=True, exist_ok=True)
    manifest_path = suite_root / "suite_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise ValueError("existing prepared suite root has different immutable identities")
    else:
        write_json(
            manifest_path,
            {
                "backend": "offline_prepared_v2", "identity": identity,
                "selected_methods": [str(name) for name in config["methods"]],
            },
        )
        shutil.copy2(config_path, suite_root / "suite_resolved_config.yaml")
        shutil.copy2(
            preparation_root / "manifests" / "selected_drift_clients.csv",
            suite_root / "selected_drift_clients.csv",
        )
        shutil.copy2(
            preparation_root / "manifests" / "selected_drift_clients_hash.json",
            suite_root / "selected_drift_clients_hash.json",
        )
        shutil.copy2(
            preparation_root / "manifests" / "split_manifest.json",
            suite_root / "split_manifest.json",
        )
        write_json(suite_root / "prepared_data_reference.json", identity)
        (suite_root / "comparison").mkdir()
    return identity
