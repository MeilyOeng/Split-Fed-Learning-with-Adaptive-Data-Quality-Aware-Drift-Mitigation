"""Part 5 (correction): diagnostic-only comparison of batch-squared vs.
per-sample-squared empirical Fisher.

This module NEVER modifies the Fisher used by production M3 -- it computes
a completely separate, second estimate over a small, configurable client/
sample subset purely to determine whether batch-gradient cancellation is
responsible for the small Fisher/EWC effect observed at lambda=1. Nothing
here is written back into ``<suite_root>/ewc_reference/``.

Reproduces production's exact forward pass (client backbone -> last
timestep -> unsqueeze to length-1 sequence -> server backbone+head,
nn.MSELoss(reduction='mean')) since fair comparison requires probing the
identical loss landscape production's own Fisher estimate probes -- this
expression is read from and matches
drift/mitigation/prepared/ewc.py::_diagonal_importance, re-implemented
locally here rather than imported (see package docstring).

    python -m drift.mitigation.diagnostics.fisher_compare \
        --checkpoint-h <checkpoint_h_root> --clean-validation-root <preparation_or_h_training_view_root> \
        --output diagnostics/fisher_compare --max-clients 2 --max-samples 64
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import common
from .representation import _h_client_checkpoint, _h_server_checkpoint

PRODUCTION_FISHER_BATCH_SIZE = 128


def _forward_loss(client_model: nn.Module, server_model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    """Reproduces the exact production Fisher forward pass (see
    drift/mitigation/prepared/ewc.py::_diagonal_importance): client backbone
    -> last timestep -> unsqueeze(1) -> server backbone+head -> MSE.
    """
    activations = client_model(x)[:, -1, :]
    predictions = server_model(activations.unsqueeze(1))
    return criterion(predictions, y.unsqueeze(1))


def _squared_gradient(client_model: nn.Module, server_model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> dict[str, torch.Tensor]:
    names = [name for name, p in client_model.named_parameters() if p.requires_grad]
    params = [p for p in client_model.parameters() if p.requires_grad]
    loss = _forward_loss(client_model, server_model, x, y, criterion)
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
    return {name: (grad.detach() ** 2) for name, grad in zip(names, grads)}


def diagonal_importance_batch(client_model: nn.Module, server_model: nn.Module, X: np.ndarray, y: np.ndarray, *, batch_size: int) -> dict[str, torch.Tensor]:
    """F_batch = mean_batches[(gradient of batch-mean loss)^2] -- production's own definition."""
    criterion = nn.MSELoss()
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1)
    accum: dict[str, torch.Tensor] | None = None
    batch_count = 0
    for start in range(0, X_t.shape[0], batch_size):
        end = min(start + batch_size, X_t.shape[0])
        squared = _squared_gradient(client_model, server_model, X_t[start:end], y_t[start:end], criterion)
        if accum is None:
            accum = {name: value.clone() for name, value in squared.items()}
        else:
            for name in accum:
                accum[name] += squared[name]
        batch_count += 1
    if accum is None or batch_count == 0:
        raise ValueError("no samples available for batch Fisher diagnostic")
    return {name: value / batch_count for name, value in accum.items()}


def diagonal_importance_per_sample(client_model: nn.Module, server_model: nn.Module, X: np.ndarray, y: np.ndarray) -> dict[str, torch.Tensor]:
    """F_sample = mean_samples[(gradient of individual-sample loss)^2].

    Expensive by construction (one backward pass per sample) -- intended
    only for a small, explicitly-bounded diagnostic subset, never the full
    clean_validation set.
    """
    return diagonal_importance_batch(client_model, server_model, X, y, batch_size=1)


def _tensor_stats(values: np.ndarray, *, near_zero_threshold: float) -> dict[str, float]:
    if values.size == 0:
        return {"n_params": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"), "fraction_near_zero": float("nan")}
    return {
        "n_params": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
        "std": float(values.std()), "fraction_near_zero": float(np.mean(np.abs(values) < near_zero_threshold)),
    }


def _discover_available_clients(clean_validation_root: Path) -> pd.DataFrame:
    """Discover (cluster_id, client_id) pairs directly from
    prepared_windows/clean_validation/cluster_*/client_*.npz.

    Deliberately NOT sourced from manifests/selected_drift_clients.csv:
    that file lives in a preparation root but not in an h_training_view
    root, since H training must never see which clients are selected for
    drift (the same historical-access-isolation boundary
    prepared.training.build_h_training_view enforces elsewhere). Globbing
    the actual clean_validation windows works identically for both root
    kinds and doesn't care about drift-cohort membership at all -- Fisher
    diagnostics here only need "clients with clean_validation data",
    nothing about who is or isn't selected for drift.
    """
    root = Path(clean_validation_root) / "prepared_windows" / "clean_validation"
    if not root.is_dir():
        raise FileNotFoundError(
            f"--clean-validation-root must contain prepared_windows/clean_validation/ "
            f"(a preparation root or an h_training_view root works); not found at {root}"
        )
    rows: list[dict[str, str]] = []
    for cluster_dir in sorted(root.glob("cluster_*")):
        cluster_id = cluster_dir.name.removeprefix("cluster_")
        for npz_path in sorted(cluster_dir.glob("client_*.npz")):
            client_id = npz_path.stem.removeprefix("client_")
            rows.append({"cluster_id": cluster_id, "client_id": client_id})
    if not rows:
        raise FileNotFoundError(f"no client_*.npz files found under {root}")
    return pd.DataFrame(rows)


def run_fisher_compare_diagnostic(
    *, checkpoint_h: str | Path, clean_validation_root: str | Path, output: str | Path,
    max_clients: int = 2, max_samples: int = 64, arch: common.ModelArchitectureSpec | None = None,
    near_zero_threshold: float = 1e-8,
) -> dict[str, Any]:
    checkpoint_h, clean_validation_root, output = Path(checkpoint_h), Path(clean_validation_root), Path(output)
    arch = arch or common.ModelArchitectureSpec()
    clients = _discover_available_clients(clean_validation_root).drop_duplicates().head(max_clients)

    per_tensor_rows: list[dict[str, Any]] = []
    per_client_ratio_rows: list[dict[str, Any]] = []
    layer_values: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    population_values: list[tuple[np.ndarray, np.ndarray]] = []

    for _, row in clients.iterrows():
        cluster_id, client_id = row["cluster_id"], row["client_id"]
        client_model = common.load_client_model(_h_client_checkpoint(checkpoint_h, cluster_id, client_id), arch)
        server_model = common.load_server_model(_h_server_checkpoint(checkpoint_h, cluster_id), arch)
        client_model.eval(); server_model.eval()

        npz = common.load_prepared_window(clean_validation_root, "clean_validation", cluster_id, client_id)
        X, y = npz["X"][:max_samples], npz["y"][:max_samples]
        if X.shape[0] == 0:
            continue

        f_batch = diagonal_importance_batch(client_model, server_model, X, y, batch_size=PRODUCTION_FISHER_BATCH_SIZE)
        f_sample = diagonal_importance_per_sample(client_model, server_model, X, y)

        ratios: list[np.ndarray] = []
        for name in f_batch:
            batch_values = f_batch[name].flatten().numpy().astype(float)
            sample_values = f_sample[name].flatten().numpy().astype(float)
            ratio = sample_values / (batch_values + near_zero_threshold)
            ratios.append(ratio)
            layer = name.rsplit(".", 1)[0] if "." in name else name
            per_tensor_rows.append({
                "cluster_id": cluster_id, "client_id": client_id, "tensor_name": name, "layer": layer,
                "n_samples_used": int(X.shape[0]),
                **{f"F_batch_{k}": v for k, v in _tensor_stats(batch_values, near_zero_threshold=near_zero_threshold).items()},
                **{f"F_sample_{k}": v for k, v in _tensor_stats(sample_values, near_zero_threshold=near_zero_threshold).items()},
                "ratio_mean": float(ratio.mean()), "ratio_median": float(np.median(ratio)), "ratio_p95": float(np.percentile(ratio, 95)),
            })
            layer_values.setdefault(layer, []).append((batch_values, sample_values))
            population_values.append((batch_values, sample_values))
        all_ratios = np.concatenate(ratios)
        per_client_ratio_rows.append({
            "cluster_id": cluster_id, "client_id": client_id, "n_samples_used": int(X.shape[0]),
            "ratio_mean": float(all_ratios.mean()), "ratio_median": float(np.median(all_ratios)), "ratio_p95": float(np.percentile(all_ratios, 95)),
        })

    per_tensor = pd.DataFrame(per_tensor_rows)
    per_client = pd.DataFrame(per_client_ratio_rows)
    common.write_csv(output / "fisher_compare_per_tensor.csv", per_tensor)
    common.write_csv(output / "fisher_compare_per_client.csv", per_client)

    summary_rows: list[dict[str, Any]] = []
    for layer, pairs in layer_values.items():
        batch_concat = np.concatenate([pair[0] for pair in pairs])
        sample_concat = np.concatenate([pair[1] for pair in pairs])
        ratio = sample_concat / (batch_concat + near_zero_threshold)
        summary_rows.append({
            "scope": "layer", "name": layer, "n_clients": len({row["client_id"] for row in per_tensor_rows if row["layer"] == layer}),
            **{f"F_batch_{k}": v for k, v in _tensor_stats(batch_concat, near_zero_threshold=near_zero_threshold).items()},
            **{f"F_sample_{k}": v for k, v in _tensor_stats(sample_concat, near_zero_threshold=near_zero_threshold).items()},
            "ratio_mean": float(ratio.mean()), "ratio_median": float(np.median(ratio)), "ratio_p95": float(np.percentile(ratio, 95)),
        })
    if population_values:
        batch_concat = np.concatenate([pair[0] for pair in population_values])
        sample_concat = np.concatenate([pair[1] for pair in population_values])
        ratio = sample_concat / (batch_concat + near_zero_threshold)
        summary_rows.append({
            "scope": "population", "name": "all_clients", "n_clients": len(clients),
            **{f"F_batch_{k}": v for k, v in _tensor_stats(batch_concat, near_zero_threshold=near_zero_threshold).items()},
            **{f"F_sample_{k}": v for k, v in _tensor_stats(sample_concat, near_zero_threshold=near_zero_threshold).items()},
            "ratio_mean": float(ratio.mean()), "ratio_median": float(np.median(ratio)), "ratio_p95": float(np.percentile(ratio, 95)),
        })
    summary = pd.DataFrame(summary_rows)
    common.write_csv(output / "fisher_compare_summary.csv", summary)

    metadata = common.diagnostic_provenance(
        diagnostic_name="fisher_compare", repo_root=Path(__file__).resolve().parents[3],
        parameters={"max_clients": max_clients, "max_samples": max_samples, "near_zero_threshold": near_zero_threshold},
        checkpoint_h=checkpoint_h,
    )
    metadata["purpose"] = (
        "diagnostic only -- determine whether batch-gradient cancellation reduces "
        "the empirical Fisher scale used by production M3; never used to modify "
        "the saved EWC reference"
    )
    common.write_json(output / "metadata.json", metadata)
    return {"output": str(output), "per_tensor_rows": len(per_tensor), "per_client_rows": len(per_client), "summary_rows": len(summary)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnostic-only batch-squared vs per-sample-squared Fisher comparison (never touches production Fisher).")
    parser.add_argument("--checkpoint-h", type=Path, required=True)
    parser.add_argument("--clean-validation-root", type=Path, required=True, help="preparation root or h_training_view root containing prepared_windows/clean_validation/")
    parser.add_argument("--output", type=Path, default=Path("diagnostics/fisher_compare"))
    parser.add_argument("--max-clients", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_fisher_compare_diagnostic(
        checkpoint_h=args.checkpoint_h, clean_validation_root=args.clean_validation_root, output=args.output,
        max_clients=args.max_clients, max_samples=args.max_samples,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())