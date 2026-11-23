"""Part E: production-equation-matched diagnostic replay of M3's
client-only EWC update, for gradient/loss/parameter-scale diagnostics only.

Explicitly NOT a bit-identical reproduction of production M3: dropout=0.4
is active during adaptation (client_model.train(), matching production),
and exact dropout masks depend on the complete RNG call history of the real
subprocess run, which this replay does not reproduce. Use the canonical
CLI's own M3(lambda=0)-vs-M2 run, not this replay, for the strict
equivalence check that gates production correctness.

What IS reproduced exactly: the same forward/backward split-learning
equations (client backbone -> last timestep -> server frozen-activation
gradient -> activations.backward(gradient=grad) -> EWC penalty backward
accumulated onto the same .grad -> clip -> optimizer.step()), the same
chronological batch order/batch_size/adaptation_epochs from the supplied
config, selected clients only, server parameters frozen throughout, no
FedAvg. Every checkpoint is loaded via torch.load into a freshly
constructed in-memory module; nothing is ever written back into checkpoint
H, an EWC reference, or a completed method's output directory.

    python -m drift.mitigation.diagnostics.ewc_replay \
        --checkpoint-h <checkpoint_h_root> --suite-root <suite_root> --preparation <preparation_root> \
        --client-lr 0.01 --lambda 1.0 --adaptation-epochs 1 --batch-size 128 \
        --output diagnostics/ewc_replay
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import common
from .representation import _h_client_checkpoint, _h_server_checkpoint, load_cohort

EPS = 1e-8


def _flatten(grad: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.flatten() for tensor in grad.values()]) if grad else torch.zeros(0)


def _l2(grad: dict[str, torch.Tensor]) -> float:
    flat = _flatten(grad)
    return float(flat.norm().item()) if flat.numel() else 0.0


def _load_ewc_reference(suite_root: Path, cluster_id: str, client_id: str) -> dict[str, Any]:
    """Reads the existing per-client EWC reference .pt file directly (same
    known artifact schema drift/mitigation/prepared/ewc.py writes) --
    intentionally not importing that module.
    """
    path = suite_root / "ewc_reference" / f"cluster_{cluster_id}" / f"client_{client_id}_ewc_reference.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def _adaptation_batches(X: np.ndarray, y: np.ndarray, *, batch_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (X[start:start + batch_size], y[start:start + batch_size])
        for start in range(0, X.shape[0], batch_size)
    ]


def _tercile(fraction: float) -> str:
    if fraction < 1 / 3:
        return "early"
    if fraction < 2 / 3:
        return "middle"
    return "late"


def replay_client(
    *, client_model: torch.nn.Module, server_model: torch.nn.Module, theta_star: dict[str, torch.Tensor],
    fisher: dict[str, torch.Tensor], lambda_: float, X: np.ndarray, y: np.ndarray, batch_size: int,
    adaptation_epochs: int, learning_rate: float, grad_clip_max_norm: float, cluster_id: str, client_id: str,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.Adam(client_model.parameters(), lr=learning_rate)
    server_model.eval()
    rows: list[dict[str, Any]] = []
    batches = _adaptation_batches(X, y, batch_size=batch_size)
    total_batches = len(batches) * adaptation_epochs
    global_batch_index = 0

    for epoch in range(1, adaptation_epochs + 1):
        for batch_index, (x_batch, y_batch) in enumerate(batches, start=1):
            client_model.train()
            optimizer.zero_grad(set_to_none=True)
            x_t = torch.as_tensor(x_batch, dtype=torch.float32)
            y_t = torch.as_tensor(y_batch, dtype=torch.float32).reshape(-1)

            activations = client_model(x_t)[:, -1, :]
            server_result = common.server_predict(server_model, activations, y_t)
            forecast_loss = float(server_result["loss"])
            activations.backward(gradient=server_result["grad_activations"])
            forecast_grad = {name: p.grad.detach().clone() for name, p in client_model.named_parameters() if p.grad is not None}
            forecast_grad_l2 = _l2(forecast_grad)

            ewc_loss = 0.0
            ewc_grad_l2 = 0.0
            grad_cosine: float | None = None
            if lambda_ > 0:
                ewc_penalty = common.ewc_penalty_loss_diagnostic(client_model.named_parameters(), theta_star, fisher, lambda_)
                ewc_loss = float(ewc_penalty.item())
                ewc_penalty.backward()
                combined_grad_after_ewc = {name: p.grad.detach().clone() for name, p in client_model.named_parameters() if p.grad is not None}
                ewc_grad = {name: combined_grad_after_ewc[name] - forecast_grad.get(name, torch.zeros_like(combined_grad_after_ewc[name])) for name in combined_grad_after_ewc}
                ewc_grad_l2 = _l2(ewc_grad)
                f_flat, e_flat = _flatten(forecast_grad), _flatten(ewc_grad)
                if f_flat.numel() and e_flat.numel() and f_flat.norm() > EPS and e_flat.norm() > EPS:
                    grad_cosine = float(F.cosine_similarity(f_flat.unsqueeze(0), e_flat.unsqueeze(0)).item())

            combined_grad_l2_pre_clip = float(
                torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in client_model.parameters() if p.grad is not None)).item()
            )
            gradient_clip_threshold = grad_clip_max_norm
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=grad_clip_max_norm)
            combined_grad_l2_post_clip = float(
                torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in client_model.parameters() if p.grad is not None)).item()
            )
            was_clipped = combined_grad_l2_pre_clip > gradient_clip_threshold + 1e-9

            theta_before = {name: p.detach().clone() for name, p in client_model.named_parameters()}
            theta_before_l2 = float(torch.sqrt(sum((t ** 2).sum() for t in theta_before.values())).item())
            optimizer.step()
            theta_after = {name: p.detach().clone() for name, p in client_model.named_parameters()}
            actual_parameter_step_l2 = float(
                torch.sqrt(sum(((theta_after[name] - theta_before[name]) ** 2).sum() for name in theta_before)).item()
            )

            global_batch_index += 1
            adaptation_fraction = global_batch_index / max(1, total_batches)
            rows.append({
                "cluster_id": cluster_id, "client_id": client_id,
                "adaptation_epoch": epoch, "batch_index": batch_index, "global_batch_index": global_batch_index,
                "adaptation_fraction": adaptation_fraction, "adaptation_phase": _tercile(adaptation_fraction),
                "forecast_loss": forecast_loss, "ewc_loss_unscaled": ewc_loss / lambda_ if lambda_ > 0 else 0.0,
                "ewc_loss_scaled": ewc_loss, "total_loss": forecast_loss + ewc_loss,
                "forecast_grad_l2": forecast_grad_l2, "ewc_grad_l2": ewc_grad_l2,
                "ewc_to_forecast_grad_ratio": ewc_grad_l2 / (forecast_grad_l2 + EPS),
                "gradient_cosine": grad_cosine,
                "combined_grad_l2_pre_clip": combined_grad_l2_pre_clip,
                "gradient_clip_threshold": gradient_clip_threshold, "was_clipped": was_clipped,
                "combined_grad_l2_post_clip": combined_grad_l2_post_clip,
                "actual_parameter_step_l2": actual_parameter_step_l2,
                "relative_parameter_step_l2": actual_parameter_step_l2 / (theta_before_l2 + EPS),
                "lambda": lambda_,
            })
    return rows


def _layer_name(tensor_name: str) -> str:
    parts = tensor_name.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else tensor_name


def run_ewc_replay_diagnostic(
    *, checkpoint_h: str | Path, suite_root: str | Path, preparation_root: str | Path, output: str | Path,
    lambda_: float, adaptation_epochs: int, client_lr: float, batch_size: int = 128, grad_clip_max_norm: float = 1.0,
    arch: common.ModelArchitectureSpec | None = None, seed: int = 42,
) -> dict[str, Any]:
    checkpoint_h, suite_root, preparation_root, output = Path(checkpoint_h), Path(suite_root), Path(preparation_root), Path(output)
    arch = arch or common.ModelArchitectureSpec()
    torch.manual_seed(seed)
    cohort = load_cohort(suite_root)
    selected = cohort.loc[cohort.selected_for_drift]

    per_batch_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        cluster_id, client_id = row["cluster_id"], row["client_id"]
        client_model = common.load_client_model(_h_client_checkpoint(checkpoint_h, cluster_id, client_id), arch)
        server_model = common.load_server_model(_h_server_checkpoint(checkpoint_h, cluster_id), arch)
        reference = _load_ewc_reference(suite_root, cluster_id, client_id)
        npz = common.load_prepared_window(preparation_root, "drifted_deployment", cluster_id, client_id)
        per_batch_rows.extend(replay_client(
            client_model=client_model, server_model=server_model,
            theta_star=reference["theta_star"], fisher=reference["fisher_diagonal"], lambda_=lambda_,
            X=npz["X"], y=npz["y"], batch_size=batch_size, adaptation_epochs=adaptation_epochs,
            learning_rate=client_lr, grad_clip_max_norm=grad_clip_max_norm, cluster_id=cluster_id, client_id=client_id,
        ))

    per_batch = pd.DataFrame(per_batch_rows)
    common.write_csv(output / "ewc_gradient_diagnostics_per_batch.csv", per_batch)

    if not per_batch.empty:
        phase_summary = per_batch.groupby("adaptation_phase").agg(
            n_batches=("global_batch_index", "count"),
            median_forecast_grad_l2=("forecast_grad_l2", "median"),
            median_ewc_grad_l2=("ewc_grad_l2", "median"),
            median_ewc_to_forecast_grad_ratio=("ewc_to_forecast_grad_ratio", "median"),
            median_actual_parameter_step_l2=("actual_parameter_step_l2", "median"),
            fraction_clipped=("was_clipped", "mean"),
        ).reset_index()
    else:
        phase_summary = pd.DataFrame(columns=["adaptation_phase", "n_batches", "median_forecast_grad_l2", "median_ewc_grad_l2", "median_ewc_to_forecast_grad_ratio", "median_actual_parameter_step_l2", "fraction_clipped"])
    common.write_csv(output / "ewc_gradient_diagnostics_summary.csv", phase_summary)

    gradient_scale_reference: dict[str, Any] = {"lambda_used": lambda_}
    if lambda_ > 0 and not per_batch.empty:
        # ewc_grad_l2 (hence the ratio r) scales linearly in lambda, since
        # the EWC loss itself is linear in lambda: r(L) = L * k where k is
        # the lambda-independent ||g_EWC_unit||/||g_forecast|| ratio. Solving
        # k*lambda_target = target_ratio for lambda_target given a
        # measurement at lambda_used: lambda_target = target_ratio * lambda_used / r.
        r = float(per_batch["ewc_to_forecast_grad_ratio"].median())
        gradient_scale_reference.update({
            "note": "gradient-scale reference values, NOT optimal lambdas -- for choosing a sensible sweep range only",
            "median_ratio_at_lambda_used": r,
            "lambda_for_ratio_0p1": (0.1 * lambda_ / r) if r > 0 else None,
            "lambda_for_ratio_1p0": (1.0 * lambda_ / r) if r > 0 else None,
        })
    common.write_json(output / "gradient_scale_reference.json", gradient_scale_reference)

    metadata = common.diagnostic_provenance(
        diagnostic_name="ewc_replay", repo_root=Path(__file__).resolve().parents[3],
        parameters={
            "lambda": lambda_, "adaptation_epochs": adaptation_epochs, "client_lr": client_lr,
            "batch_size": batch_size, "grad_clip_max_norm": grad_clip_max_norm, "seed": seed,
        },
        source_suite_root=suite_root, checkpoint_h=checkpoint_h, preparation_root=preparation_root,
    )
    metadata["replay_kind"] = "production-equation-matched diagnostic replay (NOT bit-identical reproduction)"
    common.write_json(output / "metadata.json", metadata)

    return {"output": str(output), "per_batch_rows": len(per_batch), "gradient_scale_reference": gradient_scale_reference}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production-equation-matched diagnostic replay of M3's client-only EWC update (read-only w.r.t. all existing artifacts).")
    parser.add_argument("--checkpoint-h", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("diagnostics/ewc_replay"))
    parser.add_argument("--lambda", dest="lambda_", type=float, required=True)
    parser.add_argument("--adaptation-epochs", type=int, default=1)
    parser.add_argument("--client-lr", type=float, required=True, help="must match the client learning rate the real run used")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-clip-max-norm", type=float, default=1.0, help="matches production's clip_grad_norm_ max_norm constant")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_ewc_replay_diagnostic(
        checkpoint_h=args.checkpoint_h, suite_root=args.suite_root, preparation_root=args.preparation,
        output=args.output, lambda_=args.lambda_, adaptation_epochs=args.adaptation_epochs,
        client_lr=args.client_lr, batch_size=args.batch_size, grad_clip_max_norm=args.grad_clip_max_norm, seed=args.seed,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())