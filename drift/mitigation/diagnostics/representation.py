"""Part B: M2/M3 representation-mismatch diagnostic against checkpoint H.

Scientific question: after client-only adaptation (M2) or client-only EWC
adaptation (M3), how far does the client's cut-layer representation h(X)
move from checkpoint H's own representation on the same X, and does the
frozen H server's performance degrade when fed the moved representation
instead of H's own?

Computed on two data contexts (per an explicit correction to the original
plan):
  - "drifted_final_holdout": the method's own final-holdout X (observed,
    post-drift) -- measures adaptation to the new regime.
  - "clean_counterfactual_final_holdout": the same clients' clean
    counterfactual final-holdout X (oracle-enabled methods only) -- measures
    how much the adapted encoder's representation of *old*-regime data
    moved, a proxy for clean-task forgetting.

Read-only: every checkpoint is loaded via torch.load and never re-saved;
every prepared window is loaded via np.load; nothing under an existing
checkpoint H root or suite root is ever written to. Server evaluation reuses
the pure, side-effect-free frozen_server_activation_step from
server/frozen_gradient.py -- inference only, never an optimizer step.

    python -m drift.mitigation.diagnostics.representation \
        --suite-root <suite_root> --checkpoint-h <checkpoint_h_root> \
        --method M2 --output diagnostics/representation_mismatch
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import common


@dataclass(frozen=True)
class DataContext:
    name: str
    npz_role: str


DRIFTED_FINAL_HOLDOUT = DataContext("drifted_final_holdout", "drifted_deployment")
CLEAN_COUNTERFACTUAL_FINAL_HOLDOUT = DataContext("clean_counterfactual_final_holdout", "clean_final_holdout_counterfactual")


def _h_client_checkpoint(checkpoint_h: Path, cluster_id: str, client_id: str) -> Path:
    return checkpoint_h / f"cluster_{cluster_id}" / f"client_{client_id}_final.pt"


def _h_server_checkpoint(checkpoint_h: Path, cluster_id: str) -> Path:
    return checkpoint_h / f"cluster_{cluster_id}" / "server_final.pt"


def _method_client_checkpoint(suite_root: Path, method: str, cluster_id: str, client_id: str) -> Path:
    return suite_root / method / f"cluster_{cluster_id}" / f"client_{client_id}_final.pt"


def _method_server_checkpoint(suite_root: Path, method: str, cluster_id: str) -> Path:
    return suite_root / method / f"cluster_{cluster_id}" / "server_final.pt"


def _final_holdout_predictions_dir(suite_root: Path, method: str, *, oracle_enabled: bool) -> Path:
    method_root = suite_root / method
    nested = method_root / "persistent_drift_final_holdout"
    if oracle_enabled and nested.is_dir():
        return nested
    return method_root


def load_cohort(suite_root: Path) -> pd.DataFrame:
    """Reads <suite_root>/selected_drift_clients.csv -- copied there directly
    (no manifests/ prefix) by prepared.suite_identity.initialize_prepared_suite_root
    from the preparation root's manifests/selected_drift_clients.csv. This
    mirrors evaluation.py's own _load_cohort exactly.
    """
    frame = pd.read_csv(Path(suite_root) / "selected_drift_clients.csv", dtype={"cluster_id": str, "client_id": str})
    frame["selected_for_drift"] = frame["selected_for_drift"].astype(str).str.lower().isin(["true", "1"])
    return frame


def _select_npz_rows(npz: dict[str, np.ndarray], predictions: pd.DataFrame) -> np.ndarray:
    """Match prepared window rows to prediction rows on
    (window_start_index, window_end_index), exactly, then return X in the
    predictions frame's row order. Fails closed on any missing/ambiguous
    match rather than assuming positional alignment.
    """
    npz_keys = pd.DataFrame({
        "window_start_index": npz["window_start_index"].astype(int),
        "window_end_index": npz["window_end_index"].astype(int),
        "_row": np.arange(len(npz["window_start_index"])),
    })
    wanted = predictions[["window_start_index", "window_end_index"]].astype(int).reset_index(drop=True)
    wanted["_order"] = np.arange(len(wanted))
    merged = wanted.merge(npz_keys, on=["window_start_index", "window_end_index"], how="left", validate="one_to_one")
    if merged["_row"].isna().any():
        missing = merged.loc[merged["_row"].isna(), ["window_start_index", "window_end_index"]].head(5)
        raise ValueError(f"prediction rows with no matching prepared window: {missing.to_dict('records')}")
    ordered = merged.sort_values("_order")
    return npz["X"][ordered["_row"].astype(int).to_numpy()]


def _load_context_samples(
    *, preparation_root: Path, suite_root: Path, method: str, oracle_enabled: bool,
    context: DataContext, cluster_id: str, client_id: str,
) -> tuple[torch.Tensor, pd.DataFrame] | None:
    """Returns (X, predictions_frame) aligned row-for-row, or None if this
    context has no data for this method (e.g. clean counterfactual holdout
    for a non-oracle method).
    """
    if context is DRIFTED_FINAL_HOLDOUT:
        pred_dir = _final_holdout_predictions_dir(suite_root, method, oracle_enabled=oracle_enabled)
        path = pred_dir / f"cluster_{cluster_id}" / f"client_{client_id}_test_predictions.csv"
        if not path.is_file():
            return None
        predictions = common.load_test_predictions_csv(pred_dir, cluster_id, client_id)
    else:
        if not oracle_enabled:
            return None
        pred_dir = suite_root / method / "counterfactual_clean_holdout"
        path = pred_dir / f"cluster_{cluster_id}" / f"client_{client_id}_test_predictions.csv"
        if not path.is_file():
            return None
        predictions = common.load_test_predictions_csv(pred_dir, cluster_id, client_id)
    npz = common.load_prepared_window(preparation_root, context.npz_role, cluster_id, client_id)
    X = _select_npz_rows(npz, predictions)
    return torch.as_tensor(X, dtype=torch.float32), predictions


def _representation_row_metrics(h_rep: torch.Tensor, other_rep: torch.Tensor, *, eps: float = 1e-8) -> dict[str, float]:
    diff = other_rep - h_rep
    l2_per_sample = diff.norm(dim=1)
    h_norm_per_sample = h_rep.norm(dim=1)
    l2_normalized = l2_per_sample / (h_norm_per_sample + eps)
    cos = torch.nn.functional.cosine_similarity(other_rep, h_rep, dim=1, eps=eps)
    mean_shift = (other_rep.mean(dim=0) - h_rep.mean(dim=0)).norm()
    std_ratio = float(other_rep.std().item() / (float(h_rep.std().item()) + eps))
    return {
        "n_samples": int(h_rep.shape[0]),
        "representation_l2": float(l2_per_sample.mean().item()),
        "representation_l2_normalized": float(l2_normalized.mean().item()),
        "cosine_similarity": float(cos.mean().item()),
        "representation_mean_shift": float(mean_shift.item()),
        "representation_std_ratio": std_ratio,
    }


def run_representation_diagnostic(
    *, preparation_root: str | Path, checkpoint_h: str | Path, suite_root: str | Path,
    method: str, output: str | Path, arch: common.ModelArchitectureSpec | None = None,
    include_m1_comparison: bool = True,
) -> dict[str, Any]:
    preparation_root, checkpoint_h, suite_root, output = (
        Path(preparation_root), Path(checkpoint_h), Path(suite_root), Path(output)
    )
    arch = arch or common.ModelArchitectureSpec()
    cohort = load_cohort(suite_root)
    selected = cohort.loc[cohort.selected_for_drift]

    per_client_rows: list[dict[str, Any]] = []
    server_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        cluster_id, client_id = row["cluster_id"], row["client_id"]
        h_client = common.load_client_model(_h_client_checkpoint(checkpoint_h, cluster_id, client_id), arch)
        method_client = common.load_client_model(_method_client_checkpoint(suite_root, method, cluster_id, client_id), arch)
        h_server = common.load_server_model(_h_server_checkpoint(checkpoint_h, cluster_id), arch)

        for context in (DRIFTED_FINAL_HOLDOUT, CLEAN_COUNTERFACTUAL_FINAL_HOLDOUT):
            loaded = _load_context_samples(
                preparation_root=preparation_root, suite_root=suite_root, method=method,
                oracle_enabled=True, context=context, cluster_id=cluster_id, client_id=client_id,
            )
            if loaded is None:
                continue
            X, predictions = loaded
            y = torch.as_tensor(predictions["y_true"].to_numpy(dtype=np.float32))

            h_rep = common.cut_layer_representation(h_client, X)
            other_rep = common.cut_layer_representation(method_client, X)
            metrics = _representation_row_metrics(h_rep, other_rep)
            per_client_rows.append({
                "cluster_id": cluster_id, "client_id": client_id, "method": method,
                "segment": context.name, **metrics,
            })

            with_h_rep = common.server_predict(h_server, h_rep, y)
            with_other_rep = common.server_predict(h_server, other_rep, y)
            mae_h = float(torch.mean(torch.abs(with_h_rep["predictions"].reshape(-1) - y)).item())
            mae_other = float(torch.mean(torch.abs(with_other_rep["predictions"].reshape(-1) - y)).item())
            server_rows.append({
                "cluster_id": cluster_id, "client_id": client_id, "method": method, "segment": context.name,
                "n_samples": int(len(y)),
                "H_server_MAE_with_H_rep": mae_h,
                "H_server_MAE_with_adapted_rep": mae_other,
                "representation_induced_MAE_delta": mae_other - mae_h,
                "H_server_loss_with_H_rep": with_h_rep["loss"],
                "H_server_loss_with_adapted_rep": with_other_rep["loss"],
            })

        if include_m1_comparison:
            m1_client_path = _method_client_checkpoint(suite_root, "M1_continued_splitfed", cluster_id, client_id)
            if m1_client_path.is_file():
                loaded = _load_context_samples(
                    preparation_root=preparation_root, suite_root=suite_root, method="M1_continued_splitfed",
                    oracle_enabled=False, context=DRIFTED_FINAL_HOLDOUT, cluster_id=cluster_id, client_id=client_id,
                )
                if loaded is not None:
                    X, predictions = loaded
                    y = torch.as_tensor(predictions["y_true"].to_numpy(dtype=np.float32))
                    m1_client = common.load_client_model(m1_client_path, arch)
                    m1_server_path = _method_server_checkpoint(suite_root, "M1_continued_splitfed", cluster_id)
                    h_rep = common.cut_layer_representation(h_client, X)
                    m1_rep = common.cut_layer_representation(m1_client, X)
                    metrics = _representation_row_metrics(h_rep, m1_rep)
                    per_client_rows.append({
                        "cluster_id": cluster_id, "client_id": client_id, "method": "M1_continued_splitfed",
                        "segment": DRIFTED_FINAL_HOLDOUT.name, **metrics,
                    })
                    with_h_server = common.server_predict(h_server, m1_rep, y)
                    row = {
                        "cluster_id": cluster_id, "client_id": client_id, "method": "M1_continued_splitfed",
                        "segment": DRIFTED_FINAL_HOLDOUT.name, "n_samples": int(len(y)),
                        "H_server_MAE_with_H_rep": float(torch.mean(torch.abs(common.server_predict(h_server, h_rep, y)["predictions"].reshape(-1) - y)).item()),
                        "H_server_MAE_with_adapted_rep": float(torch.mean(torch.abs(with_h_server["predictions"].reshape(-1) - y)).item()),
                    }
                    row["representation_induced_MAE_delta"] = row["H_server_MAE_with_adapted_rep"] - row["H_server_MAE_with_H_rep"]
                    if m1_server_path.is_file():
                        m1_server = common.load_server_model(m1_server_path, arch)
                        with_m1_server = common.server_predict(m1_server, m1_rep, y)
                        row["M1_server_MAE_with_M1_rep"] = float(torch.mean(torch.abs(with_m1_server["predictions"].reshape(-1) - y)).item())
                    server_rows.append(row)

    per_client = pd.DataFrame(per_client_rows)
    server_compat = pd.DataFrame(server_rows)

    suite_id = suite_root.name
    out_dir = output / suite_id
    common.write_csv(out_dir / "representation_per_client.csv", per_client)
    common.write_csv(out_dir / "server_compatibility_per_client.csv", server_compat)

    if not per_client.empty:
        rep_summary = per_client.groupby(["method", "segment"]).agg(
            mean_representation_l2=("representation_l2", "mean"),
            median_representation_l2=("representation_l2", "median"),
            mean_cosine_similarity=("cosine_similarity", "mean"),
            n_clients=("client_id", "nunique"),
        ).reset_index()
    else:
        rep_summary = pd.DataFrame(columns=["method", "segment", "mean_representation_l2", "median_representation_l2", "mean_cosine_similarity", "n_clients"])
    common.write_csv(out_dir / "representation_summary.csv", rep_summary)

    if not server_compat.empty:
        server_summary = server_compat.groupby(["method", "segment"]).agg(
            mean_MAE_with_H_rep=("H_server_MAE_with_H_rep", "mean"),
            mean_MAE_with_adapted_rep=("H_server_MAE_with_adapted_rep", "mean"),
            mean_representation_induced_MAE_delta=("representation_induced_MAE_delta", "mean"),
            n_clients=("client_id", "nunique"),
        ).reset_index()
    else:
        server_summary = pd.DataFrame(columns=["method", "segment", "mean_MAE_with_H_rep", "mean_MAE_with_adapted_rep", "mean_representation_induced_MAE_delta", "n_clients"])
    common.write_csv(out_dir / "server_compatibility_summary.csv", server_summary)

    metadata = common.diagnostic_provenance(
        diagnostic_name="representation_mismatch",
        repo_root=Path(__file__).resolve().parents[3],
        parameters={"method": method, "include_m1_comparison": include_m1_comparison},
        source_suite_root=suite_root, checkpoint_h=checkpoint_h, preparation_root=preparation_root,
    )
    common.write_json(out_dir / "metadata.json", metadata)

    return {
        "output": str(out_dir),
        "per_client_rows": len(per_client),
        "server_compatibility_rows": len(server_compat),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M2/M3 representation-mismatch diagnostic against checkpoint H (read-only).")
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--checkpoint-h", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--method", default="M2_oracle_client_only")
    parser.add_argument("--output", type=Path, default=Path("diagnostics/representation_mismatch"))
    parser.add_argument("--no-m1-comparison", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_representation_diagnostic(
        preparation_root=args.preparation, checkpoint_h=args.checkpoint_h, suite_root=args.suite_root,
        method=args.method, output=args.output, include_m1_comparison=not args.no_m1_comparison,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())