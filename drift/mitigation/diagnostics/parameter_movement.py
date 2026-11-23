"""Part D2: parameter movement from checkpoint H (theta_star) to a completed
method's final client checkpoint, per tensor / layer / client / overall.

Read-only: opens checkpoints via torch.load and never writes to them.
theta_star is read directly from checkpoint H's own client_final.pt (the
synchronized H client final state), which is the same tensor content the
EWC reference's own theta_star was copied from at construction time.

    python -m drift.mitigation.diagnostics.parameter_movement \
        --checkpoint-h <checkpoint_h_root> --suite-root <suite_root> \
        --methods M2_oracle_client_only M3_oracle_client_ewc --output diagnostics/parameter_movement
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import common
from .representation import _h_client_checkpoint, _method_client_checkpoint, load_cohort


def _layer_name(tensor_name: str) -> str:
    parts = tensor_name.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else tensor_name


def run_parameter_movement_diagnostic(
    *, checkpoint_h: str | Path, suite_root: str | Path, methods: list[str], output: str | Path, eps: float = 1e-8,
) -> dict[str, Any]:
    checkpoint_h, suite_root, output = Path(checkpoint_h), Path(suite_root), Path(output)
    cohort = load_cohort(suite_root)
    selected = cohort.loc[cohort.selected_for_drift]

    per_tensor_rows: list[dict[str, Any]] = []
    per_client_rows: list[dict[str, Any]] = []

    for method in methods:
        for _, row in selected.iterrows():
            cluster_id, client_id = row["cluster_id"], row["client_id"]
            theta_star = common.load_model_state_dict(_h_client_checkpoint(checkpoint_h, cluster_id, client_id))
            after_path = _method_client_checkpoint(suite_root, method, cluster_id, client_id)
            if not after_path.is_file():
                continue
            theta_after = common.load_model_state_dict(after_path)

            layer_l2sq: dict[str, float] = {}
            overall_l2sq = 0.0
            overall_star_norm_sq = 0.0
            for name, star_tensor in theta_star.items():
                after_tensor = theta_after[name]
                diff = (after_tensor.float() - star_tensor.float())
                l2sq = float((diff ** 2).sum().item())
                star_norm_sq = float((star_tensor.float() ** 2).sum().item())
                l2 = float(np.sqrt(l2sq))
                star_norm = float(np.sqrt(star_norm_sq))
                per_tensor_rows.append({
                    "method": method, "cluster_id": cluster_id, "client_id": client_id, "tensor_name": name,
                    "layer": _layer_name(name), "parameter_movement_l2": l2,
                    "relative_movement": l2 / (star_norm + eps),
                })
                layer_l2sq[_layer_name(name)] = layer_l2sq.get(_layer_name(name), 0.0) + l2sq
                overall_l2sq += l2sq
                overall_star_norm_sq += star_norm_sq

            per_client_rows.append({
                "method": method, "cluster_id": cluster_id, "client_id": client_id,
                "parameter_movement_l2_overall": float(np.sqrt(overall_l2sq)),
                "relative_movement_overall": float(np.sqrt(overall_l2sq)) / (float(np.sqrt(overall_star_norm_sq)) + eps),
                **{f"parameter_movement_l2_{layer}": float(np.sqrt(value)) for layer, value in layer_l2sq.items()},
            })

    per_tensor = pd.DataFrame(per_tensor_rows)
    per_client = pd.DataFrame(per_client_rows)
    common.write_csv(output / "parameter_movement_per_tensor.csv", per_tensor)
    common.write_csv(output / "parameter_movement_per_client.csv", per_client)

    metadata = common.diagnostic_provenance(
        diagnostic_name="parameter_movement", repo_root=Path(__file__).resolve().parents[3],
        parameters={"methods": methods}, source_suite_root=suite_root, checkpoint_h=checkpoint_h,
    )
    common.write_json(output / "metadata.json", metadata)
    return {"output": str(output), "per_tensor_rows": len(per_tensor), "per_client_rows": len(per_client)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parameter movement from checkpoint H to a completed method's final checkpoint (read-only).")
    parser.add_argument("--checkpoint-h", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["M2_oracle_client_only", "M3_oracle_client_ewc"])
    parser.add_argument("--output", type=Path, default=Path("diagnostics/parameter_movement"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_parameter_movement_diagnostic(
        checkpoint_h=args.checkpoint_h, suite_root=args.suite_root, methods=args.methods, output=args.output,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())