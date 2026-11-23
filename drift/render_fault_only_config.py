#!/usr/bin/env python3
"""Render configs for the fault-only (no-drift) mitigation ablation.

Companion to drift/run_fault_only_sweep.sh. Two things a plain per-seed
config sweep (see drift/render_seed_config.py) doesn't do on its own:

  - Pins the fault-injection cohort to the EXACT client IDs the drift
    matrix (drift/mitigation/matrix/) would select for drift_count=8 at
    the same seed, via the same stable_select_per_cluster used for real
    drift selection -- so "faulty clients" and "would-be-drifted clients"
    are identical, seed for seed. The DQ fault injector's own
    per_cluster_random selection is a different algorithm/seed derivation
    (see data quality/modules/fault_injection.py) and would NOT reproduce
    the same client set even at matching seed values, so the aligned
    cohort has to be computed once here and passed in via
    clients.selection_mode: explicit.
  - Repurposes selected_drift_clients.csv's selected_for_drift column (via
    relabel-cohort, run after prepare-data) to mark that same fault cohort.
    Real process drift is disabled in this suite family
    (population.drifted_per_cluster: 0 -- a literal zero-magnitude drift is
    rejected by ProcessSpec.validate(), so this is the only way to turn
    drift off), which means the column would otherwise be False for every
    client and evaluate_completed_suite's affected/control split -- and
    therefore recovery_summary.csv -- would be entirely empty. Relabeling
    it is what lets the existing evaluation/report machinery produce a
    meaningful fault-vs-control comparison unmodified.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from drift.mitigation.core import selected_cohort_hash, stable_select_per_cluster, write_json

DEFAULT_METHODS = ("R0_clean_frozen", "M0_drift_frozen", "M1_continued_splitfed", "M4_quality_aware_splitfed")


def load_cluster_map(assignment_path: Path) -> dict[str, list[str]]:
    frame = pd.read_csv(assignment_path, dtype=str)
    columns = {str(column).lower(): column for column in frame.columns}
    if not {"id", "cluster"}.issubset(columns):
        raise ValueError(f"assignment CSV must contain ID and Cluster columns: {assignment_path}")
    clusters: dict[str, list[str]] = {}
    for _, row in frame.iterrows():
        clusters.setdefault(str(row[columns["cluster"]]), []).append(str(row[columns["id"]]))
    return clusters


def compute_cohort(assignment_path: Path, seed: int, per_cluster: int) -> list[dict[str, Any]]:
    clusters = load_cluster_map(assignment_path)
    return stable_select_per_cluster(clusters, per_cluster=per_cluster, seed=seed)


def render_fault_config(base_fault_config: Path, seed: int, cohort: list[dict[str, Any]]) -> dict[str, Any]:
    """Every fault-severity parameter is copied from base_fault_config
    unchanged; only clients.selection_mode/selected and the top-level seed
    are overridden.
    """
    config = yaml.safe_load(base_fault_config.read_text(encoding="utf-8"))
    selected_clients = sorted(str(row["client_id"]) for row in cohort if row["selected_for_drift"])
    if not selected_clients:
        raise ValueError("computed fault cohort is empty -- check per_cluster/assignment inputs")
    config["seed"] = int(seed)
    clients_block = dict(config.get("clients") or {})
    clients_block["selection_mode"] = "explicit"
    clients_block["selected"] = selected_clients
    config["clients"] = clients_block
    return config


def render_scenario_config(
    base_template: Path,
    *,
    seed: int,
    epochs: int,
    fault_config_path: str,
    methods: tuple[str, ...] = DEFAULT_METHODS,
) -> dict[str, Any]:
    config = yaml.safe_load(base_template.read_text(encoding="utf-8"))
    config["suite_id"] = f"fault_only_s{int(seed)}_f08_e{int(epochs):02d}"

    seeds = dict(config["seeds"])
    for field in (
        "experiment_seed", "client_selection_seed", "data_order_seed",
        "drift_seed", "fault_selection_seed", "adaptation_seed",
    ):
        seeds[field] = int(seed)
    config["seeds"] = seeds

    population = dict(config["population"])
    clusters = int(population["clusters"])
    total_clients = int(population["total_clients"])
    population["drifted_per_cluster"] = 0
    population["controls_per_cluster"] = total_clients // clusters
    config["population"] = population

    faults = dict(config.get("faults") or {})
    faults["config_path"] = fault_config_path
    config["faults"] = faults

    config["methods"] = list(methods)

    adaptation = dict(config.get("adaptation") or {})
    adaptation["adaptation_epochs"] = int(epochs)
    adaptation["client_only_adaptation_epochs"] = int(epochs)
    adaptation["ewc_adaptation_epochs"] = int(epochs)
    config["adaptation"] = adaptation

    return config


def relabel_cohort_as_fault(preparation_root: Path, cohort: list[dict[str, Any]]) -> dict[str, Any]:
    manifests = preparation_root / "manifests"
    csv_path = manifests / "selected_drift_clients.csv"
    hash_path = manifests / "selected_drift_clients_hash.json"
    if not csv_path.is_file():
        raise FileNotFoundError(f"not a prepared preparation root (missing {csv_path})")

    fault_selected = {(str(row["cluster_id"]), str(row["client_id"])) for row in cohort if row["selected_for_drift"]}
    frame = pd.read_csv(csv_path, dtype={"cluster_id": str, "client_id": str})
    frame["selected_for_drift"] = [
        (cluster_id, client_id) in fault_selected
        for cluster_id, client_id in zip(frame["cluster_id"], frame["client_id"])
    ]
    frame.to_csv(csv_path, index=False)

    hash_rows = [
        {
            "cluster_id": str(row.cluster_id), "client_id": str(row.client_id),
            "selected_for_drift": bool(row.selected_for_drift), "stable_client_order": int(row.stable_client_order),
        }
        for row in frame.itertuples()
    ]
    new_hash = selected_cohort_hash(hash_rows)
    write_json(hash_path, {"sha256": new_hash, "selected_count": len(fault_selected)})
    write_json(manifests / "fault_only_relabel_note.json", {
        "note": (
            "This suite disables real process drift (population.drifted_per_cluster=0). "
            "selected_for_drift in selected_drift_clients.csv has been repurposed post-preparation "
            "to mark the fault-injection cohort instead -- the same client IDs the drift matrix "
            "would select for drift_count=8 at this seed -- so evaluate_completed_suite's "
            "affected/control split and recovery_summary.csv reflect fault exposure, not drift "
            "exposure. See drift/render_fault_only_config.py:relabel_cohort_as_fault."
        ),
        "fault_cohort_size": len(fault_selected),
    })
    return {"csv": str(csv_path), "hash_json": str(hash_path), "selected_count": len(fault_selected)}


def _int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="render the per-seed fault config and per-epoch scenario configs")
    render.add_argument("--seed", type=int, required=True)
    render.add_argument("--epochs", type=_int_list, default=[1, 3], help="comma-separated epoch settings")
    render.add_argument("--per-cluster", type=int, default=8)
    render.add_argument("--scenario-template", type=Path, required=True)
    render.add_argument("--fault-base", type=Path, required=True)
    render.add_argument("--assignments", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)

    relabel = sub.add_parser(
        "relabel-cohort",
        help="repurpose an existing preparation's selected_drift_clients.csv as the fault cohort",
    )
    relabel.add_argument("--preparation-root", type=Path, required=True)
    relabel.add_argument("--assignments", type=Path, required=True)
    relabel.add_argument("--seed", type=int, required=True)
    relabel.add_argument("--per-cluster", type=int, default=8)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "render":
        cohort = compute_cohort(args.assignments, args.seed, args.per_cluster)
        fault_config = render_fault_config(args.fault_base, args.seed, cohort)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        fault_config_path = args.output_dir / f"s{args.seed}_fault.yaml"
        fault_config_path.write_text(yaml.safe_dump(fault_config, sort_keys=False), encoding="utf-8")
        # Absolute, not repo-relative: this file's location tracks wherever
        # --output-dir (driven by the sweep script's OUTPUT_ROOT) actually
        # is, so embedding anything CWD-relative here would silently break
        # the moment OUTPUT_ROOT differs from the script's default.
        fault_config_path_absolute = str(fault_config_path.resolve())
        written = [str(fault_config_path)]
        for epochs in args.epochs:
            scenario_config = render_scenario_config(
                args.scenario_template, seed=args.seed, epochs=epochs,
                fault_config_path=fault_config_path_absolute,
            )
            scenario_path = args.output_dir / f"s{args.seed}_epoch{int(epochs):02d}.yaml"
            scenario_path.write_text(yaml.safe_dump(scenario_config, sort_keys=False), encoding="utf-8")
            written.append(str(scenario_path))
        print("\n".join(written))
        return 0

    if args.command == "relabel-cohort":
        cohort = compute_cohort(args.assignments, args.seed, args.per_cluster)
        result = relabel_cohort_as_fault(args.preparation_root, cohort)
        print(result)
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
