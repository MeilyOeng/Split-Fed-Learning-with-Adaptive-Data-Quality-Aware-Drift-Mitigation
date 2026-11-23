"""Condition-level (5-seed) mitigation report builder.

:mod:`drift.mitigation_report` already implements every table, plot, and
Markdown section a cross-seed mitigation report needs; its ``main()``
assumes one ``suite_root`` containing sibling ``runseed<N>/`` and
``run2_dataseed<N>/`` directories for every seed. The experiment matrix
instead prepares and runs each seed as its own independent scenario (seed
is part of scenario identity, see paths.scenario_id), so this module's only
job is to materialize that legacy-shaped view -- via symlinks into
``paths.condition_staging_dir()``, never by copying data -- and then
delegate everything else to ``drift.mitigation_report``'s existing
loaders/table-builders/plotters/``render_report``, replicating its
``main()`` body exactly (same call sequence, same functions) so a
condition report can never drift from what the standalone tool produces
for a legacy-shaped suite.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ... import mitigation_report as report
from . import paths


def _stage_condition(
    output_root: Path, *, drift_type: str, severity: str, drift_count: int, epochs: int, seeds: tuple[int, ...],
) -> Path:
    staging_dir = paths.condition_staging_dir(output_root, drift_count, drift_type, severity, epochs)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    for seed in seeds:
        scen_id = paths.scenario_id(seed, drift_type, severity, drift_count)
        seed_suite_root = paths.suite_root(output_root, drift_count, drift_type, scen_id, epochs)
        seed_prepared_root = paths.prepared_root(output_root, drift_count, drift_type, scen_id)
        acceptance_path = seed_suite_root / "comparison" / "acceptance_report.json"
        if not acceptance_path.is_file():
            raise FileNotFoundError(
                f"seed {seed} suite is not evaluated yet (missing {acceptance_path}); a condition "
                f"report requires every seed's suite to reach the 'evaluated' state first"
            )
        (staging_dir / f"runseed{seed}").symlink_to(seed_suite_root, target_is_directory=True)
        (staging_dir / f"run2_dataseed{seed}").symlink_to(seed_prepared_root, target_is_directory=True)
    return staging_dir


def build_condition_report(
    protocol: dict[str, Any],
    output_root: str | Path,
    *,
    drift_type: str,
    severity: str,
    drift_count: int,
    epochs: int,
    seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    seeds = tuple(seeds) if seeds is not None else tuple(protocol["seeds"])
    staging_dir = _stage_condition(
        output_root, drift_type=drift_type, severity=severity, drift_count=drift_count, epochs=epochs, seeds=seeds,
    )
    condition_id = paths.condition_id(drift_type, severity, drift_count, epochs)
    report_dir = paths.condition_reports_dir(output_root, drift_count, drift_type, severity, epochs)
    tables_dir, plots_dir = report_dir / "tables", report_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -- mirrors drift.mitigation_report.main() exactly, staging_dir standing
    # in for its --suite-root and report_dir for its --output. --
    seed_runs = report.discover_seed_runs(staging_dir)
    affected_clients = report.load_affected_clients(seed_runs)
    retention_clients = report.load_retention_clients(seed_runs)
    client_benefits = report.build_client_benefits(affected_clients)
    control_clients = report.load_control_clients(seed_runs)
    control_impact = report.build_control_impact(control_clients)
    tables = report.build_tables(seed_runs, affected_clients, retention_clients, client_benefits, control_impact)
    for name, frame in tables.items():
        frame.to_csv(tables_dir / f"{name}.csv", index=False)

    recovery_mae = tables["recovery_by_seed"]
    recovery_mae = recovery_mae[recovery_mae.metric == "mae"]
    client_stats = report.load_affected_client_stats(affected_clients)

    client_scales = report.load_client_scales(seed_runs)
    affected_clients_scaled = report.attach_nmae(affected_clients, client_scales)
    nmae_stats = report.load_affected_client_nmae_stats(affected_clients_scaled)
    tables["nmae_by_seed"] = nmae_stats.round(4)
    tables["nmae_aggregate"] = report.summarise_nmae_aggregate(nmae_stats)
    tables["nmae_by_seed"].to_csv(tables_dir / "nmae_by_seed.csv", index=False)
    tables["nmae_aggregate"].to_csv(tables_dir / "nmae_aggregate.csv", index=False)

    captions = {
        "affected_mae": report.plot_recovery_by_seed(client_stats, plots_dir),
        "affected_mae_distribution": report.plot_mae_distribution_by_seed(client_stats, plots_dir),
        "affected_nmae": report.plot_nmae_by_seed(nmae_stats, plots_dir),
        "recovery_rate": report.plot_recovery_rate_by_seed(recovery_mae, plots_dir),
        "client_benefit": report.plot_client_benefit_distribution(client_benefits, plots_dir),
        "control_impact": report.plot_control_impact_distribution(control_impact, plots_dir),
        "retention": report.plot_retention_distribution(retention_clients, plots_dir),
        "confound": report.plot_continued_training_confound(tables["continued_training_control_by_seed"], plots_dir),
        "heterogeneity": report.plot_cohort_heterogeneity(
            tables["cluster_r0_heterogeneity"], tables["affected_r0_by_seed_cluster"], plots_dir
        ),
    }
    for legacy_stem in ("recovery_rate_boxplot", "cluster_heterogeneity"):
        for suffix in (".png", ".pdf"):
            (plots_dir / f"{legacy_stem}{suffix}").unlink(missing_ok=True)

    client_index = report.build_all_client_plots(seed_runs, plots_dir)
    report.render_report(
        Path(condition_id), seed_runs, tables, retention_clients, client_index, report_dir, plots_dir, captions,
    )
    return {"condition_id": condition_id, "report_dir": str(report_dir), "seeds": sorted(seed_runs)}
