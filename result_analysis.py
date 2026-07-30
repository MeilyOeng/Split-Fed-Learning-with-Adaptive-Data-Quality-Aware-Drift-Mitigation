#!/usr/bin/env python3
"""
result_analysis.py

Compares baseline (clean-data reference) against several faulty-data training
configs (faulty_noscore, faulty_finalscore, faulty_weightAccscore, faulty_5dimsscore, ...)
across 84 clients, 4 of which had faults injected.

Two comparison frames are computed for every client / metric:

  1. COST frame:      baseline_to_<config>   -> "how far does this config fall from the
                       clean-data ideal?" Used mainly on the CLEAN group (80 clients) to
                       measure how much a config costs clean clients.

  2. RECOVERY frame:  faulty_noscore_to_<config> -> "does this config actually help,
                       compared to doing nothing (no score)?" Used mainly on the FAULTY
                       group (4 clients) to measure real recovery, since faulty_noscore
                       is the fairest do-nothing baseline for faulty data (unlike
                       `baseline`, which is trained on clean data and can never be fully
                       reached by any faulty-data config).

Run this from the project root (same folder as before). Requires matplotlib for plots
(pip install matplotlib --break-system-packages) -- if unavailable, plotting is skipped
gracefully and everything else still runs.
"""

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

# Increase CSV field size limit for large JSON/text columns
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int //= 10

# Optional deps -- degrade gracefully if missing
try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


# =====================================================================
# CONFIG
# =====================================================================
ROOT = Path(__file__).resolve().parent
EXPERIMENT = "outputs/run4/E3Seed42"

OUTPUTS = {
    "baseline": ROOT / "outputs" / "baseline-bs128",
    "faulty_noscore": ROOT / EXPERIMENT / "faulty-noscore",
    "faulty_finalscore": ROOT / EXPERIMENT / "faulty-input-finalscore",
    "faulty_weightAccscore": ROOT / EXPERIMENT / "faulty-weightAcc-score",
    "faulty_weightAccscore_and_final_score":  ROOT / EXPERIMENT / "faulty-input-finalscore-with-weight",
    "faulty_weightAccscore_nosoft":  ROOT / EXPERIMENT / "faulty-weightAcc-score-nosoft",
    "faulty_weightLoss":  ROOT / EXPERIMENT / "faulty-weightLoss",
    "faulty_weightAnchor": ROOT / EXPERIMENT / "faulty-anchor-only" }
#     "faulty_5dimsscore": ROOT / EXPERIMENT / "faulty-5dims-score",
# }

RUN_DIR = ROOT / "dataset" / "run4" / "E3Seed42"
QUALITY_DIR = RUN_DIR / "quality_output"
REPORT_DIR = ROOT / EXPERIMENT / "report"
PLOTS_DIR = REPORT_DIR / "plots"
FAULT_SUMMARY_PATH = QUALITY_DIR / "fault_injection_report_summary.csv"

DELTA_EPS = 1e-6

METRICS = ("mae", "rmse", "mse", "r2", "smape")
LOWER_BETTER = {"mae", "rmse", "mse", "smape"}

REFERENCE = "baseline"
NOSCORE_REFERENCE = "faulty_noscore"

# Metrics to actually generate plots for (keeps file count sane -- add more if wanted)
PLOT_METRICS = ("mae", "r2")

# Signals checked against faulty-client recovery magnitude (targeting analysis)
TARGETING_SIGNALS = ("client_pca_quality_score", "achieved_faulty_fraction", "window_quality_all_mean")


def experiment_label():
    """Human-readable label for the configured experiment output path."""
    path = Path(EXPERIMENT)
    parts = path.parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return str(path)


# =====================================================================
# UTILITIES
# =====================================================================
def fnum(value):
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value, digits=4):
    value = fnum(value)
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_pct(value, digits=1):
    value = fnum(value)
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}%"


def safe_div(num, den):
    num = fnum(num)
    den = fnum(den)
    if num is None or den in (None, 0.0):
        return None
    return num / den


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    clean = [fnum(v) for v in values]
    clean = [v for v in clean if v is not None]
    if not clean:
        return {}
    ordered = sorted(clean)

    def pct(q):
        if not ordered:
            return None
        pos = (len(ordered) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)

    avg = mean(clean)
    var = mean([(v - avg) ** 2 for v in clean]) if clean else None
    return {
        "count": len(clean),
        "mean": avg,
        "median": median(clean),
        "std": math.sqrt(var) if var is not None else None,
        "min": min(clean),
        "p10": pct(0.10),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": max(clean),
    }


def signed_delta(metric, old, new):
    old = fnum(old)
    new = fnum(new)
    if old is None or new is None:
        return None
    if metric in LOWER_BETTER:
        return old - new  # positive = improvement (error decreased)
    else:
        return new - old  # positive = improvement (score increased)


def rank_values(values):
    """Average-rank transform (1-indexed), ties get the mean of their tied ranks."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(x_values, y_values):
    """Returns (rho, p_value, n). p_value is None if scipy isn't available."""
    pairs = [(x, y) for x, y in zip(x_values, y_values) if fnum(x) is not None and fnum(y) is not None]
    n = len(pairs)
    if n < 3:
        return None, None, n
    xs = [fnum(p[0]) for p in pairs]
    ys = [fnum(p[1]) for p in pairs]

    if HAVE_SCIPY:
        rho, p = scipy_stats.spearmanr(xs, ys)
        return (rho if math.isfinite(rho) else None), p, n

    rx, ry = rank_values(xs), rank_values(ys)
    mrx, mry = mean(rx), mean(ry)
    cov = sum((a - mrx) * (b - mry) for a, b in zip(rx, ry))
    var_x = sum((a - mrx) ** 2 for a in rx)
    var_y = sum((b - mry) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return None, None, n
    rho = cov / math.sqrt(var_x * var_y)
    return rho, None, n


def paired_wilcoxon(deltas):
    """Signed-rank test that the deltas differ from 0. Returns (statistic, p_value)."""
    clean = [fnum(v) for v in deltas]
    clean = [v for v in clean if v is not None and v != 0]
    if not HAVE_SCIPY or len(clean) < 6:
        return None, None
    try:
        stat, p = scipy_stats.wilcoxon(clean)
        return stat, p
    except ValueError:
        return None, None


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


# =====================================================================
# LOADERS
# =====================================================================
def load_output_metrics():
    metrics_rows = []
    by_config_client = {}
    for config_name, root in OUTPUTS.items():
        if not root.exists():
            continue
        for path in sorted(root.glob("cluster_*/client_*_test_metrics.json")):
            payload = json.loads(path.read_text())
            client_id = payload.get("client_id")
            if client_id is None:
                stem = path.stem
                parts = stem.split("_")
                client_id = parts[1] if len(parts) >= 2 else None
            client_id = str(client_id)

            cluster_id = payload.get("cluster_id")
            if cluster_id is None:
                cluster_id = path.parent.name.replace("cluster_", "")
            cluster_id = str(cluster_id)

            raw_metrics = payload.get("raw_metrics", {})
            norm_metrics = payload.get("normalized_metrics", {})

            row = {
                "config": config_name,
                "output_dir": str(root),
                "cluster_id": cluster_id,
                "client_id": client_id,
                "dq_mode": payload.get("dq_mode"),
                "use_quality_as_input": payload.get("use_quality_as_input"),
                "use_quality_weights": payload.get("use_quality_weights"),
                "use_quality_aggregation_weights": payload.get("use_quality_aggregation_weights"),
                "num_samples": payload.get("num_samples"),
                "num_batches": payload.get("num_batches"),
            }
            for metric in METRICS:
                row[f"raw_{metric}"] = fnum(raw_metrics.get(metric))
                row[f"norm_{metric}"] = fnum(norm_metrics.get(metric))

            metrics_rows.append(row)
            by_config_client[(config_name, client_id)] = row

    # Loud warning (not a crash) for any config that produced no rows -- e.g. a run
    # that's still in progress or a folder path that's wrong/missing.
    for config_name in OUTPUTS:
        count = sum(1 for r in metrics_rows if r["config"] == config_name)
        if count == 0:
            print(f"WARNING: '{config_name}' produced 0 rows -- check the run has finished "
                  f"and OUTPUTS['{config_name}'] points at the right folder.")

    # Catch mislabeled folders: every file inside one config's folder should share the
    # same dq_mode / quality-usage flags.
    for config_name in OUTPUTS:
        sigs = {
            (r["dq_mode"], r["use_quality_as_input"], r["use_quality_weights"], r["use_quality_aggregation_weights"])
            for r in metrics_rows if r["config"] == config_name
        }
        assert len(sigs) <= 1, f"'{config_name}' has mixed config signatures: {sigs}"

    return metrics_rows, by_config_client


def load_faults(cluster_by_client):
    rows = read_csv(FAULT_SUMMARY_PATH)
    print("Loaded fault summary rows:", len(rows))

    fault_summary = {}
    for row in rows:
        client_id = (row.get("client_id") or "").strip()
        if not client_id:
            continue

        fault_summary[client_id] = {
            "client_id": client_id,
            "cluster_id": row.get("cluster_id") or cluster_by_client.get(client_id, ""),
            "is_fault_injected": True,
            "achieved_faulty_fraction": fnum(row.get("achieved_faulty_fraction")),
            "number_of_windows": fnum(row.get("number_of_windows")),
            "selected_faulty_windows": fnum(row.get("selected_faulty_windows")),
            "windows_allocated_completeness": fnum(row.get("windows_allocated_completeness")),
            "fault_count_completeness": fnum(row.get("fault_count_completeness")),
            "windows_allocated_consistency": fnum(row.get("windows_allocated_consistency")),
            "fault_count_consistency": fnum(row.get("fault_count_consistency")),
            "windows_allocated_accuracy": fnum(row.get("windows_allocated_accuracy")),
            "fault_count_accuracy": fnum(row.get("fault_count_accuracy")),
            "windows_allocated_validity": fnum(row.get("windows_allocated_validity")),
            "fault_count_validity": fnum(row.get("fault_count_validity")),
            "windows_allocated_mixed": fnum(row.get("windows_allocated_mixed")),
            "fault_count_mixed": fnum(row.get("fault_count_mixed")),
            "fault_count_cluster": fnum(row.get("fault_count_cluster")),
        }

    run_meta = {}
    if rows:
        first = rows[0]
        run_meta = {
            "client_selection_mode": first.get("client_selection_mode"),
            "client_selection_seed": first.get("client_selection_seed"),
            "selected_client_count": fnum(first.get("selected_client_count")),
            "eligible_client_count": fnum(first.get("eligible_client_count")),
            "assigned_client_count": fnum(first.get("assigned_client_count")),
            "unassigned_client_count": fnum(first.get("unassigned_client_count")),
        }

    return fault_summary, run_meta


def load_quality_scores():
    quality = defaultdict(dict)

    pca_rows = read_csv(QUALITY_DIR / "client_pca_quality_scores.csv")
    print("PCA rows:", len(pca_rows))
    for row in pca_rows:
        client_id = (row.get("client_id") or "").strip()
        if not client_id:
            continue
        quality[client_id]["client_pca_client_id"] = client_id
        for key, value in row.items():
            if key == "client_id":
                continue
            converted = fnum(value)
            quality[client_id][f"client_pca_{key}"] = converted if converted is not None else value

    window_rows = read_csv(QUALITY_DIR / "window_quality_scores.csv")
    print("Window rows:", len(window_rows))
    window_by_client_mode = defaultdict(lambda: defaultdict(list))
    for row in window_rows:
        client_id = (row.get("client_id") or "").strip()
        if not client_id:
            continue
        mode = (row.get("mode") or "unknown").strip()
        score = fnum(row.get("quality_score"))
        if score is not None:
            window_by_client_mode[client_id][mode].append(score)

    for client_id, modes in window_by_client_mode.items():
        for mode, scores in modes.items():
            s = stats(scores)
            quality[client_id][f"window_quality_{mode}_mean"] = s.get("mean")
            quality[client_id][f"window_quality_{mode}_min"] = s.get("min")
            quality[client_id][f"window_quality_{mode}_p10"] = s.get("p10")

        all_scores = [score for scores in modes.values() for score in scores]
        s = stats(all_scores)
        quality[client_id]["window_quality_all_mean"] = s.get("mean")
        quality[client_id]["window_quality_all_min"] = s.get("min")
        quality[client_id]["window_quality_all_p10"] = s.get("p10")

    return dict(quality)


# =====================================================================
# TABLE BUILDERS
# =====================================================================
def make_per_client_master(metrics_rows, by_config_client, fault_summary, quality, present_configs):
    master = []
    client_ids = sorted({r["client_id"] for r in metrics_rows if r["config"] == REFERENCE})

    for client_id in client_ids:
        baseline_row = by_config_client.get((REFERENCE, client_id), {})
        row = {"client_id": client_id, "cluster_id": baseline_row.get("cluster_id", "")}

        if client_id in fault_summary:
            row.update(fault_summary[client_id])
        else:
            row.update({
                "is_fault_injected": False,
                "achieved_faulty_fraction": 0.0,
                "number_of_windows": None,
                "selected_faulty_windows": 0.0,
                "windows_allocated_completeness": 0.0, "fault_count_completeness": 0.0,
                "windows_allocated_consistency": 0.0, "fault_count_consistency": 0.0,
                "windows_allocated_accuracy": 0.0, "fault_count_accuracy": 0.0,
                "windows_allocated_validity": 0.0, "fault_count_validity": 0.0,
                "windows_allocated_mixed": 0.0, "fault_count_mixed": 0.0,
                "fault_count_cluster": 0.0,
            })

        row.update(quality.get(client_id, {}))

        for config_name in present_configs:
            metric_row = by_config_client.get((config_name, client_id), {})
            for metric in METRICS:
                row[f"{config_name}_raw_{metric}"] = metric_row.get(f"raw_{metric}")
                row[f"{config_name}_norm_{metric}"] = metric_row.get(f"norm_{metric}")
            row[f"{config_name}_num_samples"] = metric_row.get("num_samples")

        master.append(row)

    return master


def add_pairwise_deltas(per_client, config_pairs):
    for row in per_client:
        for (ref, cfg) in config_pairs:
            for metric in METRICS:
                old_value = row.get(f"{ref}_raw_{metric}")
                new_value = row.get(f"{cfg}_raw_{metric}")
                delta = signed_delta(metric, old_value, new_value)

                old_num = fnum(old_value)
                denom = abs(old_num) if old_num is not None else None
                pct_ratio = safe_div(delta, denom)
                pct = pct_ratio * 100 if pct_ratio is not None else None

                row[f"{ref}_to_{cfg}_{metric}_delta"] = delta
                row[f"{ref}_to_{cfg}_{metric}_delta_pct"] = pct
    return per_client


def build_group_comparison_summary(per_client, config_pairs):
    output_rows = []
    for (ref, cfg) in config_pairs:
        for metric in METRICS:
            groups = [
                ("clean", [r for r in per_client if not r["is_fault_injected"]]),
                ("faulty", [r for r in per_client if r["is_fault_injected"]]),
            ]
            for group_name, group_rows in groups:
                delta_key = f"{ref}_to_{cfg}_{metric}_delta"
                raw_values = [r.get(delta_key) for r in group_rows]
                clean_values = [v for v in (fnum(v) for v in raw_values) if v is not None]

                n_improved = sum(1 for v in clean_values if v > DELTA_EPS)
                n_degraded = sum(1 for v in clean_values if v < -DELTA_EPS)
                n_unchanged = len(clean_values) - n_improved - n_degraded
                total = len(clean_values)

                pct_improved = safe_div(n_improved, total)
                pct_degraded = safe_div(n_degraded, total)
                pct_unchanged = safe_div(n_unchanged, total)
                pct_improved = pct_improved * 100 if pct_improved is not None else None
                pct_degraded = pct_degraded * 100 if pct_degraded is not None else None
                pct_unchanged = pct_unchanged * 100 if pct_unchanged is not None else None

                s = stats(raw_values)
                wilcoxon_stat, wilcoxon_p = paired_wilcoxon(raw_values)

                output_rows.append({
                    "comparison_pair": f"{ref}_to_{cfg}",
                    "metric": metric,
                    "group": group_name,
                    "n": s.get("count"),
                    "mean_delta": s.get("mean"),
                    "median_delta": s.get("median"),
                    "std_delta": s.get("std"),
                    "p10_delta": s.get("p10"),
                    "p90_delta": s.get("p90"),
                    "p95_delta": s.get("p95"),
                    "min_delta": s.get("min"),
                    "max_delta": s.get("max"),
                    "n_improved": n_improved,
                    "n_degraded": n_degraded,
                    "n_unchanged": n_unchanged,
                    "pct_improved": pct_improved,
                    "pct_degraded": pct_degraded,
                    "pct_unchanged": pct_unchanged,
                    "wilcoxon_p": wilcoxon_p,
                })
    return output_rows


def build_tradeoff_summary(per_client, recovery_configs, reference=REFERENCE, noscore_ref=NOSCORE_REFERENCE):
    """
    One row per (config, metric). config ranges over every scored config EXCEPT noscore
    itself (noscore is the do-nothing reference for recovery).

    cost      = mean baseline_to_<config> delta on the CLEAN group
                (how much clean-client performance drops vs the clean-data ideal)
    recovery  = mean noscore_to_<config> delta on the FAULTY group
                (how much better this config does than doing nothing, for the clients
                that actually had faults injected)
    """
    clean_rows = [r for r in per_client if not r["is_fault_injected"]]
    faulty_rows = [r for r in per_client if r["is_fault_injected"]]

    rows = []
    for cfg in recovery_configs:
        for metric in METRICS:
            cost_key = f"{reference}_to_{cfg}_{metric}_delta"
            cost_stats = stats([r.get(cost_key) for r in clean_rows])

            recovery_key = f"{noscore_ref}_to_{cfg}_{metric}_delta"
            recovery_stats = stats([r.get(recovery_key) for r in faulty_rows])

            # context: how far this config's faulty clients still sit from the clean ideal
            still_gap_key = f"{reference}_to_{cfg}_{metric}_delta"
            still_gap_stats = stats([r.get(still_gap_key) for r in faulty_rows])

            cost = cost_stats.get("mean")
            recovery = recovery_stats.get("mean")
            ratio = safe_div(abs(recovery) if recovery is not None else None,
                              abs(cost) if cost is not None else None)
            net = (cost + recovery) if (cost is not None and recovery is not None) else None

            rows.append({
                "config": cfg,
                "metric": metric,
                "n_clean": cost_stats.get("count"),
                "n_faulty": recovery_stats.get("count"),
                "cost_mean_delta_clean_vs_baseline": cost,
                "cost_median_delta_clean_vs_baseline": cost_stats.get("median"),
                "recovery_mean_delta_faulty_vs_noscore": recovery,
                "recovery_median_delta_faulty_vs_noscore": recovery_stats.get("median"),
                "still_below_baseline_mean_delta_faulty": still_gap_stats.get("mean"),
                "recovery_to_cost_ratio": ratio,
                "net_score": net,
            })
    return rows


def build_targeting_correlations(per_client, recovery_pairs, signal_cols=TARGETING_SIGNALS):
    faulty_rows = [r for r in per_client if r["is_fault_injected"]]
    rows = []
    for (ref, cfg) in recovery_pairs:
        for metric in METRICS:
            delta_key = f"{ref}_to_{cfg}_{metric}_delta"
            deltas = [r.get(delta_key) for r in faulty_rows]
            for signal in signal_cols:
                signal_values = [r.get(signal) for r in faulty_rows]
                rho, p, n = spearman(signal_values, deltas)
                rows.append({
                    "comparison_pair": f"{ref}_to_{cfg}",
                    "metric": metric,
                    "signal": signal,
                    "n": n,
                    "spearman_rho": rho,
                    "p_value": p,
                })
    return rows


def performance_disparity(metric, clean_value, faulty_value):
    """Positive means faulty clients perform worse than clean clients."""
    clean_value = fnum(clean_value)
    faulty_value = fnum(faulty_value)
    if clean_value is None or faulty_value is None:
        return None
    if metric in LOWER_BETTER:
        return faulty_value - clean_value
    return clean_value - faulty_value


def build_faulty_client_recovery(per_client, recovery_configs,
                                 reference=REFERENCE, noscore_ref=NOSCORE_REFERENCE):
    """One row per faulty client, recovery config, and metric."""
    rows = []
    for row in per_client:
        if not row.get("is_fault_injected"):
            continue
        for cfg in recovery_configs:
            for metric in METRICS:
                baseline_value = fnum(row.get(f"{reference}_raw_{metric}"))
                noscore_value = fnum(row.get(f"{noscore_ref}_raw_{metric}"))
                method_value = fnum(row.get(f"{cfg}_raw_{metric}"))
                recovery = signed_delta(metric, noscore_value, method_value)
                remaining_gap = signed_delta(metric, baseline_value, method_value)
                original_damage = signed_delta(metric, baseline_value, noscore_value)
                recovered_fraction = None
                if recovery is not None and original_damage is not None and original_damage < -DELTA_EPS:
                    recovered_fraction = recovery / abs(original_damage)
                rows.append({
                    "client_id": row["client_id"],
                    "cluster_id": row.get("cluster_id"),
                    "config": cfg,
                    "metric": metric,
                    "baseline_value": baseline_value,
                    "faulty_noscore_value": noscore_value,
                    "method_value": method_value,
                    "recovery_delta": recovery,
                    "remaining_gap_delta_vs_baseline": remaining_gap,
                    "original_fault_damage_delta": original_damage,
                    "recovered_fraction": recovered_fraction,
                    "recovered_percent": recovered_fraction * 100 if recovered_fraction is not None else None,
                    "client_pca_quality_score": row.get("client_pca_quality_score"),
                    "window_quality_all_mean": row.get("window_quality_all_mean"),
                    "achieved_faulty_fraction": row.get("achieved_faulty_fraction"),
                })
    return rows


def build_fairness_gap_summary(per_client, present_configs):
    """Clean-vs-faulty disparity for each configuration and metric."""
    clean_rows = [r for r in per_client if not r.get("is_fault_injected")]
    faulty_rows = [r for r in per_client if r.get("is_fault_injected")]
    rows = []
    for cfg in sorted(present_configs):
        for metric in METRICS:
            clean_stats = stats([r.get(f"{cfg}_raw_{metric}") for r in clean_rows])
            faulty_stats = stats([r.get(f"{cfg}_raw_{metric}") for r in faulty_rows])
            clean_mean = clean_stats.get("mean")
            faulty_mean = faulty_stats.get("mean")
            disparity = performance_disparity(metric, clean_mean, faulty_mean)
            normalized_gap = safe_div(disparity, abs(clean_mean) if clean_mean is not None else None)
            rows.append({
                "config": cfg,
                "metric": metric,
                "n_clean": clean_stats.get("count"),
                "n_faulty": faulty_stats.get("count"),
                "clean_mean": clean_mean,
                "faulty_mean": faulty_mean,
                "clean_median": clean_stats.get("median"),
                "faulty_median": faulty_stats.get("median"),
                "performance_gap_faulty_worse_positive": disparity,
                "normalized_gap": normalized_gap,
                "normalized_gap_percent": normalized_gap * 100 if normalized_gap is not None else None,
            })
    return rows


def build_tail_risk_summary(per_client, present_configs):
    """Median, upper tail, and worst-client metrics for overall/clean/faulty groups."""
    groups = {
        "overall": per_client,
        "clean": [r for r in per_client if not r.get("is_fault_injected")],
        "faulty": [r for r in per_client if r.get("is_fault_injected")],
    }
    rows = []
    for cfg in sorted(present_configs):
        for group_name, group_rows in groups.items():
            for metric in METRICS:
                values = [r.get(f"{cfg}_raw_{metric}") for r in group_rows]
                s = stats(values)
                valid = [(fnum(r.get(f"{cfg}_raw_{metric}")), r["client_id"])
                         for r in group_rows if fnum(r.get(f"{cfg}_raw_{metric}")) is not None]
                worst_client = None
                worst_value = None
                if valid:
                    if metric in LOWER_BETTER:
                        worst_value, worst_client = max(valid, key=lambda x: x[0])
                    else:
                        worst_value, worst_client = min(valid, key=lambda x: x[0])
                rows.append({
                    "config": cfg,
                    "group": group_name,
                    "metric": metric,
                    "n": s.get("count"),
                    "mean": s.get("mean"),
                    "median": s.get("median"),
                    "std": s.get("std"),
                    "p90": s.get("p90"),
                    "p95": s.get("p95"),
                    "worst_value": worst_value,
                    "worst_client_id": worst_client,
                })
    return rows


TRAIN_STEP_QUALITY_RE = re.compile(
    r"train_step\s+cluster=(?P<cluster>\S+)\s+round_id=(?P<source_round>\d+)\s+client=(?P<client>\S+)"
    r".*?quality_mean=(?P<quality>[-+0-9.eE]+)"
    r".*?soft_quality=(?P<soft_quality>[-+0-9.eE]+)"
    r".*?round_complete=(?P<round_complete>True|False)"
)


def _round_quality_row(config_name, cluster_id, round_number, source_round_id, bucket, faulty_clients):
    clients = bucket["clients"]
    q_stats = stats(bucket["quality"])
    return {
        "Configuration": config_name,
        "Round": round_number,
        "Cluster": cluster_id,
        "Clients": len(clients),
        "Faulty clients": sum(1 for client_id in clients if client_id in faulty_clients),
        "Mean quality": q_stats.get("mean"),
        "p10 quality": q_stats.get("p10"),
        "Std quality": q_stats.get("std"),
        "Effective weight": sum(bucket["soft_quality"]),
        "Source round id": source_round_id,
    }


def build_round_quality_summary(fault_summary):
    """Aggregate server-side batch quality and effective aggregation weights per completed round."""
    faulty_clients = set(fault_summary)
    rows = []

    for config_name, root in OUTPUTS.items():
        if not root.exists():
            continue
        for log_path in sorted(root.glob("cluster_*/server.log")):
            round_number = 0
            bucket = {"clients": set(), "quality": [], "soft_quality": []}
            bucket_cluster = None
            bucket_source_round = None
            try:
                handle = log_path.open("r", encoding="utf-8", errors="replace")
            except OSError:
                continue
            with handle:
                for line in handle:
                    match = TRAIN_STEP_QUALITY_RE.search(line)
                    if not match:
                        continue
                    quality = fnum(match.group("quality"))
                    soft_quality = fnum(match.group("soft_quality"))
                    if quality is None or soft_quality is None:
                        continue

                    cluster_id = match.group("cluster")
                    source_round_id = int(match.group("source_round"))
                    if bucket["clients"] and (
                        bucket_cluster != cluster_id or bucket_source_round != source_round_id
                    ):
                        round_number += 1
                        rows.append(
                            _round_quality_row(
                                config_name,
                                bucket_cluster,
                                round_number,
                                bucket_source_round,
                                bucket,
                                faulty_clients,
                            )
                        )
                        bucket = {"clients": set(), "quality": [], "soft_quality": []}

                    bucket_cluster = cluster_id
                    bucket_source_round = source_round_id
                    bucket["clients"].add(match.group("client"))
                    bucket["quality"].append(quality)
                    bucket["soft_quality"].append(soft_quality)

                    if match.group("round_complete") == "True":
                        round_number += 1
                        rows.append(
                            _round_quality_row(
                                config_name,
                                bucket_cluster,
                                round_number,
                                bucket_source_round,
                                bucket,
                                faulty_clients,
                            )
                        )
                        bucket = {"clients": set(), "quality": [], "soft_quality": []}
                        bucket_cluster = None
                        bucket_source_round = None

            if bucket["clients"]:
                round_number += 1
                rows.append(
                    _round_quality_row(
                        config_name,
                        bucket_cluster,
                        round_number,
                        bucket_source_round,
                        bucket,
                        faulty_clients,
                    )
                )

    return rows


def summarize_group(rows, group_keys, metric_prefix="raw_"):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in group_keys)].append(row)

    out = []
    for key, items in sorted(groups.items(), key=lambda kv: kv[0]):
        base = {group_keys[i]: key[i] for i in range(len(group_keys))}
        base["num_clients"] = len({r["client_id"] for r in items})
        for metric in METRICS:
            s = stats([r.get(f"{metric_prefix}{metric}") for r in items])
            for name, value in s.items():
                base[f"{metric}_{name}"] = value

            def val(r):
                v = fnum(r.get(f"{metric_prefix}{metric}"))
                return v

            scored = [(val(r), r) for r in items if val(r) is not None]
            if scored:
                if metric == "r2":
                    best = max(scored, key=lambda t: t[0])[1]
                    worst = min(scored, key=lambda t: t[0])[1]
                else:
                    best = min(scored, key=lambda t: t[0])[1]
                    worst = max(scored, key=lambda t: t[0])[1]
                base[f"best_{metric}_client"] = best["client_id"]
                base[f"worst_{metric}_client"] = worst["client_id"]
        out.append(base)
    return out


# =====================================================================
# PLOTS
# =====================================================================
def _savefig(fig, name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path.name


def plot_group_split_boxplot(per_client, recovery_pairs, metric):
    if not HAVE_MPL:
        return None
    fig, axes = plt.subplots(1, len(recovery_pairs), figsize=(4.5 * len(recovery_pairs), 4), squeeze=False)
    axes = axes[0]
    for ax, (ref, cfg) in zip(axes, recovery_pairs):
        key = f"{ref}_to_{cfg}_{metric}_delta"
        clean_vals = [v for v in (r.get(key) for r in per_client if not r["is_fault_injected"]) if v is not None]
        faulty_vals = [v for v in (r.get(key) for r in per_client if r["is_fault_injected"]) if v is not None]
        ax.boxplot([clean_vals, faulty_vals], tick_labels=["clean", "faulty"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"{ref}\n-> {cfg}", fontsize=9)
        ax.set_ylabel(f"{metric} delta (+ = better)")
    fig.suptitle(f"Clean vs faulty delta distribution -- {metric}")
    fig.tight_layout()
    return _savefig(fig, f"boxplot_{metric}")


def plot_client_trajectory(per_client, configs, metric):
    if not HAVE_MPL:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    x = list(range(len(configs)))
    for row in per_client:
        ys = [row.get(f"{cfg}_raw_{metric}") for cfg in configs]
        if any(v is None for v in ys):
            continue
        color = "tab:red" if row["is_fault_injected"] else "tab:blue"
        alpha = 0.9 if row["is_fault_injected"] else 0.15
        lw = 2.0 if row["is_fault_injected"] else 0.8
        ax.plot(x, ys, color=color, alpha=alpha, linewidth=lw)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylabel(f"raw {metric}")
    ax.set_title(f"Per-client trajectory across configs -- {metric}\n(red = faulty clients, blue = clean)")
    fig.tight_layout()
    return _savefig(fig, f"trajectory_{metric}")


def plot_tradeoff_scatter(tradeoff_rows, metric):
    if not HAVE_MPL:
        return None
    rows = [r for r in tradeoff_rows if r["metric"] == metric]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in rows:
        cost = fnum(r.get("cost_mean_delta_clean_vs_baseline"))
        recovery = fnum(r.get("recovery_mean_delta_faulty_vs_noscore"))
        if cost is None or recovery is None:
            continue
        ax.scatter(cost, recovery, s=80)
        ax.annotate(r["config"], (cost, recovery), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("mean cost to clean clients (vs baseline, + = no cost / gain)")
    ax.set_ylabel("mean recovery for faulty clients (vs no-score, + = improvement)")
    ax.set_title(f"Cost vs recovery trade-off -- {metric}")
    fig.tight_layout()
    return _savefig(fig, f"tradeoff_{metric}")


def plot_cluster_heatmap(metrics_rows, present_configs, metric):
    if not HAVE_MPL:
        return None
    clusters = sorted({r["cluster_id"] for r in metrics_rows}, key=lambda x: (len(x), x))
    configs = list(present_configs)
    grid = []
    for cfg in configs:
        row_vals = []
        for cl in clusters:
            vals = [r.get(f"raw_{metric}") for r in metrics_rows if r["config"] == cfg and r["cluster_id"] == cl]
            s = stats(vals)
            row_vals.append(s.get("mean"))
        grid.append(row_vals)

    fig, ax = plt.subplots(figsize=(1.2 * len(clusters) + 2, 0.6 * len(configs) + 2))
    display_grid = [[v if v is not None else float("nan") for v in row] for row in grid]
    im = ax.imshow(display_grid, aspect="auto", cmap="RdYlGn_r" if metric in LOWER_BETTER else "RdYlGn")
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([f"cluster {c}" for c in clusters])
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs)
    for i in range(len(configs)):
        for j in range(len(clusters)):
            v = grid[i][j]
            if v is not None:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=f"mean raw {metric}")
    ax.set_title(f"Cluster x config mean {metric}")
    fig.tight_layout()
    return _savefig(fig, f"heatmap_{metric}")


def plot_faulty_client_recovery(faulty_recovery_rows, recovery_configs, metric):
    if not HAVE_MPL:
        return None
    rows = [r for r in faulty_recovery_rows if r["metric"] == metric]
    clients = sorted({r["client_id"] for r in rows})
    if not rows or not clients:
        return None
    fig, ax = plt.subplots(figsize=(max(7, len(clients) * 1.5), 5))
    width = 0.8 / max(1, len(recovery_configs))
    x = list(range(len(clients)))
    for j, cfg in enumerate(recovery_configs):
        vals = []
        lookup = {(r["client_id"], r["config"]): r for r in rows}
        for client in clients:
            vals.append(fnum(lookup.get((client, cfg), {}).get("recovery_delta")) or 0.0)
        offsets = [i - 0.4 + width / 2 + j * width for i in x]
        ax.bar(offsets, vals, width=width, label=cfg)
    ax.axhline(0, linewidth=0.9, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(clients, rotation=25, ha="right")
    ax.set_ylabel(f"{metric} recovery vs no-score (+ = better)")
    ax.set_title(f"Faulty-client recovery by method -- {metric}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _savefig(fig, f"faulty_client_recovery_{metric}")


def plot_remaining_gap(faulty_recovery_rows, recovery_configs, metric):
    if not HAVE_MPL:
        return None
    rows = [r for r in faulty_recovery_rows if r["metric"] == metric]
    clients = sorted({r["client_id"] for r in rows})
    if not rows or not clients:
        return None
    fig, ax = plt.subplots(figsize=(max(7, len(clients) * 1.5), 5))
    lookup = {(r["client_id"], r["config"]): r for r in rows}
    x = list(range(len(clients)))
    for cfg in recovery_configs:
        vals = [fnum(lookup.get((client, cfg), {}).get("remaining_gap_delta_vs_baseline")) for client in clients]
        ax.plot(x, vals, marker="o", label=cfg)
    ax.axhline(0, linewidth=0.9, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(clients, rotation=25, ha="right")
    ax.set_ylabel(f"delta vs clean baseline (+ = better)")
    ax.set_title(f"Remaining faulty-client gap to clean baseline -- {metric}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _savefig(fig, f"remaining_gap_{metric}")


def plot_fairness_gap(fairness_rows, metric):
    if not HAVE_MPL:
        return None
    rows = [r for r in fairness_rows if r["metric"] == metric]
    if not rows:
        return None
    configs = [r["config"] for r in rows]
    values = [r["performance_gap_faulty_worse_positive"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(7, len(configs) * 1.5), 4.8))
    ax.bar(range(len(configs)), values)
    ax.axhline(0, linewidth=0.9, linestyle="--")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylabel(f"clean-faulty {metric} gap (+ = faulty worse)")
    ax.set_title(f"Clean-faulty performance disparity -- {metric}")
    fig.tight_layout()
    return _savefig(fig, f"fairness_gap_{metric}")


def plot_tail_risk(tail_rows, metric, group="overall"):
    if not HAVE_MPL:
        return None
    rows = [r for r in tail_rows if r["metric"] == metric and r["group"] == group]
    if not rows:
        return None
    configs = [r["config"] for r in rows]
    x = list(range(len(configs)))
    fig, ax = plt.subplots(figsize=(max(7, len(configs) * 1.5), 5))
    ax.plot(x, [r["median"] for r in rows], marker="o", label="median")
    ax.plot(x, [r["p90"] for r in rows], marker="o", label="p90")
    ax.plot(x, [r["p95"] for r in rows], marker="o", label="p95")
    ax.plot(x, [r["worst_value"] for r in rows], marker="o", label="worst client")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylabel(f"raw {metric}")
    ax.set_title(f"Client tail risk across configurations -- {metric} ({group})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _savefig(fig, f"tail_risk_{group}_{metric}")


def _ecdf_xy(values):
    ordered = sorted(v for v in (fnum(value) for value in values) if v is not None)
    if not ordered:
        return [], []
    y = [(idx + 1) / len(ordered) for idx in range(len(ordered))]
    return ordered, y


def plot_metric_ecdf_full_zoom(per_client, present_configs, metric="mae", group="overall", zoom_max=None):
    if not HAVE_MPL:
        return None
    if group == "clean":
        rows = [r for r in per_client if not r.get("is_fault_injected")]
    elif group == "faulty":
        rows = [r for r in per_client if r.get("is_fault_injected")]
    else:
        rows = list(per_client)
    if not rows:
        return None

    config_values = []
    for cfg in sorted(present_configs):
        x, y = _ecdf_xy([r.get(f"{cfg}_raw_{metric}") for r in rows])
        if not x:
            continue
        config_values.append((cfg, x, y))

    if not config_values:
        return None

    all_x = [value for _, x, _ in config_values for value in x]
    if zoom_max is None:
        zoom_max = sorted(all_x)[int(0.90 * (len(all_x) - 1))] if len(all_x) > 1 else max(all_x)
    zoom_min = 0.0 if metric in LOWER_BETTER else min(all_x)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), sharey=True)
    for ax in axes:
        for cfg, x, y in config_values:
            ax.step(x, y, where="post", linewidth=1.8, label=cfg)
        ax.set_xlabel(f"raw {metric.upper()}")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cumulative fraction of clients")
    axes[0].set_title(f"Full range ({group})")
    axes[1].set_xlim(zoom_min, zoom_max)
    axes[1].set_title(f"Zoomed range ({zoom_min:g}-{zoom_max:g})")
    if metric in LOWER_BETTER:
        fig.suptitle(f"ECDF of client {metric.upper()} (left/up is better)")
    else:
        fig.suptitle(f"ECDF of client {metric.upper()} (right/up is better)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return _savefig(fig, f"ecdf_{group}_{metric}_full_zoom")


def plot_delta_ecdf(per_client, reference, configs, metric="mae", group="overall"):
    if not HAVE_MPL or not configs:
        return None
    if group == "clean":
        rows = [r for r in per_client if not r.get("is_fault_injected")]
    elif group == "faulty":
        rows = [r for r in per_client if r.get("is_fault_injected")]
    else:
        rows = list(per_client)
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for cfg in configs:
        deltas = [
            signed_delta(metric, r.get(f"{reference}_raw_{metric}"), r.get(f"{cfg}_raw_{metric}"))
            for r in rows
        ]
        x, y = _ecdf_xy(deltas)
        if not x:
            continue
        ax.step(x, y, where="post", linewidth=1.8, label=f"{cfg} (n={len(x)})")
        plotted = True
    if not plotted:
        plt.close(fig)
        return None

    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel(f"{metric.upper()} improvement vs {reference} (+ = better)")
    ax.set_ylabel("cumulative fraction of clients")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.set_title(f"ECDF of client {metric.upper()} delta vs {reference} -- {group}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _savefig(fig, f"ecdf_delta_vs_{reference}_{group}_{metric}")


def plot_metric_survival(per_client, present_configs, metric="mae", group="overall"):
    if not HAVE_MPL:
        return None
    if group == "clean":
        rows = [r for r in per_client if not r.get("is_fault_injected")]
    elif group == "faulty":
        rows = [r for r in per_client if r.get("is_fault_injected")]
    else:
        rows = list(per_client)
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for cfg in sorted(present_configs):
        values = sorted(
            v for v in (fnum(r.get(f"{cfg}_raw_{metric}")) for r in rows)
            if v is not None
        )
        if not values:
            continue
        n = len(values)
        survival = [1.0 - ((idx + 1) / n) for idx in range(n)]
        ax.step(values, survival, where="post", linewidth=1.8, label=f"{cfg} (n={n})")
        plotted = True
    if not plotted:
        plt.close(fig)
        return None

    ax.set_xlabel(f"raw {metric.upper()}")
    ax.set_ylabel(f"fraction of clients with {metric.upper()} > x")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.set_title(f"Tail / survival curve of client {metric.upper()} -- {group}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _savefig(fig, f"survival_{group}_{metric}")


def plot_targeting_scatter(faulty_recovery_rows, recovery_configs, metric,
                           signal="client_pca_quality_score"):
    if not HAVE_MPL:
        return None
    rows = [r for r in faulty_recovery_rows if r["metric"] == metric]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 5))
    any_points = False
    for cfg in recovery_configs:
        cfg_rows = [r for r in rows if r["config"] == cfg]
        xs = [fnum(r.get(signal)) for r in cfg_rows]
        ys = [fnum(r.get("recovery_delta")) for r in cfg_rows]
        valid = [(x, y, r["client_id"]) for x, y, r in zip(xs, ys, cfg_rows)
                 if x is not None and y is not None]
        if not valid:
            continue
        any_points = True
        ax.scatter([v[0] for v in valid], [v[1] for v in valid], label=cfg, s=60)
        for x, y, client in valid:
            ax.annotate(client, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7)
    if not any_points:
        plt.close(fig)
        return None
    ax.axhline(0, linewidth=0.9, linestyle="--")
    ax.set_xlabel(signal)
    ax.set_ylabel(f"{metric} recovery vs no-score (+ = better)")
    ax.set_title(f"Targeting check: recovery vs quality signal -- {metric}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _savefig(fig, f"targeting_{signal}_{metric}")


# =====================================================================
# REPORT
# =====================================================================
def group_summary_table_for_metric(group_summary, metric):
    rows = [r for r in group_summary if r["metric"] == metric]
    table_rows = []
    for r in rows:
        table_rows.append({
            "comparison": r["comparison_pair"],
            "group": r["group"],
            "n": r["n"],
            "mean delta": fmt(r.get("mean_delta")),
            "median delta": fmt(r.get("median_delta")),
            "std": fmt(r.get("std_delta")),
            "% improved": fmt_pct(r.get("pct_improved")),
            "% degraded": fmt_pct(r.get("pct_degraded")),
            "wilcoxon p": fmt(r.get("wilcoxon_p"), 4) if r.get("wilcoxon_p") is not None else "n/a (n too small or scipy missing)",
        })
    return markdown_table(["comparison", "group", "n", "mean delta", "median delta", "std", "% improved", "% degraded", "wilcoxon p"], table_rows)


def tradeoff_leaderboard_table(tradeoff_rows, metric):
    rows = [r for r in tradeoff_rows if r["metric"] == metric]
    rows_sorted = sorted(
        rows,
        key=lambda r: (fnum(r.get("recovery_to_cost_ratio")) is None, -(fnum(r.get("recovery_to_cost_ratio")) or 0)),
    )
    table_rows = []
    for rank, r in enumerate(rows_sorted, start=1):
        table_rows.append({
            "rank": rank,
            "config": r["config"],
            "n_clean": r["n_clean"],
            "n_faulty": r["n_faulty"],
            "cost (clean, vs baseline)": fmt(r.get("cost_mean_delta_clean_vs_baseline")),
            "recovery (faulty, vs noscore)": fmt(r.get("recovery_mean_delta_faulty_vs_noscore")),
            "still below baseline (faulty)": fmt(r.get("still_below_baseline_mean_delta_faulty")),
            "recovery/cost ratio": fmt(r.get("recovery_to_cost_ratio"), 3),
        })
    return markdown_table(["rank", "config", "n_clean", "n_faulty", "cost (clean, vs baseline)",
                            "recovery (faulty, vs noscore)", "still below baseline (faulty)",
                            "recovery/cost ratio"], table_rows)


def targeting_table(corr_rows, metric):
    rows = [r for r in corr_rows if r["metric"] == metric]
    table_rows = []
    for r in rows:
        table_rows.append({
            "comparison": r["comparison_pair"],
            "signal": r["signal"],
            "n": r["n"],
            "spearman rho": fmt(r.get("spearman_rho"), 3),
            "p value": fmt(r.get("p_value"), 4) if r.get("p_value") is not None else "n/a (scipy missing or n<3)",
        })
    return markdown_table(["comparison", "signal", "n", "spearman rho", "p value"], table_rows)


def cluster_config_table(cluster_summary):
    rows = []
    for r in cluster_summary:
        rows.append({
            "config": r["config"],
            "cluster": r["cluster_id"],
            "clients": r["num_clients"],
            "MAE mean": fmt(r.get("mae_mean")),
            "MAE p95": fmt(r.get("mae_p95")),
            "RMSE mean": fmt(r.get("rmse_mean")),
            "R2 mean": fmt(r.get("r2_mean")),
            "worst MAE client": r.get("worst_mae_client", ""),
        })
    return markdown_table(["config", "cluster", "clients", "MAE mean", "MAE p95", "RMSE mean", "R2 mean", "worst MAE client"], rows)


def faulty_recovery_table(faulty_recovery_rows, metric):
    rows = [r for r in faulty_recovery_rows if r["metric"] == metric]
    table_rows = []
    for r in rows:
        table_rows.append({
            "client": r["client_id"],
            "cluster": r.get("cluster_id", ""),
            "config": r["config"],
            "baseline": fmt(r.get("baseline_value")),
            "no-score": fmt(r.get("faulty_noscore_value")),
            "method": fmt(r.get("method_value")),
            "recovery": fmt(r.get("recovery_delta")),
            "remaining gap": fmt(r.get("remaining_gap_delta_vs_baseline")),
            "% damage recovered": fmt_pct(r.get("recovered_percent")),
        })
    return markdown_table(["client", "cluster", "config", "baseline", "no-score", "method",
                           "recovery", "remaining gap", "% damage recovered"], table_rows)


def fairness_gap_table(fairness_rows, metric):
    rows = [r for r in fairness_rows if r["metric"] == metric]
    table_rows = []
    for r in rows:
        table_rows.append({
            "config": r["config"],
            "clean mean": fmt(r.get("clean_mean")),
            "faulty mean": fmt(r.get("faulty_mean")),
            "gap (+ = faulty worse)": fmt(r.get("performance_gap_faulty_worse_positive")),
            "normalized gap": fmt_pct(r.get("normalized_gap_percent")),
        })
    return markdown_table(["config", "clean mean", "faulty mean", "gap (+ = faulty worse)",
                           "normalized gap"], table_rows)


def tail_risk_table(tail_rows, metric, group="overall"):
    rows = [r for r in tail_rows if r["metric"] == metric and r["group"] == group]
    table_rows = []
    for r in rows:
        table_rows.append({
            "config": r["config"],
            "n": r["n"],
            "median": fmt(r.get("median")),
            "p90": fmt(r.get("p90")),
            "p95": fmt(r.get("p95")),
            "worst": fmt(r.get("worst_value")),
            "worst client": r.get("worst_client_id", ""),
        })
    return markdown_table(["config", "n", "median", "p90", "p95", "worst", "worst client"], table_rows)


def thesis_summary_table(tradeoff_rows, fairness_rows, faulty_recovery_rows, metric="mae"):
    trade = {r["config"]: r for r in tradeoff_rows if r["metric"] == metric}
    fair = {r["config"]: r for r in fairness_rows if r["metric"] == metric}
    configs = sorted(trade)
    table_rows = []
    for cfg in configs:
        individual = [r for r in faulty_recovery_rows if r["metric"] == metric and r["config"] == cfg]
        improved = sum(1 for r in individual if (fnum(r.get("recovery_delta")) or 0) > DELTA_EPS)
        total = len(individual)
        recovered_pct = safe_div(improved, total)
        recovered_pct = recovered_pct * 100 if recovered_pct is not None else None
        table_rows.append({
            "config": cfg,
            "clean cost": fmt(trade[cfg].get("cost_mean_delta_clean_vs_baseline")),
            "faulty recovery": fmt(trade[cfg].get("recovery_mean_delta_faulty_vs_noscore")),
            "faulty improved": f"{improved}/{total} ({fmt_pct(recovered_pct)})" if total else "n/a",
            "remaining gap": fmt(trade[cfg].get("still_below_baseline_mean_delta_faulty")),
            "clean-faulty gap": fmt(fair.get(cfg, {}).get("performance_gap_faulty_worse_positive")),
        })
    return markdown_table(["config", "clean cost", "faulty recovery", "faulty improved",
                           "remaining gap", "clean-faulty gap"], table_rows)


def build_report(run_meta, per_client, group_summary, tradeoff_rows, corr_rows, cluster_summary,
                  faulty_recovery_rows, fairness_rows, tail_rows,
                  present_configs, recovery_configs, plot_paths):
    n_clean = sum(1 for r in per_client if not r["is_fault_injected"])
    n_faulty = sum(1 for r in per_client if r["is_fault_injected"])
    total_clients = n_clean + n_faulty
    selected_count = run_meta.get("selected_client_count")
    selected_count_text = str(int(selected_count)) if fnum(selected_count) is not None else "n/a"
    eligible_count = run_meta.get("eligible_client_count")
    eligible_count_text = str(int(eligible_count)) if fnum(eligible_count) is not None else "n/a"
    exp_label = experiment_label()

    lines = []
    lines.append(f"# {exp_label} -- Fault Recovery & Sacrifice Analysis")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Configs found: {', '.join(sorted(present_configs))}")
    lines.append(f"- Reference (clean-data ideal): `{REFERENCE}`")
    lines.append(f"- Do-nothing faulty reference: `{NOSCORE_REFERENCE}`")
    lines.append(f"- Recovery configs compared against `{NOSCORE_REFERENCE}`: {', '.join(recovery_configs)}")
    lines.append(f"- Total clients: {total_clients}  |  Clean clients: {n_clean}  |  Faulty clients: {n_faulty}")
    lines.append(f"- Fault selection: `{run_meta.get('client_selection_mode')}`, "
                  f"seed `{run_meta.get('client_selection_seed')}`, "
                  f"{selected_count_text} of {eligible_count_text} "
                  f"eligible clients selected for fault injection")
    lines.append("")
    lines.append(f"> **Note on sample size:** this run has {n_faulty} faulty clients and {n_clean} clean "
                  "clients. Read small groups as case-study evidence; larger groups support more stable "
                  "descriptive statistics and significance tests.")
    lines.append("")

    lines.append("## Recovery Leaderboard (headline)")
    lines.append("")
    lines.append(f"Ranked by `recovery / cost` ratio: how much a config helps the {n_faulty} faulty clients "
                  f"(vs. doing nothing / `faulty_noscore`) per unit of performance it costs the {n_clean} clean "
                  "clients (vs. the clean-data `baseline`). Higher is better. Always check the raw "
                  "cost/recovery columns too -- a config that barely touches either number can still "
                  "show a high ratio.")
    lines.append("")
    lines.append("### MAE")
    lines.append("")
    lines.append(tradeoff_leaderboard_table(tradeoff_rows, "mae"))
    if "tradeoff_mae" in plot_paths:
        lines.append("")
        lines.append(f"![tradeoff mae](plots/{plot_paths['tradeoff_mae']})")
    lines.append("")
    for metric in METRICS:
        if metric == "mae":
            continue
        lines.append(f"### {metric.upper()}")
        lines.append("")
        lines.append(tradeoff_leaderboard_table(tradeoff_rows, metric))
        lines.append("")

    lines.append("## Thesis Summary Table")
    lines.append("")
    lines.append("This compact table combines recovery, sacrifice, and fairness for the primary MAE metric. "
                 "All deltas use the report convention: positive is better; the clean-faulty gap is "
                 "positive when faulty clients perform worse.")
    lines.append("")
    lines.append(thesis_summary_table(tradeoff_rows, fairness_rows, faulty_recovery_rows, "mae"))
    lines.append("")

    lines.append("## Faulty-Client Case Analysis")
    lines.append("")
    lines.append("Because the faulty group is small, every affected client is shown explicitly. Recovery is "
                 "measured against `faulty_noscore`; remaining gap is measured against the clean-data baseline.")
    lines.append("")
    for metric in PLOT_METRICS:
        lines.append(f"### {metric.upper()}")
        lines.append("")
        lines.append(faulty_recovery_table(faulty_recovery_rows, metric))
        for key in (f"faulty_client_recovery_{metric}", f"remaining_gap_{metric}"):
            fname = plot_paths.get(key)
            if fname:
                lines.append("")
                lines.append(f"![{key}](plots/{fname})")
        lines.append("")

    lines.append("## Fairness: Clean-Faulty Performance Gap")
    lines.append("")
    lines.append("The gap is oriented so that positive values mean the faulty group performs worse. "
                 "A smaller value therefore indicates less disparity.")
    lines.append("")
    for metric in PLOT_METRICS:
        lines.append(f"### {metric.upper()}")
        lines.append("")
        lines.append(fairness_gap_table(fairness_rows, metric))
        fname = plot_paths.get(f"fairness_gap_{metric}")
        if fname:
            lines.append("")
            lines.append(f"![fairness gap {metric}](plots/{fname})")
        lines.append("")

    lines.append("## Tail Risk and Worst-Client Performance")
    lines.append("")
    lines.append("Median, p90, p95, and worst-client values show whether a method improves the average while "
                 "leaving a small number of clients severely degraded.")
    lines.append("")
    for metric in PLOT_METRICS:
        lines.append(f"### {metric.upper()}")
        lines.append("")
        lines.append(tail_risk_table(tail_rows, metric, "overall"))
        fname = plot_paths.get(f"tail_risk_overall_{metric}")
        if fname:
            lines.append("")
            lines.append(f"![tail risk {metric}](plots/{fname})")
        lines.append("")

    lines.append("## Client-Level MAE Distribution")
    lines.append("")
    lines.append("Empirical CDFs show the whole client distribution for each configuration. Curves that are "
                 "farther left and higher indicate that more clients achieve lower MAE. A zoomed panel is "
                 "included because a few extreme clients stretch the full-range x-axis.")
    lines.append("")
    fname = plot_paths.get("ecdf_overall_mae_full_zoom")
    if fname:
        lines.append("### Raw MAE ECDF")
        lines.append("")
        lines.append(f"![ecdf mae](plots/{fname})")
        lines.append("")
    fname = plot_paths.get(f"ecdf_delta_vs_{NOSCORE_REFERENCE}_overall_mae")
    if fname:
        lines.append("### MAE Recovery ECDF vs Faulty No-Score")
        lines.append("")
        lines.append("Positive values mean the method improved a client relative to `faulty_noscore`; "
                     "negative values mean it degraded that client.")
        lines.append("")
        lines.append(f"![delta ecdf vs noscore](plots/{fname})")
        lines.append("")
    fname = plot_paths.get(f"ecdf_delta_vs_{REFERENCE}_overall_mae")
    if fname:
        lines.append("### MAE Delta ECDF vs Baseline")
        lines.append("")
        lines.append("Positive values mean the method beats the clean-data baseline for that client; "
                     "negative values show remaining degradation relative to baseline.")
        lines.append("")
        lines.append(f"![delta ecdf vs baseline](plots/{fname})")
        lines.append("")
    fname = plot_paths.get("survival_overall_mae")
    if fname:
        lines.append("### MAE Tail / Survival Curve")
        lines.append("")
        lines.append("This shows the fraction of clients whose MAE remains above each threshold, making p90/p95 "
                     "and worst-client behavior easier to see.")
        lines.append("")
        lines.append(f"![survival mae](plots/{fname})")
        lines.append("")

    lines.append("## Clean vs Faulty Group Comparison (detail)")
    lines.append("")
    lines.append("Positive delta = better than the reference in that comparison. Comparisons named "
                  "`baseline_to_X` measure cost/degradation vs the clean-data ideal. Comparisons named "
                  f"`{NOSCORE_REFERENCE}_to_X` measure recovery vs doing nothing.")
    lines.append("")
    for metric in METRICS:
        lines.append(f"### {metric.upper()}")
        lines.append("")
        lines.append(group_summary_table_for_metric(group_summary, metric))
        if f"boxplot_{metric}" in plot_paths.values() or metric in PLOT_METRICS:
            fname = plot_paths.get(f"boxplot_{metric}")
            if fname:
                lines.append("")
                lines.append(f"![boxplot {metric}](plots/{fname})")
        lines.append("")

    lines.append("## Per-Client Trajectories")
    lines.append("")
    lines.append("Each line is one client's raw metric value across configs. Red = faulty clients, "
                  "blue = clean clients (faint). A faulty (red) line dropping back down toward the "
                  "blue band means that config is recovering that client's performance.")
    lines.append("")
    for metric in PLOT_METRICS:
        fname = plot_paths.get(f"trajectory_{metric}")
        if fname:
            lines.append(f"### {metric.upper()}")
            lines.append("")
            lines.append(f"![trajectory {metric}](plots/{fname})")
            lines.append("")

    lines.append("## Cluster Robustness Check")
    lines.append("")
    lines.append("Same comparisons, but split by cluster, to check the headline story isn't being "
                  "driven by one cluster's scale (cluster MAE values differ a lot in raw terms).")
    lines.append("")
    for metric in PLOT_METRICS:
        fname = plot_paths.get(f"heatmap_{metric}")
        if fname:
            lines.append(f"![heatmap {metric}](plots/{fname})")
            lines.append("")

    lines.append("## Targeting Check")
    lines.append("")
    lines.append("Does the recovery magnitude for faulty clients correlate with how bad their quality "
                  "signal was? A negative rho for `client_pca_quality_score` (lower quality -> bigger "
                  "recovery) or a positive rho for `achieved_faulty_fraction` (more faults -> bigger "
                  "recovery) would suggest the scoring mechanism is targeted rather than uniform. "
                  f"**With {n_faulty} faulty clients, treat these correlations according to that sample size.**")
    lines.append("")
    lines.append(targeting_table(corr_rows, "mae"))
    fname = plot_paths.get("targeting_client_pca_quality_score_mae")
    if fname:
        lines.append("")
        lines.append(f"![targeting quality vs recovery](plots/{fname})")
    lines.append("")

    lines.append("## Appendix: Overall Cluster x Config Summary")
    lines.append("")
    lines.append("_Not adjusted for the clean/faulty split -- included for reference only. This is the "
                  "kind of table that hides the sacrifice/recovery story; use the sections above for "
                  "the real comparison._")
    lines.append("")
    lines.append(cluster_config_table(cluster_summary))
    lines.append("")

    lines.append("## Generated Files")
    lines.append("")
    for fname in ("per_client_master.csv", "group_comparison_summary.csv", "tradeoff_summary.csv",
                  "faulty_client_recovery.csv", "fairness_gap_summary.csv", "tail_risk_summary.csv",
                  "round_quality_summary.csv", "targeting_correlations.csv", "cluster_config_summary.csv"):
        lines.append(f"- `{fname}`")
    for fname in sorted(plot_paths.values()):
        lines.append(f"- `plots/{fname}`")
    lines.append("")

    return "\n".join(lines)


# =====================================================================
# MAIN
# =====================================================================
def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Report dir:", REPORT_DIR)

    # 1. Load
    metrics_rows, by_config_client = load_output_metrics()
    print(f"metrics_rows: {len(metrics_rows)}")

    present_configs = {cfg for cfg in OUTPUTS if any(r["config"] == cfg for r in metrics_rows)}
    missing_configs = set(OUTPUTS) - present_configs
    if missing_configs:
        print(f"Skipping configs with no data yet: {sorted(missing_configs)}")

    if REFERENCE not in present_configs:
        raise RuntimeError(f"Reference config '{REFERENCE}' has no data -- cannot proceed.")

    compare_configs = sorted(c for c in present_configs if c != REFERENCE)
    if NOSCORE_REFERENCE in present_configs:
        recovery_configs = sorted(c for c in present_configs if c not in (REFERENCE, NOSCORE_REFERENCE))
    else:
        print(f"WARNING: '{NOSCORE_REFERENCE}' not present -- recovery-vs-noscore comparisons skipped.")
        recovery_configs = []

    config_pairs = [(REFERENCE, c) for c in compare_configs]
    recovery_pairs = [(NOSCORE_REFERENCE, c) for c in recovery_configs]
    all_pairs = config_pairs + recovery_pairs

    print("Cost comparisons (vs baseline):", config_pairs)
    print("Recovery comparisons (vs noscore):", recovery_pairs)

    cluster_by_client = {r["client_id"]: r["cluster_id"] for r in metrics_rows}
    fault_summary, run_meta = load_faults(cluster_by_client)
    quality = load_quality_scores()

    # 2. Merge
    per_client = make_per_client_master(metrics_rows, by_config_client, fault_summary, quality, present_configs)
    add_pairwise_deltas(per_client, all_pairs)

    # 3. Analyze
    group_summary = build_group_comparison_summary(per_client, all_pairs)
    tradeoff_rows = build_tradeoff_summary(per_client, recovery_configs) if recovery_configs else []
    corr_rows = build_targeting_correlations(per_client, recovery_pairs) if recovery_pairs else []
    cluster_summary = summarize_group(metrics_rows, ["config", "cluster_id"])
    faulty_recovery_rows = build_faulty_client_recovery(per_client, recovery_configs) if recovery_configs else []
    fairness_rows = build_fairness_gap_summary(per_client, present_configs)
    tail_rows = build_tail_risk_summary(per_client, present_configs)
    round_quality_rows = build_round_quality_summary(fault_summary)

    # 4. Plots
    plot_paths = {}
    if HAVE_MPL:
        for metric in PLOT_METRICS:
            box_pairs = recovery_pairs if recovery_pairs else config_pairs
            fname = plot_group_split_boxplot(per_client, box_pairs, metric)
            if fname:
                plot_paths[f"boxplot_{metric}"] = fname

            traj_configs = [REFERENCE] + ([NOSCORE_REFERENCE] if NOSCORE_REFERENCE in present_configs else []) + recovery_configs
            traj_configs = list(dict.fromkeys(traj_configs))  # dedupe, keep order
            fname = plot_client_trajectory(per_client, traj_configs, metric)
            if fname:
                plot_paths[f"trajectory_{metric}"] = fname

            if tradeoff_rows:
                fname = plot_tradeoff_scatter(tradeoff_rows, metric)
                if fname:
                    plot_paths[f"tradeoff_{metric}"] = fname

            fname = plot_cluster_heatmap(metrics_rows, sorted(present_configs), metric)
            if fname:
                plot_paths[f"heatmap_{metric}"] = fname

            if faulty_recovery_rows:
                fname = plot_faulty_client_recovery(faulty_recovery_rows, recovery_configs, metric)
                if fname:
                    plot_paths[f"faulty_client_recovery_{metric}"] = fname
                fname = plot_remaining_gap(faulty_recovery_rows, recovery_configs, metric)
                if fname:
                    plot_paths[f"remaining_gap_{metric}"] = fname
                fname = plot_targeting_scatter(faulty_recovery_rows, recovery_configs, metric)
                if fname:
                    plot_paths[f"targeting_client_pca_quality_score_{metric}"] = fname

            fname = plot_fairness_gap(fairness_rows, metric)
            if fname:
                plot_paths[f"fairness_gap_{metric}"] = fname

            fname = plot_tail_risk(tail_rows, metric, "overall")
            if fname:
                plot_paths[f"tail_risk_overall_{metric}"] = fname

            if metric == "mae":
                fname = plot_metric_ecdf_full_zoom(per_client, present_configs, metric, "overall", zoom_max=6.0)
                if fname:
                    plot_paths[f"ecdf_overall_{metric}_full_zoom"] = fname

                if NOSCORE_REFERENCE in present_configs and recovery_configs:
                    fname = plot_delta_ecdf(per_client, NOSCORE_REFERENCE, recovery_configs, metric, "overall")
                    if fname:
                        plot_paths[f"ecdf_delta_vs_{NOSCORE_REFERENCE}_overall_{metric}"] = fname

                baseline_delta_configs = sorted(c for c in present_configs if c != REFERENCE)
                fname = plot_delta_ecdf(per_client, REFERENCE, baseline_delta_configs, metric, "overall")
                if fname:
                    plot_paths[f"ecdf_delta_vs_{REFERENCE}_overall_{metric}"] = fname

                fname = plot_metric_survival(per_client, present_configs, metric, "overall")
                if fname:
                    plot_paths[f"survival_overall_{metric}"] = fname
    else:
        print("matplotlib not available -- skipping plots (pip install matplotlib --break-system-packages)")

    # 5. Write CSVs
    per_client_fields = sorted({k for row in per_client for k in row})
    write_csv(REPORT_DIR / "per_client_master.csv", per_client, per_client_fields)

    group_summary_fields = sorted({k for row in group_summary for k in row})
    write_csv(REPORT_DIR / "group_comparison_summary.csv", group_summary, group_summary_fields)

    if tradeoff_rows:
        tradeoff_fields = sorted({k for row in tradeoff_rows for k in row})
        write_csv(REPORT_DIR / "tradeoff_summary.csv", tradeoff_rows, tradeoff_fields)

    if corr_rows:
        corr_fields = sorted({k for row in corr_rows for k in row})
        write_csv(REPORT_DIR / "targeting_correlations.csv", corr_rows, corr_fields)

    if faulty_recovery_rows:
        fields = sorted({k for row in faulty_recovery_rows for k in row})
        write_csv(REPORT_DIR / "faulty_client_recovery.csv", faulty_recovery_rows, fields)

    fairness_fields = sorted({k for row in fairness_rows for k in row})
    write_csv(REPORT_DIR / "fairness_gap_summary.csv", fairness_rows, fairness_fields)

    tail_fields = sorted({k for row in tail_rows for k in row})
    write_csv(REPORT_DIR / "tail_risk_summary.csv", tail_rows, tail_fields)

    if round_quality_rows:
        round_quality_fields = [
            "Configuration",
            "Round",
            "Cluster",
            "Clients",
            "Faulty clients",
            "Mean quality",
            "p10 quality",
            "Std quality",
            "Effective weight",
        ]
        write_csv(REPORT_DIR / "round_quality_summary.csv", round_quality_rows, round_quality_fields)

    cluster_fields = sorted({k for row in cluster_summary for k in row})
    write_csv(REPORT_DIR / "cluster_config_summary.csv", cluster_summary, cluster_fields)

    # 6. Report
    report_text = build_report(run_meta, per_client, group_summary, tradeoff_rows, corr_rows,
                                cluster_summary, faulty_recovery_rows, fairness_rows, tail_rows,
                                present_configs, recovery_configs, plot_paths)
    (REPORT_DIR / "report.md").write_text(report_text)

    print(f"per_client rows: {len(per_client)}")
    print(f"group_summary rows: {len(group_summary)}")
    print(f"tradeoff rows: {len(tradeoff_rows)}")
    print(f"correlation rows: {len(corr_rows)}")
    print(f"faulty-client recovery rows: {len(faulty_recovery_rows)}")
    print(f"fairness-gap rows: {len(fairness_rows)}")
    print(f"tail-risk rows: {len(tail_rows)}")
    print(f"round-quality rows: {len(round_quality_rows)}")
    print("Report written to:", REPORT_DIR / "report.md")


if __name__ == "__main__":
    main()
