import csv
import json
import math
import sys
from collections import defaultdict, Counter
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

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "outputs/run4/E4Seed42"
OUTPUTS = {
    "baseline": ROOT / "outputs" / "baseline-bs128",
    "faulty_noscore": ROOT / EXPERIMENT / "faulty-noscore",
    "faulty_finalscore": ROOT / EXPERIMENT / "faulty-input-finalscore",
    # "faulty_weightAccscore": ROOT / EXPERIMENT / "faulty-weightAcc-score",
}
RUN_DIR = ROOT / "dataset" / "run4/E4Seed42"
QUALITY_DIR = RUN_DIR / "quality_output"
REPORT_DIR = ROOT / EXPERIMENT / "report"
PLOTS_DIR = REPORT_DIR / "plots"
FAULT_SUMMARY_PATH = QUALITY_DIR / "fault_injection_report_summary.csv"

DELTA_EPS = 1e-6

METRICS = ("mae", "rmse", "mse", "r2", "smape")
LOWER_BETTER = {"mae", "rmse", "mse", "smape"}

REFERENCE = "baseline"
# COMPARE_CONFIGS = ["faulty_noscore", "faulty_finalscore", "faulty_weightAccscore"]
COMPARE_CONFIGS = ["faulty_noscore", "faulty_finalscore"]  # add back "faulty_weightAccscore" 
CONFIG_PAIRS = [(REFERENCE, cfg) for cfg in COMPARE_CONFIGS]

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
        # Positive = improvement (error decreased)
        return old - new
    else:
        # Positive = improvement (score increased)
        return new - old

def load_output_metrics():
    metrics_rows = []
    by_config_client = {}
    for config_name, root in OUTPUTS.items():
        for path in sorted(root.glob("cluster_*/client_*_test_metrics.json")):
            payload = json.loads(path.read_text())
            client_id = payload.get("client_id")
            if client_id is None:
                stem = path.stem  # client_123_test_metrics
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


    return metrics_rows, by_config_client

def load_faults(cluster_by_client):
    rows = read_csv(FAULT_SUMMARY_PATH)
    print("Loaded fault summary rows:", len(rows))

    fault_summary = {}
    for row in rows:
        client_id = (row.get("client_id") or "").strip()
        if not client_id:
            continue  # shouldn't happen in this file, but guard anyway

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

    # run-level metadata, same on every row — grab once for the report header
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

    # -----------------------------
    # 1. Client PCA quality scores
    # -----------------------------
    pca_path = QUALITY_DIR / "client_pca_quality_scores.csv"
    pca_rows = read_csv(pca_path)

    print("PCA rows:", len(pca_rows))

    for row in pca_rows:
        client_id = (row.get("client_id") or "").strip()

        if not client_id:
            continue

        quality[client_id]["client_pca_client_id"] = client_id

        for key, value in row.items():
            if key == "client_id":
                continue

            quality[client_id][f"client_pca_{key}"] = fnum(value) \
                if fnum(value) is not None else value


    # -----------------------------
    # 2. Window quality scores
    # -----------------------------
    window_path = QUALITY_DIR / "window_quality_scores.csv"
    window_rows = read_csv(window_path)

    print("Window rows:", len(window_rows))

    # group by client -> mode
    window_by_client_mode = defaultdict(lambda: defaultdict(list))

    for row in window_rows:
        client_id = (row.get("client_id") or "").strip()

        if not client_id:
            continue

        mode = (row.get("mode") or "unknown").strip()

        score = fnum(row.get("quality_score"))

        if score is not None:
            window_by_client_mode[client_id][mode].append(score)


    def summarize(scores):
        if not scores:
            return {}

        ordered = sorted(scores)

        def percentile(values, q):
            if not values:
                return None

            pos = (len(values)-1) * q
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))

            if lo == hi:
                return values[lo]

            return (
                values[lo] * (hi - pos)
                + values[hi] * (pos - lo)
            )

        return {
            "mean": mean(scores),
            "min": min(scores),
            "p10": percentile(ordered, 0.10),
        }


    for client_id, modes in window_by_client_mode.items():

        # Aggregate per mode
        for mode, scores in modes.items():
            summary = summarize(scores)

            quality[client_id][
                f"window_quality_{mode}_mean"
            ] = summary["mean"]

            quality[client_id][
                f"window_quality_{mode}_min"
            ] = summary["min"]

            quality[client_id][
                f"window_quality_{mode}_p10"
            ] = summary["p10"]


        # Aggregate across all modes
        all_scores = [
            score
            for scores in modes.values()
            for score in scores
        ]

        summary = summarize(all_scores)

        quality[client_id]["window_quality_all_mean"] = summary["mean"]
        quality[client_id]["window_quality_all_min"] = summary["min"]
        quality[client_id]["window_quality_all_p10"] = summary["p10"]


    return dict(quality)
def make_per_client_master(metrics_rows, by_config_client, fault_summary, quality):
    master = []

    # Get unique clients from baseline (84 expected)
    client_ids = sorted({
        r["client_id"]
        for r in metrics_rows
        if r["config"] == REFERENCE
    })

    for client_id in client_ids:

        # -------------------------
        # Base client information
        # -------------------------
        baseline_row = by_config_client.get(
            (REFERENCE, client_id),
            {}
        )

        row = {
            "client_id": client_id,
            "cluster_id": baseline_row.get("cluster_id", ""),
        }


        # -------------------------
        # Fault information
        # -------------------------
        if client_id in fault_summary:
            row.update(fault_summary[client_id])
        else:
            row.update({
                "is_fault_injected": False,
                "fault_count": 0,
                "fault_types": "",
                "fault_dimensions": "",
                "fault_scope": "",
                "affected_rows_total": 0,
                "affected_fraction_sum": 0.0,
            })


        # -------------------------
        # Quality information
        # -------------------------
        row.update(
            quality.get(client_id, {})
        )


        # -------------------------
        # Model/config metrics
        # -------------------------
        for config_name in OUTPUTS:

            metric_row = by_config_client.get(
                (config_name, client_id),
                {}
            )

            for metric in METRICS:
                row[
                    f"{config_name}_raw_{metric}"
                ] = metric_row.get(
                    f"raw_{metric}"
                )

                row[
                    f"{config_name}_norm_{metric}"
                ] = metric_row.get(
                    f"norm_{metric}"
                )

            row[
                f"{config_name}_num_samples"
            ] = metric_row.get(
                "num_samples"
            )


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

    
def main():

    metrics_rows, by_config_client = load_output_metrics()

    cluster_by_client = {
        r["client_id"]: r["cluster_id"]
        for r in metrics_rows
    }

    fault_summary, run_meta = load_faults(
    cluster_by_client
)

    print("Faulty clients:", len(fault_summary))
    print("Run metadata:", run_meta)

    sample_id = next(iter(fault_summary))
    print(sample_id, fault_summary[sample_id])

    quality = load_quality_scores()

    master = make_per_client_master(
        metrics_rows,
        by_config_client,
        fault_summary,
        quality,
    )

    print("Master rows:", len(master))
    print("Columns:", len(master[0]))

    print(master[0])
    per_client = make_per_client_master(metrics_rows, by_config_client, fault_summary, quality)
    add_pairwise_deltas(per_client, CONFIG_PAIRS)

    row_9900070 = next(r for r in per_client if r["client_id"] == "9900070")
    print(row_9900070["baseline_to_faulty_noscore_mae_delta"])
    print(row_9900070["baseline_to_faulty_noscore_mae_delta_pct"])
if __name__ == "__main__":
    main()