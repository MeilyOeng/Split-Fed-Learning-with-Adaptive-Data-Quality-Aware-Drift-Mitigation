#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 CLEAN_CHECKPOINT_DIR [OUTPUT_ROOT] [pilot|all]"
    echo "  pilot: run sudden drift at magnitudes 0.25 and 0.5 (default)"
    echo "  all:   run planned Experiments 1-4"
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
clean_checkpoint_dir="$(realpath "$1")"
output_root="${2:-outputs/drift}"
phase="${3:-pilot}"
mkdir -p "$output_root"
output_root="$(realpath "$output_root")"

if [[ "$phase" != "pilot" && "$phase" != "all" ]]; then
    usage >&2
    exit 2
fi
if [[ ! -d "$clean_checkpoint_dir" ]]; then
    echo "Clean checkpoint directory does not exist: $clean_checkpoint_dir" >&2
    exit 2
fi
if ! find "$clean_checkpoint_dir" -type f -name '*.pt' -print -quit | grep -q .; then
    echo "No .pt checkpoints found under: $clean_checkpoint_dir" >&2
    exit 2
fi

python_bin="${PYTHON_BIN:-$repo_root/venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "Python executable not found or not executable: $python_bin" >&2
    exit 2
fi

seed="${DRIFT_RANDOM_SEED:-42}"
selected_fraction="${DRIFT_SELECTED_FRACTION:-0.5}"
start_ratio="${DRIFT_START_RATIO:-0.5}"
transition_start_ratio="${DRIFT_TRANSITION_START_RATIO:-0.3}"
transition_end_ratio="${DRIFT_TRANSITION_END_RATIO:-0.7}"
recurring_start_ratio="${DRIFT_RECURRING_START_RATIO:-0.3}"
planned_magnitude="${DRIFT_MAGNITUDE:-0.5}"
period="${DRIFT_PERIOD:-168}"
batch_size="${BATCH_SIZE:-128}"

# Selected clean-calibration result for Page-Hinkley.
ph_min_instances="${PH_MIN_INSTANCES:-500}"
ph_delta="${PH_DELTA:-0.01}"
ph_threshold="${PH_THRESHOLD:-500}"
ph_alpha="${PH_ALPHA:-0.9999}"

# ADWIN remains configurable until its clean calibration is selected.
adwin_delta="${ADWIN_DELTA:-0.002}"
adwin_clock="${ADWIN_CLOCK:-32}"
adwin_max_buckets="${ADWIN_MAX_BUCKETS:-5}"
adwin_min_window_length="${ADWIN_MIN_WINDOW_LENGTH:-5}"
adwin_grace_period="${ADWIN_GRACE_PERIOD:-10}"

name_token() {
    local value="$1"
    value="${value//./p}"
    value="${value//-/_}"
    printf '%s' "$value"
}

stage_checkpoints() {
    local run_dir="$1"
    if [[ -e "$run_dir" ]]; then
        echo "Refusing to reuse an existing run directory: $run_dir" >&2
        exit 2
    fi
    mkdir -p "$run_dir"
    while IFS= read -r -d '' checkpoint; do
        local relative_path="${checkpoint#"$clean_checkpoint_dir"/}"
        mkdir -p "$run_dir/$(dirname "$relative_path")"
        cp -p "$checkpoint" "$run_dir/$relative_path"
    done < <(find "$clean_checkpoint_dir" -type f -name '*.pt' -print0)
}

write_detector_parameters() {
    local run_dir="$1"
    printf '%s\n' \
        "PH_MIN_INSTANCES=$ph_min_instances" \
        "PH_DELTA=$ph_delta" \
        "PH_THRESHOLD=$ph_threshold" \
        "PH_ALPHA=$ph_alpha" \
        "ADWIN_DELTA=$adwin_delta" \
        "ADWIN_CLOCK=$adwin_clock" \
        "ADWIN_MAX_BUCKETS=$adwin_max_buckets" \
        "ADWIN_MIN_WINDOW_LENGTH=$adwin_min_window_length" \
        "ADWIN_GRACE_PERIOD=$adwin_grace_period" \
        > "$run_dir/detector_parameters.env"
}

run_experiment() {
    local run_name="$1"
    shift
    local run_dir="$output_root/$run_name"
    stage_checkpoints "$run_dir"
    write_detector_parameters "$run_dir"
    echo "Running $run_name -> $run_dir"
    (
        cd "$repo_root"
        env \
            OUTPUT_DIR="$run_dir" \
            CLUSTER_IDS= \
            CLIENT_IDS= \
            BATCH_SIZE="$batch_size" \
            SKIP_TRAINING=true \
            FROZEN_VALIDATION_MODE=true \
            NORMALIZE=true \
            NORMALIZE_METHOD=zscore \
            DRIFT_SOURCE=input \
            DRIFT_SCOPE=test \
            DRIFT_DETECTION_ENABLED=true \
            DRIFT_DETECTORS=page_hinkley,adwin \
            DRIFT_RANDOM_SEED="$seed" \
            DRIFT_RUN_ID="$run_name" \
            DRIFT_CONFIGURATION_ID="$run_name" \
            PH_MIN_INSTANCES="$ph_min_instances" \
            PH_DELTA="$ph_delta" \
            PH_THRESHOLD="$ph_threshold" \
            PH_ALPHA="$ph_alpha" \
            ADWIN_DELTA="$adwin_delta" \
            ADWIN_CLOCK="$adwin_clock" \
            ADWIN_MAX_BUCKETS="$adwin_max_buckets" \
            ADWIN_MIN_WINDOW_LENGTH="$adwin_min_window_length" \
            ADWIN_GRACE_PERIOD="$adwin_grace_period" \
            "$@" \
            "$python_bin" run_split.py
    ) 2>&1 | tee "$run_dir/run.log"
}

run_sudden() {
    local prefix="$1"
    local magnitude="$2"
    local magnitude_token
    magnitude_token="$(name_token "$magnitude")"
    run_experiment \
        "${prefix}_sudden_f$(name_token "$selected_fraction")_start$(name_token "$start_ratio")_mag${magnitude_token}_seed${seed}" \
        DRIFT_INJECTION_ENABLED=true \
        DRIFT_TYPE=sudden \
        DRIFT_CLIENT_IDS= \
        DRIFT_CLIENT_FRACTION="$selected_fraction" \
        DRIFT_START_RATIO="$start_ratio" \
        DRIFT_MAGNITUDE="$magnitude"
}

if [[ "$phase" == "pilot" ]]; then
    run_sudden "pilot_exp1" 0.25
    run_sudden "pilot_exp1" 0.5
    echo "Pilot complete. Review both experiment summaries before running phase 'all'."
    exit 0
fi

# Experiment 1: sudden drift magnitude comparison.
run_sudden "exp1" 0.25
run_sudden "exp1" 0.5

# Experiment 2: incremental test-input drift.
run_experiment \
    "exp2_incremental_f$(name_token "$selected_fraction")_start$(name_token "$transition_start_ratio")_end$(name_token "$transition_end_ratio")_mag$(name_token "$planned_magnitude")_seed${seed}" \
    DRIFT_INJECTION_ENABLED=true \
    DRIFT_TYPE=incremental \
    DRIFT_TRANSITION=linear \
    DRIFT_CLIENT_IDS= \
    DRIFT_CLIENT_FRACTION="$selected_fraction" \
    DRIFT_START_RATIO="$transition_start_ratio" \
    DRIFT_END_RATIO="$transition_end_ratio" \
    DRIFT_MAGNITUDE="$planned_magnitude"

# Experiment 3: sudden drift prevalence per cluster.
for fraction in 0.25 0.50 0.75; do
    run_experiment \
        "exp3_partial_sudden_f$(name_token "$fraction")_start$(name_token "$start_ratio")_mag$(name_token "$planned_magnitude")_seed${seed}" \
        DRIFT_INJECTION_ENABLED=true \
        DRIFT_TYPE=sudden \
        DRIFT_CLIENT_IDS= \
        DRIFT_CLIENT_FRACTION="$fraction" \
        DRIFT_START_RATIO="$start_ratio" \
        DRIFT_MAGNITUDE="$planned_magnitude"
done

# Experiment 4: recurring square-wave input drift.
run_experiment \
    "exp4_recurring_f$(name_token "$selected_fraction")_start$(name_token "$recurring_start_ratio")_period${period}_mag$(name_token "$planned_magnitude")_seed${seed}" \
    DRIFT_INJECTION_ENABLED=true \
    DRIFT_TYPE=recurring \
    DRIFT_CLIENT_IDS= \
    DRIFT_CLIENT_FRACTION="$selected_fraction" \
    DRIFT_START_RATIO="$recurring_start_ratio" \
    DRIFT_PERIOD="$period" \
    DRIFT_MAGNITUDE="$planned_magnitude"

echo "Completed frozen drift Experiments 1-4 under: $output_root"
