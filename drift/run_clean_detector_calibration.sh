#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
baseline="${BASELINE_DIR:-$repo_root/outputs/baseline-bs128}"
output_root="${OUTPUT_ROOT:-$repo_root/outputs/drift/calibration}"
python_bin="${PYTHON_BIN:-$repo_root/venv/bin/python}"

if [[ ! -d "$baseline" ]]; then
    echo "Baseline directory does not exist: $baseline" >&2
    exit 2
fi
if ! find "$baseline" -type f -name '*.pt' -print -quit | grep -q .; then
    echo "No .pt checkpoints found under baseline: $baseline" >&2
    exit 2
fi
if [[ ! -x "$python_bin" ]]; then
    echo "Python executable not found or not executable: $python_bin" >&2
    exit 2
fi

mkdir -p "$output_root"

run_clean_calibration() {
    local name="$1"
    local detector="$2"
    local ph_min="$3"
    local ph_delta="$4"
    local ph_threshold="$5"
    local adwin_delta="$6"
    local run_dir="$output_root/$name"

    if [[ -e "$run_dir" ]]; then
        echo "Refusing to overwrite existing run directory: $run_dir" >&2
        exit 2
    fi

    cp -a "$baseline" "$run_dir"

    printf '%s\n' \
        "DRIFT_DETECTORS=$detector" \
        "PH_MIN_INSTANCES=$ph_min" \
        "PH_DELTA=$ph_delta" \
        "PH_THRESHOLD=$ph_threshold" \
        "PH_ALPHA=0.9999" \
        "ADWIN_DELTA=$adwin_delta" \
        "ADWIN_CLOCK=32" \
        "ADWIN_MAX_BUCKETS=5" \
        "ADWIN_MIN_WINDOW_LENGTH=5" \
        "ADWIN_GRACE_PERIOD=10" \
        > "$run_dir/detector_parameters.env"

    echo "Starting $name"
    echo "Output: $run_dir"

    (
        cd "$repo_root"
        env \
            OUTPUT_DIR="$run_dir" \
            CLUSTER_IDS= \
            CLIENT_IDS= \
            BATCH_SIZE=128 \
            NORMALIZE=true \
            NORMALIZE_METHOD=zscore \
            SKIP_TRAINING=true \
            FROZEN_VALIDATION_MODE=true \
            DRIFT_INJECTION_ENABLED=false \
            DRIFT_DETECTION_ENABLED=true \
            DRIFT_SOURCE=input \
            DRIFT_SCOPE=test \
            DRIFT_TYPE=none \
            DRIFT_CLIENT_IDS= \
            DRIFT_CLIENT_FRACTION=0 \
            DRIFT_DETECTORS="$detector" \
            DRIFT_RUN_ID="$name" \
            DRIFT_CONFIGURATION_ID="$name" \
            DRIFT_RANDOM_SEED=42 \
            PH_MIN_INSTANCES="$ph_min" \
            PH_DELTA="$ph_delta" \
            PH_THRESHOLD="$ph_threshold" \
            PH_ALPHA=0.9999 \
            ADWIN_DELTA="$adwin_delta" \
            ADWIN_CLOCK=32 \
            ADWIN_MAX_BUCKETS=5 \
            ADWIN_MIN_WINDOW_LENGTH=5 \
            ADWIN_GRACE_PERIOD=10 \
            "$python_bin" run_split.py
    ) 2>&1 | tee "$run_dir/run.log"
}

# Page-Hinkley threshold sweep. PH-A through PH-C isolate PH_THRESHOLD.
run_clean_calibration \
    "exp0_PH-A_min500_delta0p01_threshold150" \
    "page_hinkley" 500 0.01 150 0.0005

run_clean_calibration \
    "exp0_PH-B_min500_delta0p01_threshold200" \
    "page_hinkley" 500 0.01 200 0.0005

run_clean_calibration \
    "exp0_PH-C_min500_delta0p01_threshold300" \
    "page_hinkley" 500 0.01 300 0.0005

# Page-Hinkley delta sweep at the selected reference threshold of 200.
run_clean_calibration \
    "exp0_PH-D_min500_delta0p02_threshold200" \
    "page_hinkley" 500 0.02 200 0.0005

run_clean_calibration \
    "exp0_PH-E_min500_delta0p05_threshold200" \
    "page_hinkley" 500 0.05 200 0.0005

# ADWIN-only delta sweep. PH values are recorded but PH is not enabled.
run_clean_calibration \
    "exp0_ADWIN-A_delta0p0001" \
    "adwin" 500 0.01 200 0.0001

run_clean_calibration \
    "exp0_ADWIN-B_delta0p00005" \
    "adwin" 500 0.01 200 0.00005

run_clean_calibration \
    "exp0_ADWIN-C_delta0p00001" \
    "adwin" 500 0.01 200 0.00001

echo "Completed clean detector calibration runs under: $output_root"
