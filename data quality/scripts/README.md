# Demand Quality Scripts

This folder now contains five CLI scripts for the wide-format demand quality pipeline.

These scripts are for a CSV such as `demand.csv` with:

- one timestamp column
- many client or meter columns
- one demand value per client per timestamp

Important data convention:

- the raw smart-meter dataset stores demand as negative values
- the shared loading path converts predominantly negative demand columns to positive consumption using `y = -y`
- all downstream fault injection, scoring, and PCA logic therefore operates on positive demand values

The new scripts are:

- `calibrate_demand_baseline.py`
- `inject_faults.py`
- `score_data_quality.py`
- `compute_pca_quality_score.py`
- `run_quality_pipeline.py`

They use the reusable implementation under `src/demand_quality/`.

## 1. `calibrate_demand_baseline.py`

Purpose:

- load clean historical demand data
- calibrate per-client local historical baselines
- save a JSON baseline artifact

Typical use:

```bash
python scripts/calibrate_demand_baseline.py \
  --input data/demand_clean.csv \
  --output output/demand_baseline.json \
  --timestamp-col timestamp \
  --config configs/data_quality_scoring.yaml
```

This is optional for the current pipeline because `score_data_quality.py` can still derive local baseline statistics directly from the clean reference CSV. It is useful when you want a persistent inspected baseline artifact for SplitFed experiments.

## 2. `inject_faults.py`

Purpose:

- load a clean wide-format demand CSV
- inject controlled faults into selected client columns
- save the faulty dataset
- save a detailed fault event report

Typical use:

```bash
python scripts/inject_faults.py \
  --input data/demand.csv \
  --output data/demand_faulty.csv \
  --report output/fault_injection_report.csv \
  --timestamp-col timestamp \
  --fault-config configs/fault_injection.yaml \
  --seed 42
```

Arguments:

- `--input`: clean input CSV
- `--output`: output CSV with injected faults
- `--report`: detailed fault report CSV
- `--timestamp-col`: timestamp column name; auto-detected if omitted
- `--fault-config`: YAML config for fault injection
- `--seed`: random seed for deterministic injection

Main outputs:

- `data/demand_faulty.csv`
- `output/fault_injection_report.csv`

YAML vs CLI responsibility:

- `configs/fault_injection.yaml` controls which faults exist, their rates, severities, and which clients are eligible
- `--input`, `--output`, and `--report` choose the actual files to read and write
- `--timestamp-col` tells the loader which timestamp column to use if you do not want auto-detection
- `--seed` controls reproducibility and is a CLI argument, not a YAML key
- `configs/data_quality_scoring.yaml` is separate and is only used later by the scorer

Faults supported:

- completeness faults:
  - random missing values
  - block missing intervals
  - optional shared dropped timestamp rows
- accuracy faults:
  - negative values
  - impossible high values
  - spikes
  - noisy values
  - jump faults
  - stuck-at constant segments
- temporal consistency faults:
  - stale blocks
  - calendar coverage holes
  - duplicated timestamp rows
  - phase shifts
  - mid-window regime shifts
- skewness faults:
  - mean shift
  - variance shift
  - peak-hour shift
  - long-term shift

Two injection modes are available:

- detailed mode:
  - leave `fault_fraction` empty in `configs/fault_injection.yaml`
  - the injector uses the per-fault settings under `faults:`
  - `global_fault_rate` scales the detailed rates up or down
- simple selected-client mode:
  - set `fault_fraction` in `configs/fault_injection.yaml`
  - the injector first chooses clients from `clients.selected`, `clients.random_fraction`, or `clients.random_count`
  - then each selected client gets about `round(valid_rows * fault_fraction)` corrupted timestamps
  - this matches the old demand-injector idea, but now only for the selected clients
  - when this mode is active, the detailed per-fault-rate settings under `faults:` are ignored

Example simple-mode config:

```yaml
fault_fraction: 0.15

clients:
  selected: [client_a, client_c]
  random_fraction: 1.0
  random_count:

simple_fault_types: [missing, out_of_range, inconsistent, timeliness]
```

This means:

- only `client_a` and `client_c` are affected
- each selected client gets faults on about 15% of its valid timestamps
- `client_b` is left unchanged
- the actual point-level fault type is sampled from `missing`, `out_of_range`, `inconsistent`, and `timeliness`

How the default fault-config numbers were chosen:

- the defaults are meant to be a moderate starting point for hourly demand data, not a universal truth
- they are intentionally strong enough to move quality scores in tests, but not so strong that every client becomes obviously broken
- `fault_fraction` is empty by default so the config starts in detailed mode rather than simple mode
- in simple mode, `fault_fraction: 0.15` means each selected client gets faults on about 15% of its valid timestamps
- `simple_fault_types: [missing, out_of_range, inconsistent, timeliness]` was chosen to stay close to the older demand injector: one missing-value fault, two strong value faults, and one timeliness fault
- `global_fault_rate: 1.0` only applies in detailed mode; it means the per-fault rates in the YAML are used as written, and lowering it to `0.5` scales all client-level fault rates down by half
- `random_missing_rate: 0.02` means about 2% sparse missingness per selected client; this is enough to reduce completeness without destroying the series
- `block_missing.events_per_client: 1`, `min_length: 4`, `max_length: 12` means one outage block of 4 to 12 hours by default; this is useful because block missingness usually hurts continuity more than sparse gaps
- `negative_rate: 0.005` and `impossible_high_rate: 0.005` are small because these are severe faults; even a few such points should already reduce `A1`
- `spike_rate: 0.01`, `spike_multiplier: 4.0` means about 1% of points receive large local spikes; the multiplier is tied to series scale so spikes stay visible across clients with different load magnitudes
- `noise_rate: 0.02`, `noise_std_fraction: 0.05` means around 2% of points get mild perturbations with standard deviation about 5% of the client scale; this is intended to stress anomaly scoring without turning the whole series chaotic
- `jump_rate: 0.01`, `jump_multiplier: 6.0` creates abrupt changes large enough to violate rate-of-change rules for hourly data
- `stuck_segments_per_client: 1`, `stuck_min_length: 4`, `stuck_max_length: 12` simulates a frozen meter for a few hours, which is long enough to trigger the context checks
- `stale_blocks_per_client: 1`, `stale_min_length: 3`, `stale_max_length: 8` gives one carry-forward stale segment per client by default
- `lag_choices: [1, 2, 24]`, `lag_probabilities: [0.5, 0.3, 0.2]` biases toward short delays, with a smaller chance of a one-day stale copy; that is usually more realistic than always using one fixed lag
- `calendar_hole_fraction: 0.03` removes roughly 3% of available hour-of-week cells inside a localized time window for a client; this reduces `TC3` without deleting a full-year pattern
- `phase_shift_client_fraction: 0.15`, `phase_shift_steps_choices: [6, 12]` means only some clients get a moderate rhythm shift in the second half of the window
- `mid_window_regime_shift_fraction: 0.2`, `mid_window_regime_shift_strength: 0.25` means only some clients get a moderate second-half level shift that breaks within-window stationarity
- `mean_shift_client_fraction: 0.25` and `variance_shift_client_fraction: 0.25` mean these broader distributional faults affect only a subset of clients by default, and now only inside temporary local windows
- `variance_scale_range: [1.5, 2.0]` was chosen to make the distribution visibly wider but still plausible
- `peak_shift_multiplier: 1.5` means peak-hour usage is increased by about 50%, which is large enough to move the peak/off-peak ratio but still interpretable
- `long_term_shift_multiplier: 0.5` adds a moderate partial-month style drift rather than a catastrophic full-series level jump

How to tune the fault config:

- if you want old-style `fault_fraction` behavior, set `fault_fraction` and use the `clients:` block to control which clients are touched
- if you want selected clients but richer event-level faults, leave `fault_fraction` empty and tune the detailed `faults:` section instead
- if the pipeline barely separates clean and faulty clients, increase event counts before increasing extreme-value multipliers
- if clients become unrealistically broken, first reduce `global_fault_rate`, then reduce the severe fault rates like `negative_rate`, `impossible_high_rate`, and `mean_shift_client_fraction`
- if your data is 15-minute or 30-minute rather than hourly, shorten stale and missing block lengths accordingly
- if your meters are industrial and naturally spiky, reduce `spike_multiplier` and rely more on distribution-shift faults than point spikes

## Fault Injection Examples

Below, `x_t` is one client column on the shared timestamp grid.

### Completeness injections

`random_missing`

- operation: choose isolated timestamps for one client and set those values to `NaN`
- dimension/subcheck: `completeness / C1`
- example:

```text
before: [10, 11, 12, 13, 14, 15]
after : [10, 11, NaN, 13, 14, NaN]
```

`block_missing`

- operation: choose one or more contiguous intervals and set the whole block to `NaN`
- dimension/subcheck: `completeness / C1`
- example:

```text
before: [10, 11, 12, 13, 14, 15, 16, 17]
after : [10, 11, 12, NaN, NaN, NaN, 16, 17]
```

`dropped_timestamp_rows`

- operation: remove whole rows from the table, which affects every client because the table is wide and timestamps are shared
- dimension/subcheck: `completeness / C3`
- example:

```text
before timestamps: 00:00, 01:00, 02:00, 03:00, 04:00
after timestamps : 00:00, 01:00, 03:00, 04:00
```

### Accuracy injections

`negative_values`

- operation: replace selected values with negative demand
- dimension/subcheck: `accuracy / A1`
- example:

```text
before: [8, 9, 10, 11]
after : [8, -9, 10, -11]
```

`impossible_high`

- operation: replace selected values with a high bound anchor times `high_multiplier`
- dimension/subcheck: `accuracy / A1`
- example:

```text
normal upper region: around 25
high_multiplier: 3.0
after injected value: around 75
```

`spikes`

- operation: add a large positive perturbation based on the client scale
- dimension/subcheck: `accuracy / A2`
- example:

```text
before: [10, 11, 10, 12, 11]
after : [10, 11, 26, 12, 11]
```

`noisy_values`

- operation: add smaller random noise to selected points
- dimension/subcheck: `accuracy / A2`
- example:

```text
before: [10, 10, 10, 10]
after : [10.0, 10.4, 9.7, 10.2]
```

`rate_jumps`

- operation: replace `x_t` using the previous value plus or minus a large jump based on client scale
- dimension/subcheck: `accuracy / A3`
- example:

```text
before: [10, 11, 12, 13]
after : [10, 11, 40, 13]
```

`stuck_constant`

- operation: choose a block and force all values in the block to the same constant
- dimension/subcheck: `accuracy / A3`
- example:

```text
before: [12, 13, 14, 15, 16, 17]
after : [12, 13, 14, 14, 14, 14]
```

### Temporal consistency injections

`stale_block`

- operation: copy earlier values forward using a sampled lag such as 1 hour, 2 hours, or 24 hours
- dimension/subcheck: `temporal_consistency / TC1`
- example with lag 1:

```text
before: [10, 11, 12, 13, 14, 15]
after : [10, 11, 12, 12, 13, 14]
```

`duplicated_timestamps`

- operation: duplicate whole rows, so the same timestamp appears twice in the table
- dimension/subcheck: mainly `temporal_consistency / TC3`
- example:

```text
before timestamps: 00:00, 01:00, 02:00
after timestamps : 00:00, 01:00, 01:00, 02:00
```

`calendar_hole`

- operation: choose a localized time window, then remove some day-of-week × hour-of-day cells only inside that window
- dimension/subcheck: `temporal_consistency / TC3`
- example:

```text
remove client observations at selected weekday-hour cells inside one local 3-14 day window
```

`mid_window_regime_shift`

- operation: add a level shift to the second half of one client series so the window becomes non-stationary
- dimension/subcheck: `temporal_consistency / TC4`
- example:

```text
before:
first half  -> around 15
second half -> around 15

after:
first half  -> around 15
second half -> around 19
```

### Skewness injections

`mean_shift`

- operation: add a constant offset based on one client standard deviation, but only inside a sampled local block
- dimension/subcheck: `skewness / SK1`
- example:

```text
before: [10, 12, 14, 16, 18, 20]
after : [10, 12, 17, 19, 21, 20]
```

`variance_shift`

- operation: keep the approximate center, but stretch the deviations from the center inside a temporary regime block
- dimension/subcheck: `skewness / SK3`
- example:

```text
before: center 15, values [13, 14, 16, 17]
after : center 15, values [11, 13, 17, 19]
```

`peak_hour_shift`

- operation: multiply values only during configured peak hours and only inside a sampled local window
- dimension/subcheck: `skewness / SK2`
- example with peak hours 17:00 to 21:00:

```text
before peak values: [20, 22, 24]
after peak values : [30, 33, 36]
```

`long_term_shift`

- operation: shift a longer partial-month style block upward or downward
- dimension/subcheck: `skewness / SK3`
- example:

```text
before:
outside block -> around 12
inside block  -> around 12

after:
outside block -> around 12
inside block  -> around 17
```

Fault report columns:

- `client_id`
- `fault_type`
- `dimension`
- `subcheck`
- `start_time`
- `end_time`
- `affected_rows`
- `affected_fraction`
- `parameters_used`
- `shared_timestamp_event`

The report now contains two kinds of rows:

- event rows:
  - one row per injected fault event
- final net summary rows:
  - `fault_type = final_net_corruption`
  - one row per selected client
  - one overall row with `client_id = __all_selected_clients__`
  - `affected_rows` and `affected_fraction` describe the final net number and percentage of original rows that ended up corrupted after all overlaps

## 3. `score_data_quality.py`

Purpose:

- load the faulty wide-format demand CSV
- compare it against the clean reference CSV
- compute per-client subcheck scores and per-dimension scores

Typical use:

```bash
python scripts/score_data_quality.py \
  --input data/demand_faulty.csv \
  --clean-reference data/demand.csv \
  --output output/client_quality_scores.csv \
  --timestamp-col timestamp \
  --config configs/data_quality_scoring.yaml
```

Arguments:

- `--input`: faulty demand CSV
- `--clean-reference`: clean reference CSV
- `--output`: output CSV with quality scores
- `--timestamp-col`: timestamp column name; auto-detected if omitted
- `--config`: YAML scoring config

Main output:

- `output/client_quality_scores.csv`

Output columns:

- `client_id`
- `C1`, `C2`, `C3`, `completeness`
- `A1`, `A2`, `A3`, `accuracy`
- `TC1`, `TC2`, `TC3`, `TC4`, `temporal_consistency`
- `SK1`, `SK2`, `SK3`, `skewness`
- `sample_count`

Scoring behavior:

- every subcheck is in `[0, 1]`
- every dimension is in `[0, 1]`
- default dimension aggregation uses equal subcheck weights
- custom subcheck weights can be provided in `configs/data_quality_scoring.yaml`

Dimensions:

- completeness
- accuracy
- temporal consistency
- skewness

## 4. `compute_pca_quality_score.py`

Purpose:

- take the per-client dimension scores
- fit a one-component PCA model
- produce one final scalar quality score per client
- convert that scalar into aggregation weights

Typical use:

```bash
python scripts/compute_pca_quality_score.py \
  --input output/client_quality_scores.csv \
  --output output/client_quality_scores_with_pca.csv \
  --config configs/data_quality_scoring.yaml \
  --model-output output/pca_quality_model.json
```

Arguments:

- `--input`: per-client quality score CSV
- `--output`: PCA score and aggregation weight CSV
- `--config`: YAML config
- `--model-output`: optional JSON file for saved PCA parameters

Main outputs:

- `output/client_quality_scores_with_pca.csv`
- optional `output/pca_quality_model.json`

Output columns:

- `client_id`
- `completeness`
- `accuracy`
- `temporal_consistency`
- `skewness`
- `pca_raw_score`
- `quality_score`
- `sample_count`
- `aggregation_weight`

Weight behavior:

- higher quality dimensions should produce higher `quality_score`
- `aggregation_weight` is normalized to sum to `1`
- if PCA becomes degenerate, the implementation falls back to a safer mean-based path

## 5. `run_quality_pipeline.py`

Purpose:

- run the full pipeline in one command

Pipeline order:

1. inject faults
2. compute per-client quality scores
3. compute PCA composite scores
4. compute aggregation weights

Typical use:

```bash
python scripts/run_quality_pipeline.py \
  --input data/demand.csv \
  --faulty-output data/demand_faulty.csv \
  --fault-report output/fault_injection_report.csv \
  --score-output output/client_quality_scores.csv \
  --pca-output output/client_quality_scores_with_pca.csv \
  --fault-config configs/fault_injection.yaml \
  --score-config configs/data_quality_scoring.yaml \
  --timestamp-col timestamp \
  --seed 42
```

Arguments:

- `--input`: clean demand CSV
- `--faulty-output`: faulty output CSV
- `--fault-report`: fault report CSV
- `--score-output`: per-client dimension score CSV
- `--pca-output`: PCA and aggregation-weight CSV
- `--fault-config`: fault injection YAML
- `--score-config`: scoring YAML
- `--timestamp-col`: timestamp column name; auto-detected if omitted
- `--seed`: random seed
- `--pca-model-output`: optional JSON path for PCA model parameters

Main outputs:

- `data/demand_faulty.csv`
- `output/fault_injection_report.csv`
- `output/client_quality_scores.csv`
- `output/client_quality_scores_with_pca.csv`

## Config Files

The scripts rely on:

- `configs/fault_injection.yaml`
- `configs/data_quality_scoring.yaml`

Use these to control:

- selected clients
- fault rates and fault strengths
- expected frequency
- anomaly thresholds
- JSD histogram settings
- staleness horizon
- peak-hour definition
- subcheck weights
- PCA epsilon settings

## How `data_quality_scoring.yaml` Numbers Were Chosen

These defaults are conservative starting values for hourly demand data with a clean historical reference.

They were chosen to satisfy three goals:

- keep clean data near score `1`
- make injected faults measurably reduce the intended dimension
- avoid using thresholds so tight that normal load variation looks faulty

### Frequency

- `frequency: "1h"` is the default because the current examples and test fixtures are hourly
- if your real data is `15min`, `30min`, or daily, change this first before tuning anything else

### Accuracy config

- `lower_bound: 0.0` assumes demand cannot be negative
- `upper_bound:` is left empty on purpose so the scorer can derive a client-specific ceiling from clean data
- `upper_multiplier: 1.5` allows moderate headroom above the clean `Q99.9` before values are marked out of range
- `deviation_method: rolling_median` is the most robust default when the series is noisy or not perfectly seasonal
- `rolling_window: 24` means compare each hourly point with roughly the last day of history
- `min_history: 6` avoids making anomaly decisions too early in the series
- `mad_multiplier: 4.0` is loose enough to tolerate normal variation but still catch clear spikes
- `min_threshold_abs: 0.1` prevents near-zero series from getting tiny anomaly thresholds
- `seasonal_period: 24` is the natural fallback for hourly daily seasonality
- `seasonal_threshold_multiplier: 3.0` is a moderate seasonal anomaly threshold
- `roc_multiplier: 1.5` means the allowed rate-of-change is based on clean high-end changes plus some margin
- `max_stuck_len: 6` treats 6 or more repeated hourly points as a suspicious freeze candidate
- `freeze_atol: 0.0` means exact equality counts as stuck by default; increase this if your meter reports rounded values

### Completeness config

- completeness now uses only `C1`, `C2`, and `C3`
- there is no `C4` field-coverage subcheck in the current demand scorer

### Temporal consistency config

- `staleness_tau: "48h"` gives a two-day grace period before staleness falls to zero; for operational monitoring you may want this shorter
- `autocorr_lags: [1, 24, 168]` compares local autocorrelation against hourly, daily, and weekly references
- `autocorr_threshold: 0.25` is a moderate tolerance for autocorrelation mismatch
- `calendar_n_min: 1` means a calendar cell counts as covered if it has at least one valid observation; increase this if you want stricter coverage
- `stationarity_segments: 4` splits the window into four contiguous pieces for the TC4 stationarity check
- `stationarity_tolerance: 1.0` is a moderate tolerance for segment-mean instability
- `epsilon: 1e-6` is there for numerical smoothing in the temporal calculations

### Skewness config

- `jsd_bins: 20` is a reasonable starting point for client-level histogram comparison
- `peak_hours: [17, 18, 19, 20, 21]` was chosen as a common evening peak window for electricity demand; change it if your domain peaks elsewhere
- `allowed_peak_ratio_shift: 0.5` means a peak/off-peak ratio change of about 0.5 starts to saturate the penalty
- `moment_shift_threshold: 3.0` means combined mean and standard-deviation drift is tolerated up to a moderate normalized level before the score collapses

### Subcheck weights

- all default weights are `1.0`
- this means equal weighting inside each dimension
- change these only when you want a clear policy decision, for example:
  - raise `C3` if continuity matters more than pointwise missingness
  - raise `A3` if sensor freezing is more important than isolated spikes
  - raise `TC4` if mid-window stationarity is your main concern
  - raise `SK2` if peak-time behavior matters more than overall distribution shift

### PCA config

- `epsilon: 1e-9` is only for numerical safety when scores are nearly identical

## How to Calibrate the YAML Values on Real Data

Recommended order:

1. set the correct `frequency`
2. run the scorer on clean data only
3. confirm most clients score high, ideally close to `1`
4. inject one fault family at a time
5. check that the intended dimension drops more than the others
6. only then tighten or loosen thresholds

Practical calibration rules:

- if clean clients score too low on `A2`, increase `mad_multiplier` or use `seasonal_naive`
- if too many normal points fail `A1`, increase `upper_multiplier` or set a fixed `upper_bound`
- if `TC3` is always too harsh, reduce `calendar_n_min`
- if `SK1` or `TC2` are unstable on sparse clients, reduce histogram bins from `20` to `10`
- if `quality_score` is dominated by one dimension, inspect the raw dimension CSV before changing PCA

## Recommended Workflow

If you want intermediate files and checks, run:

1. `inject_faults.py`
2. `score_data_quality.py`
3. `compute_pca_quality_score.py`

If you want one command, run:

1. `run_quality_pipeline.py`

## Federated Learning Use

The final file `client_quality_scores_with_pca.csv` is the one intended for later federated aggregation.

Typical usage:

- use `quality_score` for filtering or diagnostics
- use `aggregation_weight` as the server-side client aggregation coefficient
- use the raw dimension columns for fairness, mitigation, or ablation analysis

## Notes

- all scripts assume wide-format demand data
- timestamp auto-detection is supported, but passing `--timestamp-col` is safer
- the clean input file is never modified in place
- deterministic behavior depends on the provided seed
