# Demand Quality Pipeline for SplitFed Smart-Meter Forecasting

## 1. Overview

This package implements a wide-format smart-meter data-quality pipeline under `src/demand_quality/`. It is adapted from the DQOps / DQSOps repository that originally targeted Open RAN 5G telemetry quality scoring, but the demand pipeline is isolated from the telecom-specific logic.

The demand pipeline is designed for smart-meter time-series forecasting and later SplitFed integration. Its purpose is to:

1. load and normalize wide demand data,
2. calibrate per-client historical baselines,
3. inject synthetic faults into selected client series,
4. compute local rule-based quality scores,
5. collapse the four dimension scores into a single scalar by PCA,
6. export outputs that can later be used as client-side quality labels or federated aggregation weights.

The four raw-data quality dimensions are:

- Completeness
- Accuracy
- Temporal Consistency
- Skewness

The final scalar `quality_score` is PCA-derived and lives in `[0, 1]`. It is intended to become the supervised label for a later XGBoost regression model that can approximate the slower rule-based + PCA pipeline during SplitFed runtime.

### Raw Demand Sign Convention

The raw smart-meter dataset stores demand as negative values. Before any baseline calibration, fault injection, scoring, or later machine-learning dataset preparation, the shared demand loading path normalizes demand to positive consumption.

Conceptually, the conversion is:

`y = -y`

Implementation detail:

- the shared loader in [src/demand_quality/io.py](/Users/meilyoeng/Documents/MAID/summer sem/Thesis Project/dqops_5g-main/src/demand_quality/io.py:1) applies `y = -y` to client columns that are predominantly negative
- columns that are already mostly positive are left unchanged
- this avoids accidentally double-flipping already processed or fault-injected demand CSVs when they are reloaded later

After loading, all DQ formulas assume:

- positive kWh-like consumption values
- higher values mean larger consumption
- negative values appearing later are treated as injected accuracy faults or malformed data

## 2. Privacy and SplitFed Assumptions

This demand-quality design follows SplitFed privacy constraints:

- raw demand data stays local to the client,
- the SplitFed server receives smashed activations, not raw usage data,
- this scorer is a raw-data local scorer, not an activation-monitoring scorer,
- no global raw-data distribution checks are used in the aligned temporal or skewness dimensions.

Implications:

- `TC2` is computed from the client's own historical autocorrelation baseline,
- `TC4` is computed from the client's own historical stationarity baseline,
- `SK1`, `SK2`, and `SK3` are computed from the client's own historical baseline,
- the scorer may compare a faulty local window against a clean local reference or a baseline artifact derived from that same client history,
- cross-client raw-distribution comparisons are intentionally avoided in the aligned scorer.

## 3. End-to-End Pipeline

The intended end-to-end workflow is:

1. Prepare and normalize raw data
   - wide CSV with one timestamp column and many client columns
   - timestamp parsing, sorting, client-column detection, frequency inference
   - raw negative demand is flipped to positive consumption
2. Calibrate historical baselines
   - derive per-client baseline statistics from clean historical demand
   - baseline includes histogram, autocorrelation, peak ratio, mean/std, and stationarity reference
3. Inject faults
   - inject synthetic faults per selected client
   - record event metadata for each injected fault
4. Rule-based dimension scoring
   - compute Completeness, Accuracy, Temporal Consistency, Skewness
   - each dimension score is in `[0, 1]`, where `1` means better quality
5. PCA label generation
   - use the four dimension scores as the PCA input vector
   - derive `pca_raw_score`, then min-max normalize to `quality_score`
6. XGBoost training
   - not yet ported into `src/demand_quality/`
   - current repo contains reusable but O-RAN-oriented XGBoost/Optuna code elsewhere
7. Runtime SplitFed prediction
   - future stage: replace slow rule-based scoring with a trained local XGBoost regressor

Correct order:

```text
raw negative demand
  -> flip sign y = -y
  -> calibrate clean baseline
  -> inject faults
  -> score faulty windows against clean baseline
  -> PCA quality label
  -> train XGBoost
```

## 4. Package Structure

### Implemented demand package

- `src/demand_quality/schema.py`
  - timestamp detection
  - client-column detection
  - frequency inference
  - expected timestamp grid creation
- `src/demand_quality/io.py`
  - CSV loading
  - YAML loading
  - CSV/JSON saving
- `src/demand_quality/statistics.py`
  - histogram helpers
  - divergence helpers used for local distribution comparison
  - rolling anomaly detection
  - stuck-segment detection
  - calendar coverage
  - autocorrelation and segment-stability helpers
- `src/demand_quality/baseline.py`
  - per-client local baseline calibration from clean demand history
- `src/demand_quality/fault_injection.py`
  - detailed and simple fault injection modes
  - event metadata reporting
- `src/demand_quality/scoring.py`
  - rule-based dimension scoring
- `src/demand_quality/pca_quality.py`
  - one-component PCA quality aggregation
- `src/demand_quality/pipeline.py`
  - thin orchestration wrapper for fault injection -> scoring -> PCA

### Thin CLIs

- `scripts/calibrate_demand_baseline.py`
  - calibrate local baseline artifact from clean history
- `scripts/inject_faults.py`
  - inject demand faults and write report
- `scripts/score_data_quality.py`
  - compute per-client subcheck and dimension scores
- `scripts/compute_pca_quality_score.py`
  - compute PCA `quality_score` and `aggregation_weight`
- `scripts/run_quality_pipeline.py`
  - run the implemented end-to-end demand pipeline

### Legacy reusable but not yet ported to demand package

- `src/training/model_trainer.py`
  - XGBoost + Optuna regression training
  - generic in objective shape, but not isolated as a demand module
- `src/deployment/dq_predictor.py`
  - model loading and clipped `[0, 1]` prediction wrapper
- `src/data_processing/feature_extractor.py`
  - feature extraction logic, but strongly tied to O-RAN cell/UE schema
- `src/training/label_generator.py`
  - PCA label-generation pattern, but built around window folders with cell/UE parquet files

## 5. Data Flow

### Input files

- clean demand CSV
- optional clean historical demand CSV for baseline calibration
- YAML configs:
  - `configs/fault_injection.yaml`
  - `configs/data_quality_scoring.yaml`

### Intermediate files

- baseline artifact JSON
- fault injection report CSV
- per-client dimension score CSV

### Output files

- `data/demand_faulty.csv`
- `output/fault_injection_report.csv`
- `output/client_quality_scores.csv`
- `output/client_quality_scores_with_pca.csv`

### Expected main columns

`client_quality_scores.csv`:

- `client_id`
- `C1`, `C2`, `C3`, `completeness`
- `A1`, `A2`, `A3`, `accuracy`
- `TC1`, `TC2`, `TC3`, `TC4`, `temporal_consistency`
- `SK1`, `SK2`, `SK3`, `skewness`
- `sample_count`

`client_quality_scores_with_pca.csv`:

- `client_id`
- `completeness`
- `accuracy`
- `temporal_consistency`
- `skewness`
- `pca_raw_score`
- `quality_score`
- `sample_count`
- `aggregation_weight`

Notes:

- completeness uses only `C1`, `C2`, and `C3`
- the shared load path converts predominantly negative raw demand columns into positive consumption before any downstream stage

### 5.1 What Baseline Calibration Does

Historical baseline calibration takes clean demand history and computes per-client reference statistics that define normal local behavior.

That clean historical data goes through the same shared load path first, so raw negative demand is flipped to positive consumption before baseline statistics are computed.

The implemented baseline utility in [src/demand_quality/baseline.py](/Users/meilyoeng/Documents/MAID/summer sem/Thesis Project/dqops_5g-main/src/demand_quality/baseline.py:1) stores or derives:

- valid lower/upper range proxy
- mean and standard deviation
- histogram edges and histogram probabilities
- reference peak/off-peak ratio
- reference autocorrelation at configured lags, typically `1`, `24`, and `168`
- stationarity reference `sigma_ref` from segment-mean variability

Why calibration must happen before fault injection:

- baselines should represent normal clean behavior
- if calibration is done on faulty data, the baseline will absorb the corruption and make injected faults harder to detect
- the scorer needs a clean reference to decide whether a new window is stale, shifted, drifted, or structurally abnormal

In the current repository there are two practical baseline modes:

- explicit baseline artifact:
  - use [scripts/calibrate_demand_baseline.py](/Users/meilyoeng/Documents/MAID/summer sem/Thesis Project/dqops_5g-main/scripts/calibrate_demand_baseline.py:1)
  - inspect or save the baseline JSON
- on-the-fly clean reference scoring:
  - `score_data_quality.py` currently recomputes equivalent local reference statistics directly from the clean reference CSV passed in at runtime

## 6. Fault Injection System

### 6.1 Fault Injection Configuration Overview

Fault injection is controlled by the YAML file [configs/fault_injection.yaml](/Users/meilyoeng/Documents/MAID/summer sem/Thesis Project/dqops_5g-main/configs/fault_injection.yaml:1).

The actual CLI that consumes this file is:

- [scripts/inject_faults.py](/Users/meilyoeng/Documents/MAID/summer sem/Thesis Project/dqops_5g-main/scripts/inject_faults.py:1)

The full end-to-end CLI [scripts/run_quality_pipeline.py](/Users/meilyoeng/Documents/MAID/summer sem/Thesis Project/dqops_5g-main/scripts/run_quality_pipeline.py:1) also uses the same fault YAML when it calls `run_quality_pipeline(...)`.

Load path:

1. `scripts/inject_faults.py` reads the YAML with `demand_quality.io.load_yaml_config(...)`.
2. The parsed dictionary is passed into `demand_quality.fault_injection.inject_faults_to_csv(...)`.
3. `inject_faults_to_csv(...)` creates a `WideDemandFaultInjector`.
4. `WideDemandFaultInjector.inject(...)` mutates a copy of the clean wide-format demand frame and returns:
   - `faulty_df`
   - `report_df`

Relationship between inputs and outputs:

- clean input data:
  - the original wide demand CSV with one shared timestamp column and many client columns
- YAML fault configuration:
  - tells the injector which clients can be touched and how much corruption to apply
- generated faulty data:
  - a mutated copy of the clean table saved as `demand_faulty.csv`
- fault metadata:
  - one row per injected event in `fault_injection_report.csv`
- downstream DQ scoring:
  - `score_data_quality.py` compares the faulty data against a clean reference and converts the injected corruption into subcheck and dimension scores

Important implementation facts:

- the current repository uses `scripts/inject_faults.py`, not `scripts/inject_demand_faults.py`
- file paths are CLI arguments, not YAML keys
- the random seed is also a CLI argument `--seed`, not a YAML field
- only selected clients receive client-level faults
- shared timestamp-row faults affect all clients because the table is wide and the timestamp grid is shared

#### YAML vs CLI Responsibility

| Concern | Controlled by | Where | Example |
|---|---|---|---|
| fault type and severity | YAML | `configs/fault_injection.yaml` | `random_missing_rate: 0.02` |
| which clients are eligible for faults | YAML | `configs/fault_injection.yaml` | `clients.selected: [client_a, client_b]` |
| simple vs detailed injection mode | YAML | `configs/fault_injection.yaml` | `fault_fraction: 0.05` |
| clean input CSV path | CLI | `scripts/inject_faults.py` or `scripts/run_quality_pipeline.py` | `--input data/demand.csv` |
| faulty output CSV path | CLI | `scripts/inject_faults.py` or `scripts/run_quality_pipeline.py` | `--output data/demand_faulty.csv` or `--faulty-output ...` |
| fault report path | CLI | `scripts/inject_faults.py` or `scripts/run_quality_pipeline.py` | `--report output/fault_injection_report.csv` |
| timestamp column name | CLI | injector and scorer CLIs | `--timestamp-col timestamp` |
| random seed | CLI | injector and pipeline CLIs | `--seed 42` |
| score config | separate YAML | `configs/data_quality_scoring.yaml` | `--config ...` or `--score-config ...` |
| scoring input/output paths | CLI | `scripts/score_data_quality.py` | `--input ... --output ...` |

Two injector modes exist:

- detailed mode:
  - active when `fault_fraction` is empty or `null`
  - uses the nested `faults:` section
  - rates are scaled by `global_fault_rate * per_client_fault_rate[client]`
- simple selected-client mode:
  - active when `fault_fraction` is set
  - ignores the nested detailed `faults:` rates
  - applies approximately `round(valid_rows * fault_fraction)` point faults per selected client

Client selection logic:

1. start from all detected client columns,
2. if `clients.selected` is non-empty, restrict candidates to those names,
3. if that restriction produces no matches, the code falls back to all detected clients,
4. if `clients.random_count` is set, sample exactly that many candidates,
5. else if `clients.random_fraction < 1.0`, sample that fraction of candidates,
6. else use all remaining candidates.

Event ordering inside the injector:

1. detect selected clients,
2. if `fault_fraction` is set, run simple mode and stop,
3. otherwise create shared row events for dropped or duplicated timestamps,
4. for each selected client, apply:
   - accuracy faults,
   - temporal faults,
   - skewness faults,
   - completeness faults,
5. append shared duplicate rows if configured,
6. drop shared rows if configured,
7. sort and save the final event report.

Exact count conversion rules:

- point-rate faults use:
  - `count = max(1, round(valid_positions * rate))` when `rate > 0`
- shared row-fraction faults use:
  - `count = max(1, round(n_rows * fraction))` when `fraction > 0`
- block event counts use:
  - `round(events_per_client * client_factor)`

This means very small positive rates still inject at least one event if valid data exists.

### 6.2 YAML Configuration Reference

The following tables document the YAML keys actually present in [configs/fault_injection.yaml](/Users/meilyoeng/Documents/MAID/summer sem/Thesis Project/dqops_5g-main/configs/fault_injection.yaml:1) and used by [src/demand_quality/fault_injection.py](/Users/meilyoeng/Documents/MAID/summer sem/Thesis Project/dqops_5g-main/src/demand_quality/fault_injection.py:1).

#### Top-level mode and client-selection keys

| YAML key | Type | Default | Valid values | Meaning | Used by | Expected DQ effect |
|---|---|---:|---|---|---|---|
| `fault_fraction` | float or null | `null` | `null` or `[0, 1]` | Switches the injector into simple selected-client mode when set | `inject()` -> `_inject_simple_fraction_mode()` | Mixed; depends on sampled simple fault types |
| `simple_fault_types` | list[str] | `[missing, out_of_range, inconsistent, timeliness]` | subset of `missing`, `out_of_range`, `inconsistent`, `timeliness`, `noise`, `spike` | Fault pool used only in simple mode | `_inject_simple_fraction_mode()` | Mixed; maps to completeness, accuracy, or temporal consistency |
| `simple_fault_probabilities` | list[float] or null | `null` | non-negative list matching `simple_fault_types` length | Sampling weights for simple fault types; normalized internally | `_inject_simple_fraction_mode()` | Mixed; biases which dimensions degrade most |
| `simple_out_of_range_range` | list[float, float] | `[1.5, 3.0]` | positive low/high | Uniform multiplier range for simple `out_of_range` faults | `_inject_simple_fraction_mode()` | Mainly `A1` |
| `simple_inconsistent_range` | list[float, float] | `[4.0, 8.0]` | positive low/high | Uniform multiplier range for simple negative `inconsistent` faults | `_inject_simple_fraction_mode()` | Mainly `A1`, `A3_nonneg` |
| `simple_timeliness_steps` | list[int] | `[-1, 1]` | non-zero integer steps | Neighbor offsets used by simple `timeliness` faults | `_inject_simple_fraction_mode()` | Mainly `TC1` |
| `simple_noise_std_fraction` | float | `0.05` | `>= 0` | Std as fraction of client scale for simple `noise` faults; dormant unless `noise` is in `simple_fault_types` | `_inject_simple_fraction_mode()` | Mainly `A2` |
| `simple_spike_multiplier` | float | `4.0` | `> 0` | Scale multiplier for simple `spike` faults; dormant unless `spike` is in `simple_fault_types` | `_inject_simple_fraction_mode()` | Mainly `A2` |
| `global_fault_rate` | float | `1.0` | practical range `>= 0` | Global multiplier applied to detailed client-level rates; ignored in simple mode | `inject()` detailed path | Mixed; scales detailed fault intensity |
| `per_client_fault_rate` | mapping `{client_id: float}` | `{}` | per-client positive multipliers | Client-specific detailed-mode intensity scaling; ignored in simple mode | `inject()` detailed path | Mixed; lets some clients be more corrupted than others |
| `clients.selected` | list[str] | `[]` | client column names | Restricts the candidate client pool before any random subsampling | `_select_clients()` | Indirect; controls which clients can be degraded |
| `clients.random_fraction` | float | `1.0` | practical range `[0, 1]` | Randomly sample a fraction of the candidate clients | `_select_clients()` | Indirect |
| `clients.random_count` | int or null | `null` | positive integer or `null` | Randomly sample an exact number of candidate clients; overrides `random_fraction` when set | `_select_clients()` | Indirect |

Implementation notes:

- `seed` is not a YAML key; use the CLI argument `--seed`
- `input`, `output`, and `report` paths are not YAML keys; use CLI arguments
- `global_fault_rate` and `per_client_fault_rate` do nothing when `fault_fraction` is active

Practical interpretation of `fault_fraction` in simple mode:

- `0.01`
  - very light corruption
  - about 1% of valid timestamps per selected client are faulted
- `0.05`
  - moderate corruption
- `0.10`
  - strong corruption
- `> 0.20`
  - severe synthetic corruption
  - usually better for stress tests than for realistic training data

#### Detailed completeness-fault keys

| YAML key | Type | Default | Valid values | Meaning | Used by | Expected DQ effect |
|---|---|---:|---|---|---|---|
| `faults.completeness.client_dropout_fraction` | float | `0.0` | practical range `[0, 1]` | Probability-like trigger for wiping an entire selected client column to `NaN` in detailed mode | `_inject_completeness_faults()` | Mainly `C2`, also `C1`, `TC1`, `TC3`, skewness collapse |
| `faults.completeness.random_missing_rate` | float | `0.02` | practical range `[0, 1]` | Sparse missing-point rate per selected client after scaling | `_inject_completeness_faults()` | Mainly `C1` |
| `faults.completeness.dropped_timestamp_rows_fraction` | float | `0.0` | practical range `[0, 1]` | Fraction of shared rows removed from the entire table | `inject()` shared event path | Mainly `C3`, secondary `TC3` |
| `faults.completeness.block_missing.events_per_client` | int/float | `1` | `>= 0` | Base number of missing blocks per client before scaling and rounding | `_inject_completeness_faults()` | Mainly `C1`, secondary `C3` |
| `faults.completeness.block_missing.min_length` | int | `4` | positive integer | Minimum contiguous block length in rows | `_inject_completeness_faults()` | Mainly `C1`, secondary `C3` |
| `faults.completeness.block_missing.max_length` | int | `12` | positive integer `>= min_length` | Maximum contiguous block length in rows | `_inject_completeness_faults()` | Mainly `C1`, secondary `C3` |

##### Completeness fault keys

This group simulates missing measurements, missing client presence, and broken timestamp continuity.

Keys in this group:

- `client_dropout_fraction`
- `random_missing_rate`
- `dropped_timestamp_rows_fraction`
- `block_missing.events_per_client`
- `block_missing.min_length`
- `block_missing.max_length`

Main expected subchecks to decrease:

- `C1` for sparse or block missingness
- `C2` for full client dropout
- `C3` for shared timestamp row removal and long continuity gaps

Practical recommendations:

- `random_missing_rate`
  - `0.01`: light sparse missingness
  - `0.05`: moderate
  - `0.10`: severe
- `dropped_timestamp_rows_fraction`
  - `0.005` to `0.02`: recommended
  - high values are harsh because they affect all clients
- `block_missing.events_per_client`
  - `1`: one outage block per client
  - `2` to `3`: repeated outages
- `block_missing.min_length` / `max_length`
  - for hourly data, `4` to `12` means a 4 to 12 hour outage
  - `24` means about one full day outage

Light / moderate / severe examples:

- light:
  - `random_missing_rate: 0.01`
  - `block_missing.events_per_client: 1`
  - `block_missing.min_length: 3`
  - `block_missing.max_length: 4`
- moderate:
  - `random_missing_rate: 0.03`
  - `dropped_timestamp_rows_fraction: 0.01`
  - `block_missing.events_per_client: 1`
  - `block_missing.min_length: 4`
  - `block_missing.max_length: 12`
- severe:
  - `random_missing_rate: 0.10`
  - `dropped_timestamp_rows_fraction: 0.02`
  - `block_missing.events_per_client: 2`
  - `block_missing.min_length: 12`
  - `block_missing.max_length: 24`

#### Detailed accuracy-fault keys

| YAML key | Type | Default | Valid values | Meaning | Used by | Expected DQ effect |
|---|---|---:|---|---|---|---|
| `faults.accuracy.negative_rate` | float | `0.005` | practical range `[0, 1]` | Sparse rate for forcing selected values negative | `_inject_accuracy_faults()` | Mainly `A1`, `A3_nonneg` |
| `faults.accuracy.impossible_high_rate` | float | `0.005` | practical range `[0, 1]` | Sparse rate for forcing selected values to a high out-of-range anchor | `_inject_accuracy_faults()` | Mainly `A1` |
| `faults.accuracy.high_multiplier` | float | `3.0` | `> 0` | Multiplier applied to the per-client `upper_anchor` during `impossible_high` faults | `_inject_accuracy_faults()` | Mainly `A1` |
| `faults.accuracy.spike_rate` | float | `0.01` | practical range `[0, 1]` | Sparse rate for large positive spikes | `_inject_accuracy_faults()` | Mainly `A2` |
| `faults.accuracy.spike_multiplier` | float | `4.0` | `> 0` | Added spike amplitude in units of client scale | `_inject_accuracy_faults()` | Mainly `A2` |
| `faults.accuracy.noise_rate` | float | `0.02` | practical range `[0, 1]` | Sparse rate for additive Gaussian noise | `_inject_accuracy_faults()` | Mainly `A2` |
| `faults.accuracy.noise_std_fraction` | float | `0.05` | `>= 0` | Noise std as a fraction of client scale | `_inject_accuracy_faults()` | Mainly `A2` |
| `faults.accuracy.jump_rate` | float | `0.01` | practical range `[0, 1]` | Sparse rate for abrupt rate-of-change jumps | `_inject_accuracy_faults()` | Mainly `A3_roc` |
| `faults.accuracy.jump_multiplier` | float | `6.0` | `> 0` | Jump amplitude in units of client scale relative to the previous reading | `_inject_accuracy_faults()` | Mainly `A3_roc` |
| `faults.accuracy.stuck_segments_per_client` | int/float | `1` | `>= 0` | Base number of frozen-value blocks per client before scaling and rounding | `_inject_accuracy_faults()` | Mainly `A3_freeze` |
| `faults.accuracy.stuck_min_length` | int | `4` | positive integer | Minimum frozen-run length in rows | `_inject_accuracy_faults()` | Mainly `A3_freeze` |
| `faults.accuracy.stuck_max_length` | int | `12` | positive integer `>= min_length` | Maximum frozen-run length in rows | `_inject_accuracy_faults()` | Mainly `A3_freeze` |

##### Accuracy fault keys

This group simulates pointwise value corruption, implausible magnitudes, abrupt jumps, and frozen sensors.

Keys in this group:

- `negative_rate`
- `impossible_high_rate`
- `high_multiplier`
- `spike_rate`
- `spike_multiplier`
- `noise_rate`
- `noise_std_fraction`
- `jump_rate`
- `jump_multiplier`
- `stuck_segments_per_client`
- `stuck_min_length`
- `stuck_max_length`

Main expected subchecks to decrease:

- `A1` for negative and impossible-high values
- `A2` for spikes and noise
- `A3` for jumps and frozen runs

Practical recommendations:

- `negative_rate`
  - `0.001` to `0.01` is usually enough
- `impossible_high_rate`
  - `0.001` to `0.01` is usually enough
- `high_multiplier`
  - `2` to `3`: moderate
  - `5+`: extreme
- `spike_rate`
  - `0.005` to `0.02`: recommended
- `spike_multiplier`
  - `3` to `5`: moderate
  - `8+`: severe
- `noise_rate`
  - `0.01` to `0.05`: recommended
- `noise_std_fraction`
  - `0.02`: light
  - `0.05`: moderate
  - `0.10`: strong
- `jump_rate`
  - `0.005` to `0.02`: recommended
- `jump_multiplier`
  - `4` to `8`: typical
- `stuck_segments_per_client`
  - `1`: one frozen period
  - `2` to `3`: multiple frozen periods
- `stuck_min_length` / `stuck_max_length`
  - values above `3` are usually needed to affect frozen-run detection
  - `4` to `12` is a good hourly-data default

Light / moderate / severe examples:

- light:
  - `negative_rate: 0.001`
  - `impossible_high_rate: 0.001`
  - `spike_rate: 0.005`
  - `noise_rate: 0.01`
  - `noise_std_fraction: 0.02`
- moderate:
  - `negative_rate: 0.005`
  - `impossible_high_rate: 0.005`
  - `spike_rate: 0.01`
  - `spike_multiplier: 4.0`
  - `jump_rate: 0.01`
  - `jump_multiplier: 6.0`
  - `stuck_segments_per_client: 1`
- severe:
  - `negative_rate: 0.01`
  - `impossible_high_rate: 0.01`
  - `high_multiplier: 5.0`
  - `spike_rate: 0.02`
  - `spike_multiplier: 8.0`
  - `noise_std_fraction: 0.10`
  - `stuck_segments_per_client: 3`

#### Detailed temporal-consistency-fault keys

| YAML key | Type | Default | Valid values | Meaning | Used by | Expected DQ effect |
|---|---|---:|---|---|---|---|
| `faults.temporal_consistency.stale_blocks_per_client` | int/float | `1` | `>= 0` | Base number of stale blocks per client before scaling and rounding | `_inject_temporal_faults()` | Mainly `TC1` |
| `faults.temporal_consistency.stale_min_length` | int | `3` | positive integer | Minimum stale block length in rows | `_inject_temporal_faults()` | Mainly `TC1` |
| `faults.temporal_consistency.stale_max_length` | int | `8` | positive integer `>= min_length` | Maximum stale block length in rows | `_inject_temporal_faults()` | Mainly `TC1` |
| `faults.temporal_consistency.lag_choices` | list[int] | `[1, 2, 24]` | positive integer row lags | Candidate lags used when copying stale values | `_inject_temporal_faults()` | Mainly `TC1` |
| `faults.temporal_consistency.lag_probabilities` | list[float] | `[0.5, 0.3, 0.2]` | non-negative weights | Sampling weights for `lag_choices`; normalized internally | `_inject_temporal_faults()` | Mainly `TC1` |
| `faults.temporal_consistency.duplicated_timestamp_fraction` | float | `0.0` | practical range `[0, 1]` | Fraction of shared rows duplicated and appended to the table | `inject()` shared event path | Mainly `TC3`, secondary `C3` |
| `faults.temporal_consistency.calendar_hole_fraction` | float | `0.03` | practical range `[0, 1]` | Fraction of available weekday-hour cells removed inside one local time window for one client | `_inject_temporal_faults()` | Mainly `TC3` |
| `faults.temporal_consistency.calendar_hole_min_length` | int | `72` | positive integer | Minimum localized calendar-hole window length in rows | `_inject_temporal_faults()` | Mainly `TC3` |
| `faults.temporal_consistency.calendar_hole_max_length` | int | `336` | positive integer `>= min_length` | Maximum localized calendar-hole window length in rows | `_inject_temporal_faults()` | Mainly `TC3` |
| `faults.temporal_consistency.phase_shift_client_fraction` | float | `0.15` | practical range `[0, 1]` | Probability-like trigger for a second-half phase shift on one client | `_inject_temporal_faults()` | Mainly `TC2` |
| `faults.temporal_consistency.phase_shift_steps_choices` | list[int] | `[6, 12]` | positive integer row shifts | Candidate circular shifts applied inside the affected second-half block | `_inject_temporal_faults()` | Mainly `TC2` |
| `faults.temporal_consistency.phase_shift_start_fraction` | float | `0.5` | practical range `[0, 1]` | Fraction of the window after which phase shifting starts | `_inject_temporal_faults()` | Mainly `TC2` |
| `faults.temporal_consistency.mid_window_regime_shift_fraction` | float | `0.2` | practical range `[0, 1]` | Probability-like trigger for a second-half level shift | `_inject_temporal_faults()` | Mainly `TC4`, secondary skewness |
| `faults.temporal_consistency.mid_window_regime_shift_strength` | float | `0.25` | `>= 0` | Shift magnitude in units of client scale | `_inject_temporal_faults()` | Mainly `TC4`, secondary skewness |
| `faults.temporal_consistency.mid_window_regime_start_fraction` | float | `0.5` | practical range `[0, 1]` | Fraction of the window after which the regime shift starts | `_inject_temporal_faults()` | Mainly `TC4` |

##### Temporal-consistency fault keys

This group simulates stale replay, rhythm disruption, missing calendar cells, and non-stationary within-window behavior.

Keys in this group:

- `stale_blocks_per_client`
- `stale_min_length`
- `stale_max_length`
- `lag_choices`
- `lag_probabilities`
- `duplicated_timestamp_fraction`
- `calendar_hole_fraction`
- `calendar_hole_min_length`
- `calendar_hole_max_length`
- `phase_shift_client_fraction`
- `phase_shift_steps_choices`
- `phase_shift_start_fraction`
- `mid_window_regime_shift_fraction`
- `mid_window_regime_shift_strength`
- `mid_window_regime_start_fraction`

Main expected subchecks to decrease:

- `TC1` for stale replay
- `TC2` for phase-shifted or rhythm-disrupted behavior
- `TC3` for calendar holes or duplicate shared rows
- `TC4` for mid-window regime shifts

Practical recommendations:

- `stale_blocks_per_client`
  - `1` is usually enough
- `stale_min_length` / `stale_max_length`
  - `3` to `8` means stale replay for several hours
- `lag_choices`
  - `1`: previous hour copy
  - `24`: previous day copy
  - `168`: previous week copy
- `calendar_hole_fraction`
  - `0.01`: light localized pattern loss
  - `0.05`: moderate
  - `0.10`: severe within the selected local window
- `calendar_hole_min_length` / `calendar_hole_max_length`
  - `72` to `336` rows is a good default for hourly data
  - this keeps the missing weekday-hour cells localized rather than spread across the full year
- `phase_shift_client_fraction`
  - `0.10`: light
  - `0.25`: moderate
  - `0.50`: strong
- `phase_shift_steps_choices`
  - `6`: quarter-day shift
  - `12`: half-day shift
  - `24`: full-day shift
- `mid_window_regime_shift_strength`
  - `0.10`: light
  - `0.25`: moderate
  - `0.50`: strong
- `mid_window_regime_start_fraction`
  - `0.5` means the regime shift starts halfway through the window

Light / moderate / severe examples:

- light:
  - `stale_blocks_per_client: 1`
  - `stale_min_length: 3`
  - `stale_max_length: 4`
  - `calendar_hole_fraction: 0.01`
  - `phase_shift_client_fraction: 0.10`
  - `mid_window_regime_shift_strength: 0.10`
- moderate:
  - `stale_blocks_per_client: 1`
  - `stale_min_length: 3`
  - `stale_max_length: 8`
  - `lag_choices: [1, 2, 24]`
  - `calendar_hole_fraction: 0.03`
  - `phase_shift_client_fraction: 0.25`
  - `mid_window_regime_shift_strength: 0.25`
- severe:
  - `stale_blocks_per_client: 2`
  - `stale_min_length: 6`
  - `stale_max_length: 24`
  - `lag_choices: [24, 168]`
  - `calendar_hole_fraction: 0.10`
  - `phase_shift_client_fraction: 0.50`
  - `phase_shift_steps_choices: [12, 24]`
  - `mid_window_regime_shift_strength: 0.50`

#### Detailed skewness-fault keys

| YAML key | Type | Default | Valid values | Meaning | Used by | Expected DQ effect |
|---|---|---:|---|---|---|---|
| `faults.skewness.mean_shift_client_fraction` | float | `0.25` | practical range `[0, 1]` | Probability-like trigger for shifting a local client block by a constant offset | `_inject_skewness_faults()` | Mainly `SK1`, `SK3` |
| `faults.skewness.mean_shift_std_multiplier` | float | `1.0` | `>= 0` | Constant shift in units of client scale | `_inject_skewness_faults()` | Mainly `SK1`, `SK3` |
| `faults.skewness.mean_shift_min_length` | int | `72` | positive integer | Minimum local mean-shift block length in rows | `_inject_skewness_faults()` | Mainly `SK1`, `SK3` |
| `faults.skewness.mean_shift_max_length` | int | `336` | positive integer `>= min_length` | Maximum local mean-shift block length in rows | `_inject_skewness_faults()` | Mainly `SK1`, `SK3` |
| `faults.skewness.variance_shift_client_fraction` | float | `0.25` | practical range `[0, 1]` | Probability-like trigger for scaling variance inside a temporary regime block | `_inject_skewness_faults()` | Mainly `SK1`, `SK3` |
| `faults.skewness.variance_scale_range` | list[float, float] | `[1.5, 2.0]` | positive low/high | Uniform range for variance scaling factor | `_inject_skewness_faults()` | Mainly `SK1`, `SK3` |
| `faults.skewness.variance_shift_min_length` | int | `72` | positive integer | Minimum local variance-shift block length in rows | `_inject_skewness_faults()` | Mainly `SK1`, `SK3` |
| `faults.skewness.variance_shift_max_length` | int | `336` | positive integer `>= min_length` | Maximum local variance-shift block length in rows | `_inject_skewness_faults()` | Mainly `SK1`, `SK3` |
| `faults.skewness.peak_shift_client_fraction` | float | `0.25` | practical range `[0, 1]` | Probability-like trigger for scaling peak-hour readings inside a local window | `_inject_skewness_faults()` | Mainly `SK2` |
| `faults.skewness.peak_shift_multiplier` | float | `1.5` | `> 0` | Multiplier applied to selected peak-hour readings | `_inject_skewness_faults()` | Mainly `SK2`, secondary `SK1` |
| `faults.skewness.peak_shift_min_length` | int | `72` | positive integer | Minimum local peak-shift window length in rows | `_inject_skewness_faults()` | Mainly `SK2` |
| `faults.skewness.peak_shift_max_length` | int | `336` | positive integer `>= min_length` | Maximum local peak-shift window length in rows | `_inject_skewness_faults()` | Mainly `SK2` |
| `faults.skewness.peak_hours` | list[int] | `[17, 18, 19, 20, 21]` | integers in `[0, 23]` | Hours treated as peak hours | `_inject_skewness_faults()` | Mainly `SK2` |
| `faults.skewness.long_term_shift_client_fraction` | float | `0.25` | practical range `[0, 1]` | Probability-like trigger for a longer partial-month behavioral drift block | `_inject_skewness_faults()` | Mainly `SK1`, `SK3` |
| `faults.skewness.long_term_shift_multiplier` | float | `0.5` | `>= 0` | Additive drift in units of client scale | `_inject_skewness_faults()` | Mainly `SK1`, `SK3` |
| `faults.skewness.long_term_shift_min_length` | int | `336` | positive integer | Minimum local long-term drift block length in rows | `_inject_skewness_faults()` | Mainly `SK1`, `SK3` |
| `faults.skewness.long_term_shift_max_length` | int | `720` | positive integer `>= min_length` | Maximum local long-term drift block length in rows | `_inject_skewness_faults()` | Mainly `SK1`, `SK3` |

##### Skewness fault keys

This group simulates distribution drift relative to each client's own historical baseline.

Keys in this group:

- `mean_shift_client_fraction`
- `mean_shift_std_multiplier`
- `mean_shift_min_length`
- `mean_shift_max_length`
- `variance_shift_client_fraction`
- `variance_scale_range`
- `variance_shift_min_length`
- `variance_shift_max_length`
- `peak_shift_client_fraction`
- `peak_shift_multiplier`
- `peak_shift_min_length`
- `peak_shift_max_length`
- `peak_hours`
- `long_term_shift_client_fraction`
- `long_term_shift_multiplier`
- `long_term_shift_min_length`
- `long_term_shift_max_length`

Main expected subchecks to decrease:

- `SK1` for histogram/distribution divergence
- `SK2` for peak/off-peak ratio changes
- `SK3` for mean/std deviation and longer-term drift

Practical recommendations:

- `mean_shift_client_fraction`
  - probability or fraction of selected clients receiving a temporary local mean shift
- `mean_shift_std_multiplier`
  - `0.5`: light
  - `1.0`: moderate
  - `2.0`: strong
- `mean_shift_min_length` / `mean_shift_max_length`
  - `72` to `336` rows is a good hourly default for 3 to 14 day drift blocks
- `variance_shift_client_fraction`
  - probability or fraction of selected clients receiving variance scaling
- `variance_scale_range`
  - `[1.2, 1.5]`: light
  - `[1.5, 2.0]`: moderate
  - `[2.0, 3.0]`: strong
- `variance_shift_min_length` / `variance_shift_max_length`
  - `72` to `336` rows gives a temporary high-variance regime instead of a full-series reshape
- `peak_shift_client_fraction`
  - probability or fraction of selected clients receiving peak amplification
- `peak_shift_multiplier`
  - `1.2`: light
  - `1.5`: moderate
  - `2.0+`: strong
- `peak_shift_min_length` / `peak_shift_max_length`
  - `72` to `336` rows keeps peak amplification localized to a realistic block of days
- `peak_hours`
  - should match the expected household evening peak hours
- `long_term_shift_client_fraction`
  - probability or fraction of clients receiving partial-month behavioral drift
- `long_term_shift_multiplier`
  - `0.25`: light
  - `0.5`: moderate
  - `1.0`: strong
- `long_term_shift_min_length` / `long_term_shift_max_length`
  - `336` to `720` rows is a good hourly default for 2 to 4 week drift blocks

Light / moderate / severe examples:

- light:
  - `mean_shift_std_multiplier: 0.5`
  - `variance_scale_range: [1.2, 1.5]`
  - `peak_shift_multiplier: 1.2`
  - `long_term_shift_multiplier: 0.25`
- moderate:
  - `mean_shift_std_multiplier: 1.0`
  - `variance_scale_range: [1.5, 2.0]`
  - `peak_shift_multiplier: 1.5`
  - `long_term_shift_multiplier: 0.5`
- severe:
  - `mean_shift_std_multiplier: 2.0`
  - `variance_scale_range: [2.0, 3.0]`
  - `peak_shift_multiplier: 2.0`
  - `long_term_shift_multiplier: 1.0`

### 6.3 Fault Injection Methods

All event rows are written through `_make_event(...)` in [src/demand_quality/fault_injection.py](/Users/meilyoeng/Documents/MAID/summer sem/Thesis Project/dqops_5g-main/src/demand_quality/fault_injection.py:611).

Common metadata fields written for every event:

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

For shared row events:

- `client_id = "__all_clients__"`
- `shared_timestamp_event = True`

The generated report also includes final net summary rows:

- `fault_type = "final_net_corruption"`
- one summary row per selected client
- one overall summary row with `client_id = "__all_selected_clients__"`
- `affected_rows` and `affected_fraction` describe the final net number and percentage of original rows that remain corrupted after all injected events and overlaps are resolved

#### Shared wide-table methods

`dropped_timestamp_rows`

- Code path:
  - `WideDemandFaultInjector.inject(...)`
- Description:
  - chooses shared row indices and removes those rows from the final wide table
- Input assumptions:
  - one shared timestamp column for the whole table
- YAML:
  - `faults.completeness.dropped_timestamp_rows_fraction`
- Transformation:
  - `drop_count = max(1, round(n_rows * fraction))`
  - the selected row indices are removed after client-level faults have run
- Example:
  - before timestamps: `[00:00, 01:00, 02:00, 03:00]`
  - after timestamps: `[00:00, 01:00, 03:00]`
- Main DQ effect:
  - `completeness / C3`
- Secondary effects:
  - can also reduce `TC3` because some calendar cells disappear
- Metadata:
  - `parameters_used` contains `{"fraction": ...}`

`duplicated_timestamps`

- Code path:
  - `WideDemandFaultInjector.inject(...)`
- Description:
  - duplicates selected shared rows, appends them, then resort by timestamp
- YAML:
  - `faults.temporal_consistency.duplicated_timestamp_fraction`
- Transformation:
  - `dup_count = max(1, round(n_rows * fraction))`
  - selected rows are copied, appended, sorted by timestamp, and index-reset
- Example:
  - before timestamps: `[00:00, 01:00, 02:00]`
  - after timestamps: `[00:00, 01:00, 01:00, 02:00]`
- Main DQ effect:
  - `temporal_consistency / TC3`
- Secondary effects:
  - can also hurt completeness continuity because the grid is no longer strictly regular
- Metadata:
  - `parameters_used` contains `{"fraction": ...}`

Implementation note:

- there is no standalone explicit `irregular_interval` fault type; irregular intervals emerge from shared row deletion or duplication.

#### Detailed completeness methods

`client_dropout`

- Code path:
  - `_inject_completeness_faults(...)`
- Description:
  - turns every finite value of the selected client into `NaN`
- YAML:
  - `faults.completeness.client_dropout_fraction`
- Transformation:
  - if a Bernoulli trigger based on `client_dropout_fraction * client_factor` fires, all finite values become `NaN`
  - the completeness helper then returns immediately, so `random_missing` and `block_missing` are skipped for that client
- Example:
  - before: `[10, 11, 12, 13]`
  - after: `[NaN, NaN, NaN, NaN]`
- Main DQ effect:
  - `completeness / C2`
- Secondary effects:
  - also collapses `C1`, staleness, calendar coverage, and local skewness
- Metadata:
  - `parameters_used = {"fraction": ...}`

`random_missing`

- Code path:
  - `_inject_completeness_faults(...)`
- Description:
  - sparse missing values on valid positions only
- YAML:
  - `faults.completeness.random_missing_rate`
- Transformation:
  - sample valid positions
  - set those cells to `NaN`
- Example:
  - before: `[10, 11, 12, 13, 14]`
  - after: `[10, NaN, 12, 13, NaN]`
- Main DQ effect:
  - `completeness / C1`
- Secondary effects:
  - if many adjacent values are removed, continuity-like checks can also drop
- Metadata:
  - `parameters_used = {"rate": ...}`

`block_missing`

- Code path:
  - `_inject_completeness_faults(...)`
- Description:
  - one or more contiguous missing intervals
- YAML:
  - `faults.completeness.block_missing.events_per_client`
  - `faults.completeness.block_missing.min_length`
  - `faults.completeness.block_missing.max_length`
- Transformation:
  - sample `round(events_per_client * client_factor)` blocks
  - each block length is a discrete uniform sample between `min_length` and `max_length`
  - set the full block to `NaN`
- Example:
  - before: `[10, 11, 12, 13, 14, 15]`
  - after: `[10, 11, NaN, NaN, NaN, 15]`
- Main DQ effect:
  - `completeness / C1`
- Secondary effects:
  - often also hurts interval continuity and calendar coverage
- Metadata:
  - `parameters_used` stores the block config dictionary

Completeness methods listed in the target taxonomy but not implemented as separate fault types:

- explicit `empty_window`:
  - implemented by `client_dropout`
- explicit `timestamp_gap`:
  - implemented by `dropped_timestamp_rows`
- explicit `irregular_interval`:
  - not implemented as its own named fault type

#### Detailed accuracy methods

`negative_values`

- Code path:
  - `_inject_accuracy_faults(...)`
- YAML:
  - `faults.accuracy.negative_rate`
- Transformation:
  - each selected value becomes `-max(abs(value), 1.0)`
- Example:
  - before: `[8, 9, 10]`
  - after: `[8, -9, 10]`
- Main DQ effect:
  - `accuracy / A1`
- Secondary effects:
  - also lowers `A3_nonneg`
- Metadata:
  - `parameters_used = {"rate": ...}`

`impossible_high`

- Code path:
  - `_inject_accuracy_faults(...)`
- YAML:
  - `faults.accuracy.impossible_high_rate`
  - `faults.accuracy.high_multiplier`
- Transformation:
  - uses `upper_anchor = max(q99(abs(values)), scale, 1.0)`
  - sets selected values to `upper_anchor * high_multiplier`
- Example:
  - if `upper_anchor = 25` and `high_multiplier = 3`, the injected value is `75`
- Main DQ effect:
  - `accuracy / A1`
- Secondary effects:
  - may also shift skewness
- Metadata:
  - `parameters_used = {"rate": ..., "multiplier": ...}`

`spikes`

- Code path:
  - `_inject_accuracy_faults(...)`
- YAML:
  - `faults.accuracy.spike_rate`
  - `faults.accuracy.spike_multiplier`
- Transformation:
  - selected values become `current + spike_multiplier * scale`
- Example:
  - before: `[10, 11, 10, 12]`
  - after: `[10, 11, 26, 12]`
- Main DQ effect:
  - `accuracy / A2`
- Secondary effects:
  - can also affect `A3_roc` and skewness
- Metadata:
  - `parameters_used = {"rate": ..., "multiplier": ...}`

`noisy_values`

- Code path:
  - `_inject_accuracy_faults(...)`
- YAML:
  - `faults.accuracy.noise_rate`
  - `faults.accuracy.noise_std_fraction`
- Transformation:
  - add Gaussian noise with `std = scale * noise_std_fraction`
- Example:
  - before: `[10.0, 10.0, 10.0]`
  - after: `[10.0, 10.4, 9.7]`
- Main DQ effect:
  - `accuracy / A2`
- Secondary effects:
  - repeated large noise can also change skewness
- Metadata:
  - `parameters_used = {"rate": ..., "noise_std_fraction": ...}`

`rate_jumps`

- Code path:
  - `_inject_accuracy_faults(...)`
- YAML:
  - `faults.accuracy.jump_rate`
  - `faults.accuracy.jump_multiplier`
- Transformation:
  - for positions `t > 0`, set:
  - `x_t = max(0, x_{t-1} +/- jump_multiplier * scale)`
- Example:
  - before: `[10, 11, 12, 13]`
  - after: `[10, 11, 40, 13]`
- Main DQ effect:
  - `accuracy / A3`
- Secondary effects:
  - may also register as a local anomaly in `A2`
- Metadata:
  - `parameters_used = {"rate": ..., "multiplier": ...}`

`stuck_constant`

- Code path:
  - `_inject_accuracy_faults(...)`
- YAML:
  - `faults.accuracy.stuck_segments_per_client`
  - `faults.accuracy.stuck_min_length`
  - `faults.accuracy.stuck_max_length`
- Transformation:
  - choose one or more blocks and copy the block's first value across the whole block
- Example:
  - before: `[12, 13, 14, 15, 16]`
  - after: `[12, 13, 13, 13, 13]`
- Main DQ effect:
  - `accuracy / A3`
- Secondary effects:
  - may also alter autocorrelation and local distribution
- Metadata:
  - `parameters_used` stores stuck block length settings

Accuracy methods listed in the target taxonomy but not implemented as separate names:

- explicit `local_anomaly` fault type:
  - not implemented as its own named event
  - approximated by `spikes`, `noisy_values`, and `rate_jumps`

#### Detailed temporal-consistency methods

`stale_block`

- Code path:
  - `_inject_temporal_faults(...)`
- YAML:
  - `faults.temporal_consistency.stale_blocks_per_client`
  - `faults.temporal_consistency.stale_min_length`
  - `faults.temporal_consistency.stale_max_length`
  - `faults.temporal_consistency.lag_choices`
  - `faults.temporal_consistency.lag_probabilities`
- Transformation:
  - choose a block
  - sample a lag
  - copy `x_{t-lag}` into `x_t` across the block
- Example:
  - before: `[10, 11, 12, 13, 14]`
  - after with lag `1`: `[10, 11, 11, 12, 13]`
- Main DQ effect:
  - `temporal_consistency / TC1`
- Secondary effects:
  - can also flatten local variance and affect skewness
- Metadata:
  - `parameters_used = {"lag": ...}`

`calendar_hole`

- Code path:
  - `_inject_temporal_faults(...)`
- YAML:
  - `faults.temporal_consistency.calendar_hole_fraction`
  - `faults.temporal_consistency.calendar_hole_min_length`
  - `faults.temporal_consistency.calendar_hole_max_length`
- Transformation:
  - sample one localized time block
  - find the weekday-hour cells available inside that block
  - remove a fraction of those cells only inside that block for the selected client
- Example:
  - if a 7-day window is selected and `(Sunday, 18:00)` is chosen inside it, only that local Sunday 18:00 pattern is removed
- Main DQ effect:
  - `temporal_consistency / TC3`
- Secondary effects:
  - also lowers completeness through missingness
- Metadata:
  - `parameters_used = {"fraction": ...}`

`phase_shift`

- Code path:
  - `_inject_temporal_faults(...)`
- YAML:
  - `faults.temporal_consistency.phase_shift_client_fraction`
  - `faults.temporal_consistency.phase_shift_steps_choices`
  - `faults.temporal_consistency.phase_shift_start_fraction`
- Transformation:
  - operate on the second-half block starting at `round(n_rows * start_fraction)`
  - circularly roll the finite values in that block by `shift_steps`
- Example:
  - if a late-afternoon peak is shifted by `12` hourly steps, it moves into the opposite half-day
- Main DQ effect:
  - `temporal_consistency / TC2`
- Secondary effects:
  - can also change `SK2` if peak periods move
- Metadata:
  - `parameters_used = {"shift_steps": ..., "start_fraction": ...}`

`mid_window_regime_shift`

- Code path:
  - `_inject_temporal_faults(...)`
- YAML:
  - `faults.temporal_consistency.mid_window_regime_shift_fraction`
  - `faults.temporal_consistency.mid_window_regime_shift_strength`
  - `faults.temporal_consistency.mid_window_regime_start_fraction`
- Transformation:
  - operate on the second-half block starting at `round(n_rows * start_fraction)`
  - compute local scale on that block
  - add `shift = strength * max(scale, 1.0)` to all finite values in that block
- Example:
  - before second half: `[14, 15, 16, 17]`
  - after: `[18, 19, 20, 21]`
- Main DQ effect:
  - `temporal_consistency / TC4`
- Secondary effects:
  - also changes skewness and potentially range checks
- Metadata:
  - `parameters_used = {"shift": ..., "start_fraction": ...}`

Temporal methods listed in the target taxonomy but not implemented as separate names:

- explicit `disrupted_autocorrelation`:
  - not implemented as its own named event
  - mainly approximated by `phase_shift`
- explicit `seasonal_rhythm_break`:
  - not implemented as its own named event
  - approximated by `phase_shift` and `stale_block`
- explicit `temporary_level_shift`:
  - implemented by `mid_window_regime_shift`

#### Detailed skewness methods

`mean_shift`

- Code path:
  - `_inject_skewness_faults(...)`
- YAML:
  - `faults.skewness.mean_shift_client_fraction`
  - `faults.skewness.mean_shift_std_multiplier`
  - `faults.skewness.mean_shift_min_length`
  - `faults.skewness.mean_shift_max_length`
- Transformation:
  - sample one local block
  - add a constant offset `shift = multiplier * scale` only inside that block
- Example:
  - before: `[10, 11, 12, 13, 14, 15]`
  - after with shift `5` in the middle block: `[10, 11, 17, 18, 14, 15]`
- Main DQ effect:
  - `skewness / SK1`
- Secondary effects:
  - also lowers `SK3`
- Metadata:
  - `parameters_used = {"shift": ...}`

`variance_shift`

- Code path:
  - `_inject_skewness_faults(...)`
- YAML:
  - `faults.skewness.variance_shift_client_fraction`
  - `faults.skewness.variance_scale_range`
  - `faults.skewness.variance_shift_min_length`
  - `faults.skewness.variance_shift_max_length`
- Transformation:
  - sample one temporary regime block
  - scale deviations around the block mean:
  - `x' = mean_block + (x - mean_block) * factor`
- Example:
  - before around mean `10`: `[8, 10, 12]`
  - after with factor `2`: `[6, 10, 14]`
- Main DQ effect:
  - `skewness / SK3`
- Secondary effects:
  - also changes `SK1`
- Metadata:
  - `parameters_used = {"factor": ...}`

`peak_hour_shift`

- Code path:
  - `_inject_skewness_faults(...)`
- YAML:
  - `faults.skewness.peak_shift_client_fraction`
  - `faults.skewness.peak_shift_multiplier`
  - `faults.skewness.peak_shift_min_length`
  - `faults.skewness.peak_shift_max_length`
  - `faults.skewness.peak_hours`
- Transformation:
  - sample one local block
  - within that block, multiply only the selected peak-hour values by `peak_shift_multiplier`
- Example:
  - peak hours before: `[20, 22, 21]`
  - after with multiplier `1.5`: `[30, 33, 31.5]`
- Main DQ effect:
  - `skewness / SK2`
- Secondary effects:
  - also changes histogram divergence `SK1`
- Metadata:
  - `parameters_used = {"multiplier": ..., "peak_hours": [...]}`

`long_term_shift`

- Code path:
  - `_inject_skewness_faults(...)`
- YAML:
  - `faults.skewness.long_term_shift_client_fraction`
  - `faults.skewness.long_term_shift_multiplier`
  - `faults.skewness.long_term_shift_min_length`
  - `faults.skewness.long_term_shift_max_length`
- Transformation:
  - sample one longer local block
  - add a constant drift only inside that block
- Example:
  - before local block: `[12, 13, 14]`
  - after: `[16, 17, 18]`
- Main DQ effect:
  - `skewness / SK3`
- Secondary effects:
  - also changes `SK1` and may look like a temporal regime change
- Metadata:
  - `parameters_used = {"shift": ..., "start_fraction": 0.5}`

Skewness methods listed in the target taxonomy but not implemented as separate names:

- explicit `distribution_shift`:
  - not implemented as a standalone histogram-targeting event
  - approximated by `mean_shift`, `variance_shift`, `peak_hour_shift`, and `long_term_shift`
- explicit `EV-like peak increase`:
  - not implemented as a separate named event
  - approximated by `peak_hour_shift`
- explicit `behavioural_baseline_change`:
  - approximated by `long_term_shift`

#### Simple selected-client mode methods

Simple mode uses `_inject_simple_fraction_mode(...)`. The YAML fields are top-level, not nested under `faults:`.

`missing`

- Transformation:
  - set the sampled point to `NaN`
- YAML:
  - `fault_fraction`
  - `simple_fault_types`
  - `simple_fault_probabilities`
- Main DQ effect:
  - `completeness / C1`

`out_of_range`

- Transformation:
  - `x' = max(abs(x), 1) * Uniform(low, high)`
- YAML:
  - `simple_out_of_range_range`
- Main DQ effect:
  - `accuracy / A1`

`inconsistent`

- Transformation:
  - `x' = -max(abs(x), 1) * Uniform(low, high)`
- YAML:
  - `simple_inconsistent_range`
- Main DQ effect:
  - `accuracy / A1`
- Secondary effects:
  - also hurts `A3_nonneg`

`timeliness`

- Transformation:
  - copy a neighboring value from `t + step` where `step` is sampled from `simple_timeliness_steps`
  - if the copied value matches the original, the code tries left/right neighbors and finally a negative fallback
- YAML:
  - `simple_timeliness_steps`
- Main DQ effect:
  - `temporal_consistency / TC1`

`noise`

- Transformation:
  - `x' = x + Normal(0, scale * simple_noise_std_fraction)`
- YAML:
  - `simple_noise_std_fraction`
- Main DQ effect:
  - `accuracy / A2`
- Status:
  - implemented in code, but not enabled in the default `simple_fault_types` list

`spike`

- Transformation:
  - `x' = x + simple_spike_multiplier * scale`
- YAML:
  - `simple_spike_multiplier`
- Main DQ effect:
  - `accuracy / A2`
- Status:
  - implemented in code, but not enabled in the default `simple_fault_types` list

### 6.4 Fault-to-Dimension Mapping Table

| Fault type | Main target dimension | Main subcheck | Secondary effects | Explanation |
|---|---|---|---|---|
| `client_dropout` | Completeness | `C2` | `C1`, `TC1`, `TC3`, skewness | Entire client window is wiped to `NaN` |
| `random_missing` | Completeness | `C1` | `C3`, `TC3` | Sparse missing values lower non-null ratio |
| `block_missing` | Completeness | `C1` | `C3`, `TC3` | Missing intervals reduce both coverage and continuity |
| `dropped_timestamp_rows` | Completeness | `C3` | `TC3` | Shared rows are removed from the timestamp grid |
| `duplicated_timestamps` | Temporal Consistency | `TC3` | `C3` | Shared rows are duplicated, making the grid irregular |
| `negative_values` | Accuracy | `A1` | `A3_nonneg` | Negative demand violates range and sign rules |
| `impossible_high` | Accuracy | `A1` | `SK1`, `SK3` | Values exceed plausible range |
| `spikes` | Accuracy | `A2` | `A3_roc`, skewness | Large point anomalies break local smoothness |
| `noisy_values` | Accuracy | `A2` | skewness | Random perturbations increase anomaly rate |
| `rate_jumps` | Accuracy | `A3` | `A2` | Abrupt changes violate rate-of-change rules |
| `stuck_constant` | Accuracy | `A3` | `TC2`, skewness | Frozen runs violate dynamic behavior |
| `stale_block` | Temporal Consistency | `TC1` | skewness | Older values are copied forward |
| `calendar_hole` | Temporal Consistency | `TC3` | `C1` | Specific weekday-hour cells disappear |
| `phase_shift` | Temporal Consistency | `TC2` | `SK2` | Rhythms are shifted without removing timestamps |
| `mid_window_regime_shift` | Temporal Consistency | `TC4` | skewness, `A1` | Second-half level shift breaks stationarity |
| `mean_shift` | Skewness | `SK1` | `SK3` | Whole-series offset changes local distribution |
| `variance_shift` | Skewness | `SK3` | `SK1` | Spread increases or decreases around the mean |
| `peak_hour_shift` | Skewness | `SK2` | `SK1` | Peak/off-peak ratio changes |
| `long_term_shift` | Skewness | `SK3` | `SK1`, `TC4` | Second-half drift changes moments and behavior |
| `missing` (simple mode) | Completeness | `C1` | `C3` | Simple-mode sparse missing value |
| `out_of_range` (simple mode) | Accuracy | `A1` | skewness | Simple-mode positive range violation |
| `inconsistent` (simple mode) | Accuracy | `A1` | `A3_nonneg` | Simple-mode negative inconsistent value |
| `timeliness` (simple mode) | Temporal Consistency | `TC1` | skewness | Simple-mode neighbor copy |
| `noise` (simple mode) | Accuracy | `A2` | skewness | Implemented but off by default |
| `spike` (simple mode) | Accuracy | `A2` | `A3_roc` | Implemented but off by default |

### 6.5 YAML Example Walkthrough

The example below is a valid detailed-mode configuration. It uses the actual keys supported by the current injector.

```yaml
fault_fraction:
global_fault_rate: 1.0
per_client_fault_rate:
  client_b: 0.5

clients:
  selected: [client_a, client_b]
  random_fraction: 1.0
  random_count:

faults:
  completeness:
    client_dropout_fraction: 0.0
    random_missing_rate: 0.02
    dropped_timestamp_rows_fraction: 0.01
    block_missing:
      events_per_client: 1
      min_length: 4
      max_length: 8

  accuracy:
    negative_rate: 0.01
    impossible_high_rate: 0.005
    high_multiplier: 3.0
    spike_rate: 0.01
    spike_multiplier: 4.0
    noise_rate: 0.01
    noise_std_fraction: 0.05
    jump_rate: 0.01
    jump_multiplier: 6.0
    stuck_segments_per_client: 1
    stuck_min_length: 4
    stuck_max_length: 12

  temporal_consistency:
    stale_blocks_per_client: 1
    stale_min_length: 3
    stale_max_length: 6
    lag_choices: [1, 24]
    lag_probabilities: [0.7, 0.3]
    duplicated_timestamp_fraction: 0.0
    calendar_hole_fraction: 0.02
    phase_shift_client_fraction: 0.2
    phase_shift_steps_choices: [6, 12]
    phase_shift_start_fraction: 0.5
    mid_window_regime_shift_fraction: 0.2
    mid_window_regime_shift_strength: 0.25
    mid_window_regime_start_fraction: 0.5

  skewness:
    mean_shift_client_fraction: 0.25
    mean_shift_std_multiplier: 1.0
    variance_shift_client_fraction: 0.25
    variance_scale_range: [1.5, 2.0]
    peak_shift_client_fraction: 0.25
    peak_shift_multiplier: 1.5
    peak_hours: [17, 18, 19, 20, 21]
    long_term_shift_client_fraction: 0.25
    long_term_shift_multiplier: 0.5
```

Line-by-line interpretation:

- `fault_fraction:`
  - left empty, so the injector stays in detailed mode
- `global_fault_rate: 1.0`
  - do not globally scale the detailed rates up or down
- `per_client_fault_rate:`
  - start a per-client scaling map
- `client_b: 0.5`
  - halve the detailed fault intensity for `client_b`
- `clients:`
  - start the client selection block
- `selected: [client_a, client_b]`
  - only these clients are candidates for client-level faults
- `random_fraction: 1.0`
  - keep all candidates instead of subsampling them
- `random_count:`
  - left empty, so no exact-count override is applied
- `faults:`
  - start the detailed fault configuration namespace
- `completeness:`
  - completeness-oriented faults
- `client_dropout_fraction: 0.0`
  - disable full-column client dropout
- `random_missing_rate: 0.02`
  - about 2% sparse missing values per selected client
- `dropped_timestamp_rows_fraction: 0.01`
  - remove about 1% of shared rows for all clients
- `block_missing.events_per_client: 1`
  - one block outage per selected client before scaling
- `block_missing.min_length: 4`
  - outage lasts at least 4 rows
- `block_missing.max_length: 8`
  - outage lasts at most 8 rows
- `accuracy:`
  - accuracy-oriented faults
- `negative_rate: 0.01`
  - about 1% of valid points become negative
- `impossible_high_rate: 0.005`
  - about 0.5% of valid points become too large
- `high_multiplier: 3.0`
  - impossible high values are roughly 3x the anchor
- `spike_rate: 0.01`
  - about 1% of points receive large spikes
- `spike_multiplier: 4.0`
  - spike size is 4x the local client scale
- `noise_rate: 0.01`
  - about 1% of points receive Gaussian noise
- `noise_std_fraction: 0.05`
  - noise std is 5% of local client scale
- `jump_rate: 0.01`
  - about 1% of points are replaced with abrupt jumps
- `jump_multiplier: 6.0`
  - jump size is 6x the local client scale
- `stuck_segments_per_client: 1`
  - inject one frozen segment per selected client before scaling
- `stuck_min_length: 4`
  - frozen segment lasts at least 4 rows
- `stuck_max_length: 12`
  - frozen segment lasts at most 12 rows
- `temporal_consistency:`
  - temporal faults
- `stale_blocks_per_client: 1`
  - inject one stale block per selected client before scaling
- `stale_min_length: 3`
  - stale block length lower bound
- `stale_max_length: 6`
  - stale block length upper bound
- `lag_choices: [1, 24]`
  - stale blocks copy from 1 row or 24 rows back
- `lag_probabilities: [0.7, 0.3]`
  - 1-row stale copies are more likely than 24-row copies
- `duplicated_timestamp_fraction: 0.0`
  - do not duplicate shared rows in this example
- `calendar_hole_fraction: 0.02`
  - remove about 2% of the available calendar cells inside one localized window per selected client
- `phase_shift_client_fraction: 0.2`
  - about 20% of selected clients get a phase shift
- `phase_shift_steps_choices: [6, 12]`
  - the second half of the series is circularly shifted by 6 or 12 rows
- `phase_shift_start_fraction: 0.5`
  - only the second half of the series is phase shifted
- `mid_window_regime_shift_fraction: 0.2`
  - about 20% of selected clients get a second-half level shift
- `mid_window_regime_shift_strength: 0.25`
  - the level shift magnitude is 25% of local client scale
- `mid_window_regime_start_fraction: 0.5`
  - regime shift starts halfway through the window
- `skewness:`
  - distribution-shift faults
- `mean_shift_client_fraction: 0.25`
  - about 25% of selected clients get a temporary local mean shift
- `mean_shift_std_multiplier: 1.0`
  - the local mean shift is about one client-scale unit
- `variance_shift_client_fraction: 0.25`
  - about 25% of selected clients get temporary variance scaling
- `variance_scale_range: [1.5, 2.0]`
  - scale deviations around the mean by a random factor between 1.5 and 2.0
- `peak_shift_client_fraction: 0.25`
  - about 25% of selected clients get peak-hour amplification
- `peak_shift_multiplier: 1.5`
  - multiply selected peak-hour values by 1.5
- `peak_hours: [17, 18, 19, 20, 21]`
  - treat late afternoon and evening as peak hours
- `long_term_shift_client_fraction: 0.25`
  - about 25% of selected clients get a longer partial-month behavioral drift
- `long_term_shift_multiplier: 0.5`
  - the localized drift equals about half of local client scale

Example of how this YAML interacts with the CLI:

```bash
python scripts/inject_faults.py \
  --input data/demand.csv \
  --output data/demand_faulty.csv \
  --report output/fault_injection_report.csv \
  --timestamp-col timestamp \
  --fault-config configs/fault_injection.yaml \
  --seed 42
```

Important clarification:

- the seed is supplied by `--seed 42`
- it is not read from the YAML file
- the input and output paths are also CLI arguments, not YAML fields

## 7. Rule-Based Scorer

All aligned dimension scores are clipped conceptually to `[0, 1]`, where:

- `1` means better quality
- `0` means worse quality

### Completeness

For client `k` over `N` timestamps:

- `C1_k = non_null_count / N`
- `C2_k = 1` if the client column exists and has at least one valid value, else `0`
- `C3_k = matched_expected_interval_pairs / max(1, N - 1)` operationally implemented using the expected timestamp grid over the observed span

Composite:

`Comp_k = (C1_k + C2_k + C3_k) / 3`

### Accuracy

- `A1`: range validation
  - current values must satisfy `lower_bound <= x_t <= upper_bound`
  - upper bound is either fixed or derived from the client's historical clean reference
- `A2`: local anomaly score
  - rolling median + MAD or seasonal naive deviation check
  - `A2_k = 1 - anomaly_fraction`
- `A3`: context and behaviour rules
  - non-negative demand
  - bounded rate of change
  - no long frozen runs

Composite:

- `A3_k = (A3_nonneg + A3_roc + A3_freeze) / 3`
- `Acc_k = (A1_k + A2_k + A3_k) / 3`

### Temporal Consistency

Aligned implementation is local-baseline only.

- `TC1`: staleness
  - `TC1_k = max(0, 1 - Age_k / tau)`
- `TC2`: autocorrelation integrity
  - compare current autocorrelation against local historical autocorrelation at configured lags
  - default lags: `{1, 24, 168}`
- `TC3`: calendar coverage
  - hour-of-day x day-of-week coverage over 168 cells
- `TC4`: within-window stationarity
  - split the current window into `K` contiguous segments
  - compare the segment-mean variability to the client's own historical stationarity reference

Composite:

`TC_k = (TC1_k + TC2_k + TC3_k + TC4_k) / 4`

### Skewness

Here skewness means distribution deviation, not only third-moment skewness.

- `SK1`: JSD between current window histogram and local historical histogram
- `SK2`: peak-hour ratio shift relative to local historical peak ratio
- `SK3`: mean/std shift relative to local historical baseline

Composite:

`Skew_k = (SK1_k + SK2_k + SK3_k) / 3`

## 8. PCA Quality Label

The PCA stage consumes the four dimension scores:

`[completeness, accuracy, temporal_consistency, skewness]`

Procedure:

1. standardize each dimension by column mean and std,
2. fit one-component PCA,
3. project each client onto PC1,
4. sign-correct PC1 if needed so that better dimension scores correspond to higher quality,
5. min-max normalize the corrected raw scores to get `quality_score`.

Outputs:

- `pca_raw_score`
  - unconstrained scalar
  - not required to be in `[0, 1]`
- `quality_score`
  - always in `[0, 1]`
  - `1` means best relative quality in the scored batch
  - `0` means worst relative quality in the scored batch

Fallback:

- if PCA collapses or all raw scores are equal, the implementation falls back to the row-wise mean of the four dimension scores
- this fallback remains in `[0, 1]`

Interpretation warning:

- `quality_score` is relative within the scored batch
- it is not an absolute guarantee of perfect quality

## 9. XGBoost Machine Learning Scorer

### Why XGBoost is needed

The rule-based scorer + PCA labeler is interpretable, but slower than a lightweight learned predictor. For SplitFed, the long-term architecture is:

1. use the rule-based scorer to generate trusted labels offline,
2. train an XGBoost regressor on engineered window features,
3. use the XGBoost model locally during training to predict quality quickly.

### Current repository status

The demand package does not yet contain:

- a demand-specific feature extractor,
- a demand-specific XGBoost training module,
- a demand-specific Optuna tuning module,
- a demand-specific prediction CLI.

### Reusable existing code

- `src/training/model_trainer.py`
  - uses XGBoost regression
  - target is continuous and compatible with `quality_score`
  - includes Optuna search
  - saves `model.bin`, `model.joblib`, `scaler.joblib`, and `meta.json`
- `src/deployment/dq_predictor.py`
  - loads trained model artifacts
  - clips predictions to `[0, 1]`

### What is not directly reusable

- `src/data_processing/feature_extractor.py`
  - O-RAN cell/UE schema specific
- `src/training/label_generator.py`
  - assumes window folders with `cell_data.parquet` and `ue_data.parquet`
- `scripts/xgboost_confusion_matrix.py`
  - treats the regression output as thresholded classification
  - not the right primary evaluation for demand quality regression

### Recommended future demand model

- estimator: `XGBRegressor`
- target: PCA-derived `quality_score`
- metrics:
  - MAE
  - RMSE
  - R²
  - Spearman correlation
  - ranking consistency
- prediction handling:
  - clip predictions to `[0, 1]`

## 10. How to Run

### Calibrate local baseline

```bash
python scripts/calibrate_demand_baseline.py \
  --input data/demand_clean.csv \
  --output output/demand_baseline.json \
  --timestamp-col timestamp \
  --config configs/data_quality_scoring.yaml
```

### Inject faults

```bash
python scripts/inject_faults.py \
  --input data/demand.csv \
  --output data/demand_faulty.csv \
  --report output/fault_injection_report.csv \
  --timestamp-col timestamp \
  --fault-config configs/fault_injection.yaml \
  --seed 42
```

### Score rule-based quality

```bash
python scripts/score_data_quality.py \
  --input data/demand_faulty.csv \
  --clean-reference data/demand.csv \
  --output output/client_quality_scores.csv \
  --timestamp-col timestamp \
  --config configs/data_quality_scoring.yaml
```

### Compute PCA quality score

```bash
python scripts/compute_pca_quality_score.py \
  --input output/client_quality_scores.csv \
  --output output/client_quality_scores_with_pca.csv \
  --config configs/data_quality_scoring.yaml \
  --model-output output/pca_quality_model.json
```

### Run implemented pipeline end to end

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
  --seed 42 \
  --pca-model-output output/pca_quality_model.json
```

## 11. How to Interpret Outputs

- dimension subchecks:
  - diagnostic signals explaining why quality dropped
- dimension scores:
  - four interpretable quality dimensions in `[0, 1]`
- `pca_raw_score`:
  - raw PC1 projection
  - useful for debugging PCA geometry
- `quality_score`:
  - normalized overall relative quality in `[0, 1]`
- `aggregation_weight`:
  - normalized server-side client weight
  - sums to `1` over the scored client set

Recommended SplitFed use:

- use dimension scores for diagnosis, filtering, and fairness analysis,
- use `quality_score` for scalar quality reporting,
- use `aggregation_weight` if you want quality-aware client aggregation.

## 12. Known Limitations

- the implemented demand package currently ends at PCA quality scoring
- demand-specific feature extraction and XGBoost training are not yet ported into `src/demand_quality/`
- `quality_score` is relative within a scored batch
- the XGBoost model, once added, will approximate the rule-based + PCA pipeline rather than replace its semantics
- the scorer currently derives local baselines from clean-reference demand history or the separate baseline calibration utility; production baseline governance still needs explicit policy
- the explicit calibrated baseline artifact is not yet a required scorer input because the current scorer can still derive local references on the fly from the clean reference CSV

## 13. Recommended Next Steps

1. Add a demand-specific feature extractor under `src/demand_quality/features.py`.
2. Build a labeled dataset of window features -> PCA `quality_score`.
3. Port XGBoost training into `src/demand_quality/train_xgboost.py`.
4. Port or wrap Optuna search into a demand-specific tuning module.
5. Add a runtime `predict_demand_quality.py` CLI and a small model wrapper for client-local prediction.
6. Integrate the frozen XGBoost scorer into the SplitFed client loop.
7. Keep the scorer fixed during one SplitFed experiment; recalibrate only between experiments or after major regime shifts.
8. Run ablations:
   - no DQ
   - rule-based DQ only
   - PCA scalar DQ
   - XGBoost-predicted DQ
