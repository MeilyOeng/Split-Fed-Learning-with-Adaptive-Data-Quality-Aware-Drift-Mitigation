# Drift Injection, Detection, Frozen Validation, and Evaluation Guide

This guide describes the implementation currently present in this repository. It separates integrated behavior, standalone-only capabilities, unsupported behavior, and future work. References use the form **path::symbol**.

## 1. Purpose and scope

### Simple intuition

A forecasting model assumes that future inputs resemble its training data. Energy demand can change abruptly, move slowly, alternate between regimes, or become noisier. Controlled injection creates a known change. Detection records whether an online algorithm notices it, how late it reacts, and how often it raises alarms when no injected drift is present.

Detection is evaluated before mitigation because an uncalibrated alarm should not control training. Frozen inference removes learning as a confounder: prediction changes then come from changed inputs and normal model behavior, not adaptation.

### Current integrated contract

~~~text
X = drifted runtime input series
y = existing clean timestamp-based target
DQ = existing unchanged quality metadata
scope = test
detection = logging only
training = unchanged
~~~

Only input drift on the test stream is integrated. Process and target types exist in standalone contracts but are rejected by the stream integration.

## 2. End-to-end architecture

The diagram separates the clean data path, optional drift layer, unchanged SplitFed inference path, and offline evaluation path. Solid arrows are runtime data flow; dashed arrows are observational metadata or artifacts.

~~~mermaid
flowchart TB
    subgraph PREP["1 · Clean data preparation"]
        direction LR
        A["CSV / cluster pickle"] --> B["Sort by client and timestamp"]
        B --> C["Missing values<br/>forward-fill → backward-fill"]
        C --> D[("Authoritative clean series<br/>client_series")]
        D --> E["Window-start split bounds<br/>train · validation · test"]
        D --> F["Clean train-only statistics<br/>μ_train · σ_train"]
        T[("Independent clean target source<br/>client_target_lookup")]
        Q[("Existing DQ metadata<br/>unchanged lookup path")]
    end

    subgraph DRIFT["2 · Optional input-drift layer · test only"]
        direction LR
        E --> G["Detector-visible test endpoints<br/>[a, b)"]
        G --> H["Resolve onset / end<br/>absolute observation coordinates"]
        D --> I["Whole-series deterministic injection<br/>selected clients only"]
        H --> I
        I --> J[("Separate runtime series<br/>client_runtime_series")]
        D --> K["Clean fallback<br/>unselected or disabled clients"]
        I -. "visible observation records" .-> IM[("injection_manifest.csv")]
    end

    subgraph WINDOW["3 · Existing stream construction"]
        direction LR
        J --> L["Select runtime values"]
        K --> L
        L --> M["Overlapping windows<br/>X shape: B × W × F"]
        T --> N["Clean timestamp target lookup<br/>y remains unchanged"]
        Q --> O["Existing DQ lookup<br/>unchanged"]
        F --> P["Normalize demand with clean statistics"]
        M --> P
        N --> P
        O --> P
    end

    subgraph DETECT["4 · Observational detector side-channel"]
        direction LR
        P --> R["One unique endpoint per window<br/>t = window_end_indices[i]<br/>d_t = X_batch[i,-1,0]"]
        R --> PH["Page-Hinkley<br/>state isolated by full DetectorKey"]
        R --> AD["ADWIN<br/>state isolated by full DetectorKey"]
        PH --> EV[("detector_events.csv")]
        AD --> EV
    end

    subgraph SPLITFED["5 · Existing frozen SplitFed inference"]
        direction LR
        P --> CF["Existing client forward"]
        CF --> SA["Smashed activations"]
        SA --> SF["Existing server forward<br/>model.eval() + torch.no_grad()"]
        SF --> PR[("Prediction CSV<br/>test metrics JSON")]
    end

    subgraph EVAL["6 · Artifact-only evaluation"]
        direction LR
        IM --> AE["Drift evaluator"]
        EV --> AE
        PR --> AE
        AE --> DS[("client_detection_summary.csv")]
        AE --> FS[("forecast_segment_summary.csv")]
        AE --> ES[("experiment_summary.json")]
        AE --> FV[("frozen_validation_report.json")]
    end

    subgraph PROTECTED["Training behavior not entered or modified"]
        direction LR
        CB["Client backward"] --- SB["Server backward"]
        SB --- GA["Gradient accumulation"]
        GA --- OP["Optimizer steps"]
        OP --- FA["FedAvg / cluster aggregation"]
        FA --- DQW["DQ weighting equations"]
    end

    SF -. "training path is not entered" .-> CB

    classDef clean fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    classDef drift fill:#fff3e0,stroke:#ef6c00,color:#7f3600;
    classDef detect fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;
    classDef frozen fill:#ede7f6,stroke:#5e35b1,color:#311b92;
    classDef artifact fill:#f5f5f5,stroke:#616161,color:#212121;
    classDef protected fill:#ffebee,stroke:#c62828,color:#7f0000,stroke-dasharray: 5 5;

    class A,B,C,D,E,F,T,Q,K,L,M,N,O,P clean;
    class G,H,I,J drift;
    class R,PH,AD detect;
    class CF,SA,SF frozen;
    class IM,EV,PR,DS,FS,ES,FV artifact;
    class CB,SB,GA,OP,FA,DQW protected;
~~~

The only value substituted into the existing input path is the selected client's runtime series. Targets and DQ metadata keep their existing sources. Detection branches from the normalized batch as a side-channel and never feeds a value back into the model or optimizer.

Drift therefore does not alter client backward propagation, server backward propagation, gradient accumulation, optimizer equations, FedAvg, DQ weighting, model architecture, checkpoint schema, or the training schedule. Frozen mode skips training and guards mutation routes.
## 3. Important concepts and terminology

- **Stationary data:** its statistical behavior remains approximately stable.
- **Distribution shift/data drift:** incoming values no longer follow the earlier distribution.
- **Input/covariate drift:** model inputs change. This is the integrated case.
- **Process drift:** the underlying generator changes, potentially changing history and future outcomes. Not integrated.
- **Target/concept drift:** target values or their relationship to inputs changes. Not integrated.
- **Sudden drift:** a one-coordinate switch.
- **Gradual drift:** old and new regimes coexist probabilistically during a transition.
- **Incremental drift:** values continuously interpolate from old to new.
- **Recurring drift:** active and inactive regimes repeat.
- **Magnitude:** multiplicative strength of the injected change.
- **Detector observation:** one scalar submitted to one detector.
- **Alarm:** River reports drift on an accepted update.
- **False alarm:** an alarm before onset, in a recurring inactive interval, or anywhere on a clean client.
- **Valid detection:** an eligible alarm in a valid drift regime.
- **Detection delay:** valid alarm coordinate minus its relevant onset.
- **Warm-up:** observations needed before detection is eligible.
- **Immutable base series:** read-only clean NumPy values returned by the injector.
- **Runtime series:** separate values used to construct drifted windows.
- **Detector-visible test segment:** absolute observations that occur as endpoints of test windows.

If clean input is (x_t), drift weight is (w(t)), and runtime input is (x'_t), the integrated detector receives

$$ dₜ = X_batch[i, −1, 0],    t = window_end_indices[i]. $$

This is standardized demand at one unique absolute endpoint.

### Mathematical notation used throughout

| Symbol | Meaning |
|---|---|
| (t) | Absolute observation index in one client series |
| (x_t) | Clean input observation at (t) |
| (x'_t) | Runtime observation after optional injection |
| (w(t)) | Theoretical drift weight in ([0,1]) |
| (s) | Resolved absolute drift start |
| (e) | Resolved transition end |
| (P) | Complete recurring cycle length |
| (W) | Input lookback/window size |
| (H) | Forecast horizon |
| (d_t) | Scalar submitted to a detector |
| (y_q,hat y_q) | Clean target and model prediction at target coordinate (q) |

The integrated experiment can be summarized as

$$ x′ₜ=T(xₜ,w(t)) for a selected client with t∈[a,b); otherwise x′ₜ=xₜ. $$

where [a,b) is the detector-visible test segment and T is the configured scaling, offset, and noise transformation. The forecasting target remains clean:

$$ X_w=(x′_w,x′_{w+1},…,x′_{w+W−1}),    q=w+W−1+H,    y_q=y_q^{clean}. $$

For irregular or independently sourced timestamps, (q) is conceptual: the repository's timestamp lookup is authoritative.

## 4. Standalone drift-package architecture

~~~text
contracts ← framework ← injector
contracts ← detectors
contracts ← evaluator
contracts ← artifacts ← frozen_validation
~~~

- **drift/contracts.py** defines strict enums, dataclasses, metadata and results.
- **drift/framework.py** contains pure absolute-index equations.
- **drift/injector.py** applies those equations once to a complete series.
- **drift/detectors.py** owns isolated River state.
- **drift/evaluator.py** evaluates artifacts without models, scalers, splitting, or injection.
- **drift/artifacts.py** writes CSV and JSON.
- **drift/frozen_validation.py** hashes checkpoint parameters.

The compatibility files **drift/drift_framework.py**, **drift/drift_injector.py**, and **drift/eval-drift.py** delegate to these modules. They are deprecated wrappers, not duplicate implementations.

## 5. drift/contracts.py

### Enums

| Enum | Values | Meaning |
|---|---|---|
| DriftType | none, sudden, gradual, incremental, recurring | Weight schedule |
| DriftSource | input, process, target | Conceptual source; only input is integrated |
| TransitionFunction | linear, sigmoid, exponential | Transition shape |
| DetectorName | page_hinkley, adwin | River detector |

**_StrictEnum.parse** lowercases and validates names. Unknown values raise; there is no fallback.

### DriftConfig

| Field | Type | Meaning | Example | Used by |
|---|---|---|---|---|
| run_id | str | Experiment identity | run42 | Records and detector key |
| configuration_id | str | Configuration label | sudden-m05 | Manifest |
| cluster_id | str | Cluster identity | 2 | Key and seed |
| client_id | str | Client identity | 9902188 | Key and seed |
| mode | str | Runtime mode | test | Identity |
| scope | str | Supported scope | test | Validation |
| drift_type | enum | Weight schedule | sudden | Framework |
| drift_source | enum | Source | input | Injector |
| start_index | int or None | Resolved absolute onset | 7500 | Framework/evaluator |
| end_index | int or None | Transition end | 7800 | Gradual/incremental |
| magnitude | float | Multiplicative effect | 0.5 | Transformation |
| offset_delta | float | Additive effect | 2 | Transformation |
| noise_sigma | float | Full-effect noise SD | 0.2 | Transformation |
| label_delta | float | Standalone target offset | 1 | Not integrated |
| period | int or None | Recurring full cycle | 168 | Framework |
| transition | enum | Transition curve | linear | Framework |
| random_seed | int | Local injector seed | derived | Injector |
| sigmoid_k | float | Sigmoid steepness | 10 | No runtime env flag |
| exp_alpha | float | Exponential curvature | 3 | No runtime env flag |
| requested_start_ratio | float or None | Requested ratio | 0.5 | Audit metadata |
| requested_end_ratio | float or None | Requested end | 0.8 | Audit metadata |
| segment_start_index | int or None | Visible start | 7000 | Audit metadata |
| segment_end_index_exclusive | int or None | Visible end | 8000 | Audit metadata |

Validation requires nonempty identity fields, finite numeric effects, nonnegative magnitude and noise, positive sigmoid steepness, scope equal to test, paired valid segment bounds, ratios in ([0,1]), a nonnegative onset, end greater than start for transitions, and a positive recurring period.

### InjectionRecord and InjectionResult

One **InjectionRecord** represents one absolute observation. It stores base and runtime values, timestamp, theoretical weight, configured boundaries, whether a regime is active, whether the numeric value changed, effects, seed, period, interval ID, ratios, and segment metadata.

**regime_active** means (w(t)>0). **transform_applied** means the runtime value differs numerically. They can differ:

- gradual drift can have positive probability but draw the old regime;
- effects can all be zero;
- a transformation can be numerically equal under the (10^{-12}) comparison tolerance.

**InjectionResult** contains a read-only base copy, separate runtime array, and immutable tuple of records.

### Detector contracts

**DetectorKey** isolates state by run, cluster, client, mode, stream, and detector.

**DetectorParameters** contains and validates PH and ADWIN parameters.

**DetectorEvent** represents every accepted unique update, not merely alarms. **event_index** is detector update order; **observation_index** is absolute series time. **detector_min_instances** contains PH warm-up and is null for ADWIN.

### Evaluation contracts

- **OnsetEvaluation:** pre/post alarm partitions, first valid detection, delay and warm-up eligibility.
- **IntervalEvaluation:** one recurring active interval.
- **DriftEvaluationResult:** complete detector/client result.
- **ForecastSegment:** one window classification.
- **ForecastSegmentMetrics:** count, MAE and RMSE.

**validate_series** accepts only nonempty finite one-dimensional data.

## 6. drift/framework.py

### transition_progress

**Purpose:** return (w(t)in[0,1]) for an absolute coordinate.  
**Input:** nonnegative observation index and validated configuration.  
**Output:** scalar weight.  
**State:** none; this function is pure.

#### Sudden

$$ w(t)=0 when t<s;    w(t)=1 when t≥s. $$

Here (s) is the resolved full-series observation index.

#### Gradual and incremental base weight

$$ w(t)=0 for t≤s;    w(t)=f((t−s)/(e−s)) for s<t<e;    w(t)=1 for t≥e. $$

At the configured start, weight is exactly zero. Evaluation still uses the configured boundary as its primary transition onset.

Gradual injection turns intermediate weight into a Bernoulli old/new choice. Incremental injection uses the continuous weight directly:

$$ x′ₜ=(1−w(t))xₜ+w(t)x̃ₜ. $$

For the implemented multiplicative/offset transform this is algebraically represented by applying weighted effects.

#### Recurring

For (t ≥ s),

$$ j=(t−s) mod P,    L=floor((P+1)/2). $$

Weight is one when (j<L), otherwise zero. This is a square wave, not sinusoidal seasonality. Odd periods contain one more active observation.

#### Transition functions

With (u=(t-s)/(e-s)):

$$ f_linear(u)=u. $$

For sigmoid, code normalizes a logistic curve so endpoints are exactly zero and one:

$$ g(u)=1/(1+exp(−k(u−1/2))),    f_sigmoid(u)=(g(u)−g(0))/(g(1)−g(0)). $$

For exponential:

$$ f_exponential(u)=(exp(αu)−1)/(exp(α)−1). $$

Approximate default values:

| Progress | Linear | Sigmoid k=10 | Exponential alpha=3 |
|---:|---:|---:|---:|
| 0% | 0.000 | 0.000 | 0.000 |
| 25% | 0.250 | 0.070 | 0.059 |
| 50% | 0.500 | 0.500 | 0.182 |
| 75% | 0.750 | 0.930 | 0.445 |
| 100% | 1.000 | 1.000 | 1.000 |

### active_interval_id

Returns the zero-based recurring cycle number only while its active half is present; otherwise it returns null.

### transform_value

Exact order:

$$ v₁=x(1+wm),    v₂=v₁+wδ,    x′=v₂+ε,    ε∼N(0,(wσ)²). $$

It requires a local RNG for noise and returns transformed value plus a Boolean numerical-change flag.

## 7. Drift transformations

### Multiplicative magnitude

$$ x′=x(1+wm). $$

Magnitude 0.5 means a 50% increase at full weight. Magnitude 1.0 doubles the value. Negative magnitude is rejected.

Example:

~~~text
base = 10
weight = 0.6
magnitude = 0.5
runtime = 10 × (1 + 0.6 × 0.5) = 13
~~~

### Offset

$$ x′=x+wδ. $$

Scaling depends on the base; offset adds the same full-effect amount.

### Noise

$$ εₜ∼N(0,(w(t)σ)²). $$

Stable client seeds make results reproducible without changing global RNG state.

### Combined transformation

Scaling is applied first, then offset, then Gaussian noise.

## 8. drift/injector.py

### derive_client_seed

SHA-256 hashes base seed, run, cluster and client separated by NUL bytes. The first eight bytes are reduced below (2^{63}-1). Unlike Python hash, this is stable between processes.

### inject_series

1. Validate and copy the complete series.
2. make the base copy read-only;
3. make a separate runtime copy;
4. validate timestamp length;
5. create a local NumPy generator;
6. iterate once per absolute observation;
7. compute weight;
8. for gradual intermediate weight, draw new regime with probability (w(t));
9. transform the scalar;
10. write one aligned record.

Injection occurs before overlapping windows. If windows are 0–23, 1–24 and 2–25, observation 24 must have one runtime value. Per-window injection could produce different noise or Bernoulli choices for observation 24. Current behavior guarantees:

~~~text
one absolute observation → one runtime value → same value in every window
~~~

## 9. Integration into data/stream_service.py

### Loading and missing values

**_prepare_from_dataframe** parses timestamps, sorts by client/time, converts values to numeric, then performs forward-fill followed by backward-fill. Remaining NaNs raise.

**_prepare_target_from_dataframe** applies the same missing-value strategy to the separate target source and creates timestamp lookup.

### Splits

For series length (N), lookback (W), and horizon (H):

$$ number_of_windows=max(0,N−W−H+1). $$

**_get_split_bounds** partitions window-start indices.

### Clean normalization

**_compute_norm_stats** uses clean values covered by training windows through train-end plus lookback and horizon. Z-score uses clean mean and standard deviation. Near-zero scale becomes one. Existing min-max normalization is also available.

### Drift helpers

**_drift_visible_bounds** converts test window-start bounds ([u,v)) to endpoint bounds:

$$ a=u+W−1,    b=v+W−1. $$

**_resolve_ratio_index** computes

$$ i=a+floor(r(b−a)). $$

Start is capped at (b-1); a transition end may equal (b).

**_drift_event_sink** writes one event immediately and increments a client/detector event count.

**_initialize_drift_runtime** runs after clean data, targets, DQ, splits and normalization exist. It validates source/scope, requires normalization for detection, selects clients, resolves boundaries, derives seeds, injects complete series, copies only visible test values, asserts outside values remain clean, stores visible manifest rows, initializes detectors, and writes resolved configuration.

Selection with an empty ID list is local to each cluster stream service. Count is Python round of client count times fraction. Explicit tokens match either a client ID or cluster:client.

### stream_next

The clean-path line is:

~~~python
values = client_runtime_series.get(
    series_key,
    client_series[series_key],
)
~~~

Only selected injected clients have runtime tensors. Targets and DQ follow unchanged lookup logic. After window construction and normalization, each detector receives exactly:

~~~python
observation_index = window_end_indices[i]
detector_value = X_batch[i, -1, 0]
~~~

Optional drift metadata is added only when a drift feature is enabled.

### Reset semantics

**stream_reset** resets only window offset. The current service has no drift-reset HTTP route. The registry has a Python reset method, but it is not exposed by the service. Replaying from earlier coordinates in the same process can therefore cause a decreasing-coordinate error.

## 10. Input and target contract

For lookback 24 and horizon 1, a window ending at input coordinate 23 forecasts the next sampling timestamp. Its history can contain drifted runtime values; the target remains the clean target found by timestamp.

This evaluates robustness to shifted input history. It is not full process drift, because the future target is not generated by the changed process. DQ lookup remains keyed by existing client, mode and window-start metadata.

## 11. Normalization

$$ zₜ=(x′ₜ−μ_train)/σ_train. $$

Order:

~~~text
clean training statistics
→ drift raw test observation
→ normalize with clean statistics
~~~

Example:

~~~text
mean = 10
scale = 2
base test value = 12
magnitude = 0.5
weight = 1
runtime raw = 12 × 1.5 = 18
normalized runtime = (18 − 10) / 2 = 4
~~~

Refitting after drift would hide part of the shift and leak test information. Detection rejects NORMALIZE=false.

## 12. Coordinate system

| Coordinate | Definition |
|---|---|
| Observation index | Full-series chronological position |
| Window start | First lookback position |
| Window end | start + W - 1 |
| Target timestamp | endpoint timestamp plus frequency times horizon |
| Conceptual target index | end + H for regular aligned data |
| Event index | Accepted detector update order |
| Detector observation index | Absolute window endpoint |
| Split bounds | Half-open window-start interval |
| Visible bounds | Half-open test endpoint interval |
| Resolved onset | Absolute drift coordinate |

Timestamp lookup remains authoritative.

Worked ratio:

~~~text
visible_start = 7000
visible_end_exclusive = 8000
ratio = 0.5
onset = 7000 + floor(0.5 × 1000) = 7500
~~~

Integrated runtime exposes ratios, not absolute index environment variables.

## 13. Runtime configuration parameters

| Parameter | Required | Default | Allowed / effect |
|---|---:|---|---|
| DRIFT_INJECTION_ENABLED | No | false | Boolean; runtime series and manifest |
| DRIFT_DETECTION_ENABLED | No | false | Boolean; detector updates and evaluation |
| DRIFT_RUN_ID | Enabled | drift-validation | Nonempty identity |
| DRIFT_CONFIGURATION_ID | Enabled | default | Nonempty label |
| DRIFT_TYPE | Injection | sudden | none, sudden, gradual, incremental, recurring |
| DRIFT_SOURCE | Enabled | input | Integrated value input only |
| DRIFT_SCOPE | Enabled | test | test only |
| DRIFT_CLIENT_IDS | No | empty | CSV client or cluster:client tokens |
| DRIFT_CLIENT_FRACTION | No | 0.5 | Per-cluster fraction in [0,1] |
| DRIFT_START_RATIO | Injection | 0.5 | Ratio in [0,1] |
| DRIFT_END_RATIO | Transition | empty | Required for gradual/incremental |
| DRIFT_MAGNITUDE | No | 0.5 | Finite nonnegative scaling |
| DRIFT_OFFSET_DELTA | No | 0 | Finite offset |
| DRIFT_NOISE_SIGMA | No | 0 | Finite nonnegative SD |
| DRIFT_PERIOD | Recurring | 168 | Positive cycle length |
| DRIFT_TRANSITION | No | linear | linear, sigmoid, exponential |
| DRIFT_RANDOM_SEED | No | 42 | Seed root |
| DRIFT_DETECTORS | Detection | page_hinkley,adwin | CSV detector enum |
| DRIFT_EVENT_BUFFER_SIZE | No | 4096 | Positive deque maximum |
| PH_MIN_INSTANCES | No | 300 | Positive PH warm-up |
| PH_DELTA | No | 0.005 | Nonnegative |
| PH_THRESHOLD | No | 50 | Positive |
| PH_ALPHA | No | 0.9999 | In (0,1] |
| ADWIN_DELTA | No | 0.002 | In (0,1) |
| ADWIN_CLOCK | No | 32 | Positive |
| ADWIN_MAX_BUCKETS | No | 5 | Positive |
| ADWIN_MIN_WINDOW_LENGTH | No | 5 | Positive |
| ADWIN_GRACE_PERIOD | No | 10 | Positive |
| FROZEN_VALIDATION_MODE | Frozen | false | Requires skipped training |
| SKIP_TRAINING | Frozen | false | Uses existing test path |
| NORMALIZE | Detection | runner true | Must be true |
| NORMALIZE_METHOD | No | zscore | zscore or minmax |
| NORMALIZE_EPS | No | 1e-8 | Scale guard |
| OUTPUT_DIR | Frozen | contextual | Staged checkpoint/output root |
| EVAL_MODE | No | final_shared | Existing checkpoint mode |
| CLUSTER_IDS | No | all | Runner filter |
| CLIENT_IDS | No | all | Runner filter |

Not implemented as environment variables: DRIFT_CLIENT_SELECTION, DRIFT_START_INDEX, DRIFT_END_INDEX, DRIFT_OUTPUT_SUBDIR, DRIFT_LOG_FLUSH_ROWS, DRIFT_LABEL_DELTA, DRIFT_SIGMOID_K, and DRIFT_EXP_ALPHA.

Important interactions:

~~~text
both drift flags false → no runtime series, registry, metadata, or drift artifacts
detection true plus normalization false → rejected
gradual/incremental → valid end ratio required
clean evaluated clients → detector updates still occur
~~~

## 14. Experiment configuration examples

Assume compatible checkpoints are staged under OUTPUT_DIR.

### A. Disabled frozen baseline

~~~bash
OUTPUT_DIR=/tmp/baseline CLUSTER_IDS=2 CLIENT_IDS=9902188 SKIP_TRAINING=true FROZEN_VALIDATION_MODE=true DRIFT_INJECTION_ENABLED=false DRIFT_DETECTION_ENABLED=false ./venv/bin/python run_split.py
~~~

Question: does integration have zero effect when disabled? No drift artifacts are produced.

### B. Sudden input drift

~~~bash
OUTPUT_DIR=/tmp/sudden CLUSTER_IDS=2 CLIENT_IDS=9902188,9905509 SKIP_TRAINING=true FROZEN_VALIDATION_MODE=true NORMALIZE=true DRIFT_INJECTION_ENABLED=true DRIFT_DETECTION_ENABLED=true DRIFT_SOURCE=input DRIFT_SCOPE=test DRIFT_CLIENT_IDS=2:9902188 DRIFT_TYPE=sudden DRIFT_START_RATIO=0.5 DRIFT_MAGNITUDE=1.0 ./venv/bin/python run_split.py
~~~

Question: detection and forecast degradation after an abrupt doubling.

### C. Gradual

~~~bash
DRIFT_TYPE=gradual DRIFT_START_RATIO=0.3 DRIFT_END_RATIO=0.7 DRIFT_TRANSITION=linear DRIFT_MAGNITUDE=0.5
~~~

### D. Incremental

~~~bash
DRIFT_TYPE=incremental DRIFT_START_RATIO=0.3 DRIFT_END_RATIO=0.7 DRIFT_TRANSITION=linear DRIFT_MAGNITUDE=0.5
~~~

### E. Recurring

~~~bash
DRIFT_TYPE=recurring DRIFT_START_RATIO=0.3 DRIFT_PERIOD=168 DRIFT_MAGNITUDE=0.5
~~~

### F. Magnitude comparison

Hold everything fixed and repeat for 0.1, 0.25, 0.5, and 1.0.

### G. Calibration

Hold injection fixed; separately sweep PH and ADWIN parameters. Do not interpret simultaneously changed injection and detector settings as a detector-only comparison.

## 15. drift/detectors.py

**_build_detector** maps validated parameters to River 0.25.

**_DetectorState** stores River object, last coordinate, bounded event deque, all alarm coordinates, and monotonic event count.

**DetectorRegistry.update** validates input, lazily creates a full-key state, rejects decreasing coordinates, skips equal duplicates, updates River once, creates one event, invokes the sink, stores alarms, and advances coordinate.

- **events:** buffered event tail.
- **alarms:** all retained alarm coordinates.
- **drain_events:** returns and clears only buffered events.
- **reset:** removes one or all River states.
- **keys:** active identities.

The event deque is bounded; the alarm list is not.

## 16. Page-Hinkley

Page-Hinkley watches whether values remain consistently above or below their historical mean.

Verified River 0.25 behavior uses the running mean and deviation dₜ=xₜ−x̄ₜ:

$$ Sₜ⁺=αSₜ₋₁⁺+dₜ−δ,    Sₜ⁻=αSₜ₋₁⁻+dₜ+δ. $$

It tracks minimum increase and maximum decrease sums, testing both directions after minimum instances.

- PH_MIN_INSTANCES: observations before testing.
- PH_DELTA: amount discounted from small change.
- PH_THRESHOLD: cumulative evidence threshold.
- PH_ALPHA: fading factor.

| Change | Expected effect |
|---|---|
| Lower threshold | More alarms, shorter delay, more false positives |
| Higher threshold | Fewer alarms, longer delay |
| Lower delta | Detect smaller changes, more noise sensitivity |
| Higher delta | Ignore small changes |

River resets PH internal state on the update after an alarm; registry history remains.

## 17. ADWIN

ADWIN maintains an adaptive window and compares older and newer means:

$$ |μ̂₀−μ̂₁|>ε(δ,n₀,n₁). $$

The exact confidence bound and bucket compression belong to River 0.25. Detection shrinks the adaptive window.

Smaller ADWIN_DELTA generally demands greater confidence; larger values increase sensitivity and possible false alarms. Clock controls test frequency. Bucket and minimum-window settings control compression and candidate splits. Grace period is ADWIN’s own startup control.

ADWIN does not use PH_MIN_INSTANCES. Evaluator PH warm-up fields are null for ADWIN.

## 18. Detector input stream

For X shape [2,24,1]:

~~~text
sample 0 → X[0,23,0]
sample 1 → X[1,23,0]
~~~

Their coordinates are the two window endpoints. Feeding all 24 values from every overlapping window would repeat observations and distort detector time, alarm rate, and delay.

## 19. Detector-event semantics

For onset 900 and alarms [700,957,1100]:

~~~text
700 = pre-drift false alarm
957 = first valid post-drift detection
1100 = later post-drift alarm
delay = 957 − 900 = 57
successful detection = true
false_alarm_present = true
~~~

Clean clients with any alarm are false-positive clients.

$$ alarms_per_1000=1000×alarm_count/accepted_updates. $$

Preserving all alarms prevents an early false alarm from hiding later valid detection.

## 20. Detector calibration

The smoke clean-client false-positive rate of 1.0 means every clean detector/client row alarmed at least once. This indicates poor calibration for the natural demand stream, not necessarily incorrect implementation.

Recommended sequence:

1. clean streams only;
2. select acceptable clean false-positive and alarm rates;
3. freeze detector parameters;
4. test known sudden drift;
5. measure valid detection and delay;
6. repeat gradual, incremental, and recurring cases;
7. validate clusters and seeds.

Example sweep values—not recommendations:

| Factor | Values | Category |
|---|---|---|
| PH minimum | 30,100,300,500 | Detector |
| PH delta | .001,.005,.01 | Detector |
| PH threshold | 25,50,100,200 | Detector |
| ADWIN delta | .0005,.001,.002,.01 | Detector |
| Magnitude | .1,.25,.5,1 | Injection |
| Onset | .3,.5,.7 | Injection |

## 21. drift/artifacts.py

**CsvArtifactWriter.append** validates fields, locks, opens in append mode, writes a header once, and immediately writes one row. It is not row-buffered.

**reset** deletes the CSV and resets header state.

**write_csv** resets then appends each row. Zero rows means the target may not be created.

**write_json** writes a temporary file then atomically replaces the target.

DRIFT_EVENT_BUFFER_SIZE affects detector memory, not disk flushing. DRIFT_LOG_FLUSH_ROWS is unsupported.

## 22. Generated artifacts

| Artifact | Producer | One row/object | Key interpretation |
|---|---|---|---|
| cluster_ID/drift/resolved_config.json | Stream startup | Local resolved run | Selected/clean clients and configuration |
| injection_manifest.csv | Injector integration | Visible test observation | Base/runtime, weight, timestamp, boundaries |
| detector_events.csv | Event sink | Unique update | Value, coordinates, alarm, warm-up |
| client_detection_summary.csv | Evaluator | Client/detector | Detection, delay, FP, rates |
| forecast_segment_summary.csv | Evaluator | Client/segment | Count, MAE, RMSE |
| experiment_summary.json | Evaluator | Run aggregate | Rates, mean delay, counts |
| frozen_validation_report.json | Runner | Frozen evidence | Before/after hashes and orchestration counts |

Synthetic examples in this guide are illustrative, not actual results.

## 23. drift/evaluator.py

Flow:

~~~text
load manifests and events
→ group by run/cluster/client/mode/stream/detector
→ identify clean or drifted
→ apply PH warm-up metadata
→ partition alarms
→ evaluate recurring intervals
→ load client prediction CSV
→ classify windows
→ write summaries
~~~

**evaluate_detection** rejects empty, mixed-identity, or duplicate-coordinate sequences. Clean clients have no primary onset and every alarm is false. Gradual/incremental results include configured onset and effective-0.5 onset. Recurring results evaluate active intervals and inactive alarms separately.

**evaluate_run_artifacts** tolerates no event files and writes zero-count summaries. Missing prediction artifacts skip segmentation.

Segment evaluator computes MAE and RMSE only. Existing client metrics additionally calculate MSE, sMAPE, and R². Client R² is null for constant targets; sMAPE is null when every denominator is zero.

## 24. Forecast segmentation

For sudden onset (s):

$$ pre: window_end<s;    mixed: window_start<s≤window_end;    fully_drifted: window_start≥s. $$

With onset 100:

| Window | Segment |
|---|---|
| 70–93 | Pre |
| 77–100 | Mixed |
| 90–113 | Mixed |
| 100–123 | Fully drifted |

Mixed windows can degrade forecasts because part of the lookback is already changed.

$$ MAE=(1/n)Σᵢ|ŷᵢ−yᵢ|,    RMSE=√((1/n)Σᵢ(ŷᵢ−yᵢ)²). $$

Segment-level sMAPE and R² are not currently implemented.

## 25. Frozen validation

**state_dict_hash** hashes sorted name, dtype, shape, and tensor bytes. **checkpoint_parameter_hash** requires model_state_dict. **snapshot_checkpoint_hashes** hashes every pt file recursively. **write_frozen_validation_report** compares before/after maps.

Valid workflow:

1. stage compatible checkpoints in a separate output root;
2. hash before;
3. enable skipped training and frozen mode;
4. start isolated services;
5. run existing client test path;
6. load client/server checkpoints;
7. perform no-grad inference;
8. generate prediction and drift artifacts;
9. hash after;
10. write report.

Server permits health, configuration, checkpoint loading, and inference. It rejects round start, training, gradient flushing, saving, and FedAvg load_weights. Frozen inference sets eval and no-grad.

Caveats: validate_step is not explicitly frozen-guarded. Server checkpoint loading restores optimizer state and counters, although inference never steps the optimizer and persisted parameter hashes remain unchanged.

## 26. run_split.py and server/server.py

**start_stream_service** passes cluster OUTPUT_DIR, placing stream artifacts under each cluster.

**run_split.py::main** requires frozen mode plus skipped training, runs the unchanged test path, invokes artifact evaluation when detection is enabled, then hashes checkpoints.

The frozen report receives zero round-start and FedAvg counts from the chosen orchestration path; it does not scan logs to calculate them.

**server::inference_step** sets eval and avoids pre-inference flush only in frozen mode. Normal behavior remains unchanged when frozen is false.

Server residual detection is not implemented.

## 27. Tests and guarantees

| Test file | Guarantee |
|---|---|
| test_drift_standalone.py | Types, validation, reproducibility, immutability, detector/evaluator semantics |
| test_drift_stream_integration.py | Test-only runtime change and clean normalization source |
| test_drift_detector_stream.py | Unique endpoints and reset separation concept |
| test_drift_evaluator_integration.py | Clean/drifted artifact aggregation |
| test_frozen_drift_validation.py | Mutation guards, no-grad, stable server hash |
| test_drift_disabled_regression.py | Disabled branch and protected paths |

Recorded results:

~~~text
23/23 standalone passed
31/31 focused drift/frozen passed
87/87 complete repository passed
~~~

A live disabled check also verified identical clean normalized X, no optional metadata, no runtime series, no detector registry, and unchanged Python, NumPy, and PyTorch RNG states.

## 28. Real smoke result

~~~text
drifted:
PH 1748 updates / 51 alarms
ADWIN 1748 updates / 11 alarms

clean:
PH 1748 updates / 24 alarms
ADWIN 1748 updates / 3 alarms
~~~

Update count is accepted unique endpoints; alarm count is a subset. Detection rate 1.0 means valid post-drift detection occurred. Clean-client false-positive rate 1.0 means every clean detector/client row had at least one alarm, not that every observation alarmed. Mean delay 57 averages successful primary delays.

The partitioning implementation can be correct while calibration is poor. These parameters are not suitable final thesis settings.

## 29. Recommended protocol

- **Experiment 0:** disabled regression; success means equal outputs and hashes.
- **Experiment 1:** clean detection calibration; vary detector parameters only.
- **Experiment 2:** sudden sensitivity; freeze detector settings, vary magnitude/onset.
- **Experiment 3:** gradual/incremental; vary transition duration and shape.
- **Experiment 4:** recurring; vary period and report active/inactive results.
- **Experiment 5:** prevalence using supported per-cluster fractions 0.25, 0.5, 0.75; persist resolved lists.
- **Experiment 6:** four clusters and 84 clients across predefined seeds.

For every experiment preserve resolved config, manifest, events, detection summary, forecast summary, prediction artifacts, hashes, and success criteria.

## 30. Common mistakes and troubleshooting

| Problem | Remedy/current behavior |
|---|---|
| Detection plus normalization false | Rejected; enable normalization |
| Invalid source/scope | Rejected |
| Absolute start/end env flags | Unsupported |
| Gradual without end | Rejected |
| Invalid recurring period | Rejected |
| No selected clients | Check local cluster, explicit tokens, and rounding |
| Missing checkpoint | Stage compatible named files under OUTPUT_DIR |
| Occupied ports | Use isolated ports and verify returned identities |
| Empty events | Confirm detection, test execution, and artifact path |
| No alarms | Inspect warm-up and sensitivity; may be valid |
| Too many alarms | Calibrate on clean streams |
| Warm-up after onset | Move onset or report minimum achievable delay |
| X/y misunderstood | X is runtime input; y is clean timestamp target |
| Repeated ordinary reset | Does not reset detector; earlier coordinates can fail |
| Every alarm considered a new drift | Use first valid detection and alarm-rate metrics |
| Event vs observation index | Sequence order versus absolute series time |
| Missing empty manifest | Zero-row writer may not create file |
| Disabled baseline expected metrics | Enable detection while leaving injection off |

## 31. Current limitations

- integrated input drift only;
- test scope only;
- standardized window-end demand only;
- process-local detector state;
- no restart persistence;
- no exposed drift reset;
- no residual or training-stream detection;
- no mitigation or adaptation;
- no integrated process or target drift;
- no automatic calibration;
- MAE/RMSE only per forecast segment;
- local-cluster fraction selection and rounding;
- unknown explicit clients not globally rejected;
- orchestration-provided zero event counts in frozen report;
- two-client smoke validation only.

## 32. Future extensions

Future work—not current behavior—includes calibrated detector settings, server residual detection, process/target drift, event-time training detection, persistent detector state, explicit experiment reset, detector-triggered mitigation, adaptation, long-term retention, and richer segment metrics.

## Appendix A. Function-level reference

### drift/framework.py::transition_progress

- Called once per injected observation.
- Reads absolute coordinate and configuration.
- Returns weight and modifies no state.
- Rejects negative coordinates.
- Implements sudden, recurring, and transition schedules described above.

### drift/framework.py::active_interval_id

- Called while building records.
- Returns recurring cycle ID only during active phases.
- Modifies no state.

### drift/framework.py::transform_value

- Called by injector.
- Reads scalar, weight, effects, and optional RNG.
- Returns transformed scalar and changed flag.
- Rejects weight outside [0,1] and missing RNG for enabled noise.

### drift/injector.py::derive_client_seed

- Called for selection and per-client injection.
- Returns deterministic integer.
- Reads no global RNG and modifies no state.

### drift/injector.py::inject_series

- Called at stream startup and standalone tests.
- Reads base values, config, optional timestamps.
- Returns immutable base, runtime values, and records.
- Uses only its local RNG.

### data/stream_service.py::_drift_visible_bounds

- Called for each selected test client.
- Reads split bounds and lookback.
- Returns absolute endpoint half-open interval.

### data/stream_service.py::_resolve_ratio_index

- Called for onset/end resolution.
- Uses floor and endpoint capping.
- Rejects ratio outside [0,1].

### data/stream_service.py::_drift_event_sink

- Called for every accepted update.
- Writes one event and updates event count.
- Disk errors propagate.

### data/stream_service.py::_initialize_drift_runtime

- Called once after clean data/DQ loading.
- Reads configuration, clean series, timestamps, splits.
- Writes runtime caches and artifacts; optionally registry.
- Returns immediately when both flags are false.

### data/stream_service.py::stream_next

- Called per requested batch.
- Reads clean/runtime values, targets, DQ, normalization and offset.
- Advances only stream offset.
- Builds X/y, updates detectors, returns serialized payload.

### drift/detectors.py::DetectorRegistry.update

- Called once per endpoint and detector.
- Reads and modifies one isolated River state.
- Returns event or null for duplicate.
- Rejects decreasing coordinate and nonfinite value.

### drift/detectors.py::events, alarms, drain_events, reset, keys

- Inspect or manage registry state.
- drain clears buffered events only.
- reset removes River state.
- alarms returns full retained alarm history.

### drift/evaluator.py::evaluate_detection

- Called for one identity.
- Reads persisted events and optional records.
- Returns detection result without modifying inputs.
- Rejects empty/mixed/duplicate sequences.

### drift/evaluator.py::evaluate_forecasts_by_segment

- Reads aligned coordinates, targets and predictions.
- Returns three segment metric records.
- Rejects unequal lengths and nonfinite values.

### drift/evaluator.py::evaluate_run_artifacts

- Called by runner or CLI.
- Reads cluster drift CSVs and prediction CSVs.
- Writes client, forecast, and experiment summaries.
- Tolerates missing event files with zero-count output.

### drift/artifacts.py::CsvArtifactWriter

- append writes one row immediately;
- reset removes file;
- for_dataclass derives exact headers.

### drift/frozen_validation.py::state_dict_hash

- Reads sorted tensor state.
- Returns SHA-256.
- Modifies no tensors.

### drift/frozen_validation.py::snapshot_checkpoint_hashes

- Recursively hashes all pt files under root.
- Raises for incompatible checkpoint contracts.

### drift/frozen_validation.py::write_frozen_validation_report

- Compares hash maps and writes JSON.
- Pass condition requires unchanged hashes and zero supplied round/FedAvg counts.

### server/server.py::SplitLearningServerState.inference_step

- Reads smashed activations and model.
- In frozen mode sets eval and no-grad.
- Returns predictions plus false gradients-enabled flag.
- Does not update parameters.

### server/server.py::SplitLearningServerState.parameter_hash

- Hashes current server parameters exactly as state_dict_hash.
- Returns hexadecimal SHA-256.

### run_split.py::main frozen branch

- Validates frozen plus skipped training.
- Hashes before, launches existing test path, evaluates artifacts, hashes after, and writes report.

## Controlled 2x2 offline diagnostic

The milestone diagnostic is implemented by `drift/run_diagnostic_2x2.py` and
`drift/diagnostic_2x2.py`. It does not add oracle or diagnostic-DQ state to the
stream service.

The primary execution uses all 84 E3 seed-42 clients. The existing selected 32
(eight per cluster) are affected; the remaining 52 are clean controls in every
condition:

| Condition | Materialized input | Affected clients | Controls |
|---|---|---:|---:|
| A | `x` | 0 | 52 plus the clean selected group |
| B | `G(x)` | selected 32 drift | 52 clean |
| C | `F(x)` | selected 32 faults | 52 clean |
| D | `G(F(x))` | selected 32 faults then drift | 52 clean |

`F` is the existing controlled-window fault injector restricted to the exact
test-input observation union 6988 through 8758. `G` is
`drift.injector.inject_series`, which uses the transformations in
`drift.framework`. For D the enforced order is clean raw values, faults, sudden
drift, the existing stream `ffill/bfill`, then unchanged clean-training
normalization. Missing values are restored after materialized drift and are not
imputed before the stream path.

The perfect oracle is generated offline from each condition manifest. It emits
one onset event for every selected client in B and D at endpoint 7885, and no
events in A or C. Oracle artifacts are joined offline by condition, cluster,
client, mode, window-start index, and window-end index. They cannot control
runtime behavior.

DQ scores are also condition-specific offline artifacts. All model-affecting
DQ flags remain false, `QUALITY_SCORE_PATH` is empty during inference, and DQ is
joined to predictions only after frozen runs complete.

PH and ADWIN run through the existing stream detector registry with runtime
drift injection disabled. PH is the primary statistical comparison detector;
ADWIN is secondary. Both are logging-only. The fixed parameters are:

```text
PH_MIN_INSTANCES=500
PH_DELTA=0.02
PH_THRESHOLD=450
PH_ALPHA=0.9999
ADWIN_DELTA=1e-16
ADWIN_CLOCK=32
ADWIN_MAX_BUCKETS=5
ADWIN_MIN_WINDOW_LENGTH=192
ADWIN_GRACE_PERIOD=500
```

Smoke execution (the selected 32 only):

```bash
./venv/bin/python -m drift.run_diagnostic_2x2 all \
  --config drift/configs/diagnostic_2x2.yaml \
  --output-root outputs/diagnostic-2x2-smoke-e3seed42 \
  --smoke
```

Primary 84-client execution:

```bash
./venv/bin/python -m drift.run_diagnostic_2x2 all \
  --config drift/configs/diagnostic_2x2.yaml \
  --output-root outputs/diagnostic-2x2-e3seed42
```

The diagnostic YAML uses dedicated stream/server/client-worker base ports
(`15101`, `15002`, and `16101`) so an unrelated run on the repository defaults
cannot cross-wire workers. The runner refuses to replace incomplete condition
directories, but it can safely reuse a condition whose frozen validation report
already proves unchanged parameters. `prepare`, `validate`, `run`, and
`evaluate` can also be invoked as separate subcommands. The authoritative
oracle-aware results are written under the experiment-level `results/`
directory; the legacy per-run drift evaluator output remains available but has
no external condition manifest and is not the primary report for this
experiment.
