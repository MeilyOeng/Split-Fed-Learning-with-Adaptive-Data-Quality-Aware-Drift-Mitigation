"""Per-window rule-based data-quality scoring for M4_quality_aware_splitfed,
wired onto the existing, independently-owned ``data quality/`` DQ-scoring
subsystem (``WideDemandQualityScorer`` + ``compute_average_quality_score``).

Computed once, population-wide, for the same eligible window set (ADAPTATION
+ FINAL_HOLDOUT, never PRE_DRIFT) that ``faults.py::apply_window_faults``
already scores -- from that same, already-materialized ``X_observed`` (raw
scale, real NaN where a fault landed), never from the raw timeline.

Design notes (see the approved plan this module implements):

  - Each window is scored with ITSELF as both ``current`` and ``clean``
    (self-referential). Using a client's full clean_train series as
    ``clean`` breaks ``FreshnessScorer``'s F2: its denominator is built
    from ``clean``'s own expected-timestamp grid via
    ``expected_timestamp_index``, which for a ~thousands-of-rows clean
    series versus a ``window_size``-row ``current`` produces a systematic
    mismatch unrelated to actual data quality.
    ``ValidityScorer``/``ConsistencyScorer`` never fall back to ``clean``
    when ``thresholds.allow_clean_reference_fallback`` is false (the real
    config's setting), so this choice only changes freshness, correctly.
  - ``A3`` (historical noise instability) is structurally neutral (``1.0``)
    for every window under per-window-isolated scoring:
    ``accuracy.historical_min_diffs`` needs that many valid trailing diffs
    before it evaluates anything, but an isolated ``window_size``-row
    window has at most ``window_size - 1`` diffs total. Accepted,
    documented limitation -- not fixed here.
  - The final blended score comes from
    ``data quality/modules/pca_quality.py::compute_average_quality_score``,
    a plain per-row weighted mean of the five dimensions (no PCA, no
    ``.fit()``, no cross-window/cross-suite renormalization) -- confirmed
    by direct code inspection before this module was written.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_DATA_QUALITY_ROOT = Path(__file__).resolve().parents[3] / "data quality"
if str(_DATA_QUALITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATA_QUALITY_ROOT))

from modules.io import load_yaml_config, load_wide_demand_dataframe  # noqa: E402
from modules.pca_quality import QUALITY_DIMENSIONS, compute_average_quality_score  # noqa: E402
from modules.scoring import WideDemandQualityScorer  # noqa: E402

_A3_NOTE = (
    "A3 (historical noise instability) is structurally neutral (1.0) for every "
    "window under per-window-isolated scoring: accuracy.historical_min_diffs "
    "requires that many valid trailing diffs before it evaluates anything, but "
    "an isolated window of window_size rows has at most window_size-1 diffs "
    "total. Accepted, documented limitation, not a bug -- completeness, "
    "validity, consistency, freshness, A1, and A2 all retain real per-window "
    "signal."
)

DQ_REPRESENTATION_METADATA = {
    "quality_weight_score": {
        "role": "dq_aggregation_weight",
        "scale": "unit_interval",
        "computed_from": "X_observed",
        "granularity": "per_window_per_client",
        "pre_drift_value": "neutral_placeholder_1.0_never_scored",
    },
}


def _hash_yaml_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_window_quality_weights(
    *,
    X_observed: dict[tuple[str, str], np.ndarray],
    segment_by_window: np.ndarray,
    window_start_index_by_window: np.ndarray,
    timestamps: np.ndarray,
    window_size: int,
    dq_config_path: str | Path,
) -> dict[str, Any]:
    """Population-wide, window-level DQ scoring.

    X_observed: {(cluster_id, client_id): array of shape [n_windows, window_size, 1]}
        (raw scale, real NaN where faults landed -- exactly
        apply_window_faults's own output, reused here without recomputation).
    segment_by_window / window_start_index_by_window / timestamps: the same
        population-invariant arrays apply_window_faults already consumes
        (segment_by_window from build_segment_rows, window_start_index_by_window
        == starts["deployment"], timestamps == reference_timestamps) --
        never re-derived here.

    Returns finite scores in [0, 1] (fail-closed otherwise), a per-client
    dimension summary, and an identity/provenance dict. Deliberately never
    persists a per-window x per-dimension artifact -- too large at real
    scale (84 clients x ~1200 eligible windows x 5 dimensions).
    """
    dq_config = load_yaml_config(dq_config_path)
    scorer = WideDemandQualityScorer(config=dq_config)
    configured_freq = dq_config.get("frequency")

    eligible_window_positions = np.flatnonzero(np.asarray(segment_by_window) != "PRE_DRIFT")
    if eligible_window_positions.size == 0:
        raise ValueError("no DQ-eligible (non-PRE_DRIFT) windows found")

    client_keys = sorted(X_observed)
    client_id_to_key = {client_id: (cluster_id, client_id) for cluster_id, client_id in client_keys}
    n_windows = len(segment_by_window)

    quality_weight_score: dict[tuple[str, str], np.ndarray] = {
        key: np.ones(n_windows, dtype=np.float64) for key in client_keys
    }
    quality_weight_score_computed: dict[tuple[str, str], np.ndarray] = {
        key: np.zeros(n_windows, dtype=bool) for key in client_keys
    }
    dimension_sums = {key: np.zeros(len(QUALITY_DIMENSIONS), dtype=np.float64) for key in client_keys}
    final_score_sum = {key: 0.0 for key in client_keys}
    final_score_min = {key: 1.0 for key in client_keys}
    below_0p5_count = {key: 0 for key in client_keys}
    scored_window_count = {key: 0 for key in client_keys}

    timestamps_arr = np.asarray(timestamps)
    for w in eligible_window_positions:
        start = int(window_start_index_by_window[w])
        ts_slice = pd.to_datetime(timestamps_arr[start:start + window_size])
        ts_col = "timestamp"
        data: dict[str, Any] = {ts_col: ts_slice}
        for key in client_keys:
            _, client_id = key
            data[client_id] = np.asarray(X_observed[key], dtype=float)[w, :, 0]
        wide_frame = pd.DataFrame(data)
        window_ds = load_wide_demand_dataframe(wide_frame, timestamp_col=ts_col, configured_freq=configured_freq)
        scores_df = scorer.score(current=window_ds, clean=window_ds)
        final_scores = compute_average_quality_score(scores_df, dq_config)

        for row, final in zip(scores_df.itertuples(index=False), final_scores):
            client_id = str(row.client_id)
            key = client_id_to_key.get(client_id)
            if key is None:
                continue
            final = float(final)
            quality_weight_score[key][w] = final
            quality_weight_score_computed[key][w] = True
            dimension_sums[key] += np.array([getattr(row, dim) for dim in QUALITY_DIMENSIONS], dtype=float)
            final_score_sum[key] += final
            final_score_min[key] = min(final_score_min[key], final)
            if final < 0.5:
                below_0p5_count[key] += 1
            scored_window_count[key] += 1

    for key in client_keys:
        arr = quality_weight_score[key]
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"quality_weight_score[{key}] contains non-finite values -- fail-closed")
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError(f"quality_weight_score[{key}] outside [0,1] -- fail-closed")

    summary_rows = []
    for key in client_keys:
        cluster_id, client_id = key
        count = max(1, scored_window_count[key])
        mean_dims = dimension_sums[key] / count
        summary_rows.append({
            "cluster_id": cluster_id, "client_id": client_id,
            **{f"mean_{dim}": float(mean_dims[i]) for i, dim in enumerate(QUALITY_DIMENSIONS)},
            "mean_final_score": final_score_sum[key] / count,
            "min_final_score": final_score_min[key],
            "below_0p5_count": below_0p5_count[key],
            "scored_window_count": scored_window_count[key],
        })
    summary_frame = pd.DataFrame(summary_rows)

    identity = {
        "dq_config_path": str(dq_config_path),
        "dq_config_sha256": _hash_yaml_file(dq_config_path),
        "window_size": int(window_size),
        "eligible_window_count": int(eligible_window_positions.size),
        "note": (
            "each eligible window is scored with itself as both current and clean "
            "(self-referential) -- a full-history clean reference breaks FreshnessScorer's "
            "F2 (denominator built from clean's own expected-timestamp grid, mismatched "
            "against a single window's row count)"
        ),
        "a3_historical_noise_instability_structurally_neutral": True,
        "a3_note": _A3_NOTE,
    }

    return {
        "quality_weight_score": quality_weight_score,
        "quality_weight_score_computed": quality_weight_score_computed,
        "summary_frame": summary_frame,
        "identity": identity,
    }
