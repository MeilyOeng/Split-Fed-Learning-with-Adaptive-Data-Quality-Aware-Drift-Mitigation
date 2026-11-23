"""Prepared-data entry point that preserves the production SplitFed scheduler."""

from __future__ import annotations

import os

import run_split


def main() -> int:
    if str(os.environ.get("PREPARED_DATASET_BACKEND", "")).lower() not in {"1", "true", "yes", "y"}:
        raise ValueError("prepared SplitFed entry requires PREPARED_DATASET_BACKEND=true")
    run_split.STREAM_SERVICE_PATH = "data/prepared_stream_service.py"
    if str(os.environ.get("RUN_TEST_AFTER_TRAINING", "true")).lower() not in {"1", "true", "yes", "y"}:
        # H validation remains inside the existing epoch-finalization path.
        # This disables only the separate post-training test/deployment phase.
        run_split.run_test_phase_for_cluster = lambda *args, **kwargs: {}
    return run_split.main()


if __name__ == "__main__":
    raise SystemExit(main())

