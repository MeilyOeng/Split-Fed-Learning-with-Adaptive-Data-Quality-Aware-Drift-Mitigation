"""Canonical entry point: ``python -m drift.mitigation.cli`` runs the prepared,
fail-closed offline mitigation pipeline (:mod:`drift.mitigation.prepared.cli`).

For the historical runtime-processing pipeline (generation 1, does not use the
fail-closed prepared dataset backend), use ``python -m drift.mitigation.legacy.cli``
explicitly, which requires ``--allow-legacy-runtime-pipeline`` on every executing
command.
"""

from __future__ import annotations

from .prepared.cli import DEFAULT_CONFIG, DEFAULT_ASSIGNMENTS, DEFAULT_SOURCE, build_parser, main

__all__ = ["DEFAULT_CONFIG", "DEFAULT_ASSIGNMENTS", "DEFAULT_SOURCE", "build_parser", "main"]

if __name__ == "__main__":
    raise SystemExit(main())