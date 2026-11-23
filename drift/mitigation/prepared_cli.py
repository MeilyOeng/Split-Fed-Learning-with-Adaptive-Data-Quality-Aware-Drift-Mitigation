"""Backward-compatible alias: ``python -m drift.mitigation.prepared_cli`` resolves
to the exact same canonical implementation as ``python -m drift.mitigation.cli``
and ``python -m drift.mitigation.prepared.cli``. Kept so existing scripts/docs
that reference the pre-migration module path continue to work unmodified.
"""

from __future__ import annotations

from .prepared.cli import DEFAULT_CONFIG, DEFAULT_ASSIGNMENTS, DEFAULT_SOURCE, build_parser, main

__all__ = ["DEFAULT_CONFIG", "DEFAULT_ASSIGNMENTS", "DEFAULT_SOURCE", "build_parser", "main"]

if __name__ == "__main__":
    raise SystemExit(main())