#!/usr/bin/env python3
"""Compatibility CLI for the standalone artifact evaluator."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drift.evaluator import main


if __name__ == "__main__":
    raise SystemExit(main())
