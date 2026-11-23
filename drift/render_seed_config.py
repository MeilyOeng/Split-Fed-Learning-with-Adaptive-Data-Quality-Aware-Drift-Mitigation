#!/usr/bin/env python3
"""Derive a per-seed mitigation config from one base template.

Overrides seeds.client_selection_seed and appends a matching _clientsel<seed>
suffix to suite_id (replacing any existing _clientsel<N> suffix), leaving
every other field from the base config untouched.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


def render(base: Path, client_selection_seed: int) -> dict:
    config = yaml.safe_load(base.read_text(encoding="utf-8"))
    config["seeds"]["client_selection_seed"] = client_selection_seed
    family = re.sub(r"_clientsel\d+$", "", str(config["suite_id"]))
    config["suite_id"] = f"{family}_clientsel{client_selection_seed}"
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--client-selection-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = render(args.base, args.client_selection_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())