"""Scenario / suite / condition-report state machines, plus the derived
overall per-scenario progress view.

Three independently-mutated, timestamped state machines back everything:

  scenario  (one seed+drift_type+severity+drift_count; governs the shared
             ``prepared/``):        planned -> config_generated -> preparation_accepted
  suite     (one scenario+epoch; 180 of these):
             planned -> config_generated -> preparation_accepted -> methods_complete -> evaluated
  condition (one drift_type+severity+drift_count+epoch, 5 seeds; 36 of these):
             waiting_for_seeds -> ready -> report_running -> report_complete -> archived

Each is a strict-single-step forward machine: a transition must move to
exactly the next state (or re-write the current state, a no-op used for
idempotent resume checks). Skipping ahead or moving backward raises,
never happens silently.

``scenario_overall_progress`` is a read-only projection over the three
machines -- it has no state of its own and can never drift from them:

  planned -> config_generated -> preparation_accepted -> epoch_01_complete
  -> epoch_03_complete -> evaluated -> archived
"""
from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

SCENARIO_STATES = ("planned", "config_generated", "preparation_accepted")
SUITE_STATES = ("planned", "config_generated", "preparation_accepted", "methods_complete", "evaluated")
CONDITION_STATES = ("waiting_for_seeds", "ready", "report_running", "report_complete", "archived")
OVERALL_SCENARIO_STATES = (
    "planned", "config_generated", "preparation_accepted", "epoch_01_complete",
    "epoch_03_complete", "evaluated", "archived",
)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class StateMachineError(ValueError):
    pass


class StatusStore:
    """CSV-backed key -> row store with a strict-single-step state machine.

    Every transition is timestamped in ``state_updated_at`` and appended to
    a sibling ``<name>_history.jsonl`` for a full audit trail (the CSV
    itself only ever holds the current row per key).
    """

    def __init__(self, path: Path, *, key_field: str, states: Sequence[str], fieldnames: Sequence[str]):
        self.path = Path(path)
        self.key_field = key_field
        self.states = tuple(states)
        self._order = {state: index for index, state in enumerate(self.states)}
        self.fieldnames = list(fieldnames)
        if key_field not in self.fieldnames or "state" not in self.fieldnames or "state_updated_at" not in self.fieldnames:
            raise ValueError("fieldnames must include key_field, 'state', and 'state_updated_at'")
        self._rows: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            with self.path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    self._rows[row[key_field]] = dict(row)

    @property
    def history_path(self) -> Path:
        return self.path.with_name(self.path.stem + "_history.jsonl")

    def get(self, key: str) -> dict[str, Any] | None:
        return dict(self._rows[key]) if key in self._rows else None

    def all_rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows.values()]

    def initialize(self, key: str, fields: Mapping[str, Any]) -> dict[str, Any]:
        """Create a row at its initial state if absent; idempotent no-op
        (with an identity check on immutable identity fields) if present.
        """
        initial_state = self.states[0]
        if key in self._rows:
            existing = self._rows[key]
            for field, value in fields.items():
                if field in ("state", "state_updated_at"):
                    continue
                if str(existing.get(field, "")) != str(value):
                    raise StateMachineError(
                        f"{self.path.name}: row {key!r} already exists with {field}={existing.get(field)!r}, "
                        f"cannot re-initialize with {field}={value!r} (identity fields are immutable)"
                    )
            return dict(existing)
        row = {name: "" for name in self.fieldnames}
        row.update({key: str(value) for key, value in fields.items()})
        row[self.key_field] = key
        row["state"] = initial_state
        row["state_updated_at"] = _now()
        self._rows[key] = row
        self._append_history(key, None, initial_state, row)
        self._flush()
        return dict(row)

    def transition(self, key: str, new_state: str, *, extra_fields: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if key not in self._rows:
            raise StateMachineError(f"{self.path.name}: no row for {key!r}; call initialize() first")
        if new_state not in self._order:
            raise StateMachineError(f"{self.path.name}: unknown state {new_state!r} for {key!r}")
        row = self._rows[key]
        current = row["state"]
        current_index, new_index = self._order[current], self._order[new_state]
        if new_index == current_index:
            # Idempotent re-check -- allowed, no timestamp churn beyond
            # updating extra_fields (e.g. refreshed counters).
            if extra_fields:
                row.update({key: str(value) for key, value in extra_fields.items()})
                self._flush()
            return dict(row)
        if new_index != current_index + 1:
            raise StateMachineError(
                f"{self.path.name}: illegal transition for {key!r}: {current!r} -> {new_state!r} "
                f"(must advance exactly one step; states={self.states})"
            )
        row["state"] = new_state
        row["state_updated_at"] = _now()
        if extra_fields:
            row.update({key: str(value) for key, value in extra_fields.items()})
        self._append_history(key, current, new_state, row)
        self._flush()
        return dict(row)

    def advance_to(self, key: str, target_state: str, *, extra_fields: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Idempotent, forward-only convenience over ``transition``: walk
        one step at a time up to ``target_state`` (recording every
        intermediate transition), doing nothing if already there or past
        it in a *different* branch is not possible here since this is a
        single linear chain. Raises if ``target_state`` is behind the
        current state (an actual backward jump).
        """
        if key not in self._rows:
            raise StateMachineError(f"{self.path.name}: no row for {key!r}; call initialize() first")
        target_index = self._order[target_state]
        current_index = self._order[self._rows[key]["state"]]
        if target_index < current_index:
            raise StateMachineError(
                f"{self.path.name}: refusing to move {key!r} backward from "
                f"{self._rows[key]['state']!r} to {target_state!r}"
            )
        while self._order[self._rows[key]["state"]] < target_index:
            next_state = self.states[self._order[self._rows[key]["state"]] + 1]
            self.transition(key, next_state)
        if extra_fields:
            self._rows[key].update({field: str(value) for field, value in extra_fields.items()})
            self._flush()
        return dict(self._rows[key])

    def _append_history(self, key: str, old_state: str | None, new_state: str, row: Mapping[str, Any]) -> None:
        import json
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "key": key, "from": old_state, "to": new_state,
                "at": row["state_updated_at"], "row": dict(row),
            }, default=str) + "\n")

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in self._rows.values():
                writer.writerow({name: row.get(name, "") for name in self.fieldnames})


SCENARIO_FIELDS = [
    "scenario_id", "seed", "drift_type", "severity", "drift_count", "state", "state_updated_at", "prepared_path",
]
SUITE_FIELDS = [
    "suite_id", "scenario_id", "seed", "drift_type", "severity", "drift_count", "epochs", "state",
    "state_updated_at", "suite_root", "config_path", "methods_complete_count",
]
CONDITION_FIELDS = [
    "condition_id", "drift_type", "severity", "drift_count", "epochs", "state", "state_updated_at",
    "seeds", "report_dir",
]


def scenario_store(status_dir: Path) -> StatusStore:
    return StatusStore(status_dir / "scenario_status.csv", key_field="scenario_id", states=SCENARIO_STATES, fieldnames=SCENARIO_FIELDS)


def suite_store(status_dir: Path) -> StatusStore:
    return StatusStore(status_dir / "suite_status.csv", key_field="suite_id", states=SUITE_STATES, fieldnames=SUITE_FIELDS)


def condition_store(status_dir: Path) -> StatusStore:
    return StatusStore(status_dir / "condition_status.csv", key_field="condition_id", states=CONDITION_STATES, fieldnames=CONDITION_FIELDS)


def scenario_overall_progress(
    scenario_row: Mapping[str, Any] | None,
    suite01_row: Mapping[str, Any] | None,
    suite03_row: Mapping[str, Any] | None,
    condition01_row: Mapping[str, Any] | None,
    condition03_row: Mapping[str, Any] | None,
) -> str:
    """Derived-only projection (§ safeguard 1) -- never persisted as an
    independent source of truth, always recomputed from the three real
    state machines.
    """
    scen_state = (scenario_row or {}).get("state", "planned")
    if scen_state != "preparation_accepted":
        return scen_state
    s1 = (suite01_row or {}).get("state", "planned")
    s3 = (suite03_row or {}).get("state", "planned")
    if s1 in ("planned", "config_generated", "preparation_accepted"):
        return "preparation_accepted"
    if s1 == "methods_complete":
        return "epoch_01_complete"
    # s1 == "evaluated"
    if s3 in ("planned", "config_generated", "preparation_accepted"):
        return "epoch_01_complete"
    if s3 == "methods_complete":
        return "epoch_03_complete"
    # s3 == "evaluated"
    c1 = (condition01_row or {}).get("state")
    c3 = (condition03_row or {}).get("state")
    if c1 == "archived" and c3 == "archived":
        return "archived"
    return "evaluated"


def build_scenario_status_report(
    scenario_rows: Sequence[Mapping[str, Any]],
    suites: StatusStore,
    conditions: StatusStore,
    *,
    scenario_id_fn,
    suite_id_fn,
    condition_id_fn,
) -> list[dict[str, Any]]:
    """Attach ``overall_progress`` to every scenario row for reporting
    (e.g. ``matrix status``), without writing it back into scenario_status.csv.
    """
    out = []
    for row in scenario_rows:
        seed, drift_type, severity, drift_count = row["seed"], row["drift_type"], row["severity"], row["drift_count"]
        scen_id = row["scenario_id"]
        suite01 = suites.get(suite_id_fn(scen_id, 1))
        suite03 = suites.get(suite_id_fn(scen_id, 3))
        cond01 = conditions.get(condition_id_fn(drift_type, severity, drift_count, 1))
        cond03 = conditions.get(condition_id_fn(drift_type, severity, drift_count, 3))
        out.append({**row, "overall_progress": scenario_overall_progress(row, suite01, suite03, cond01, cond03)})
    return out
