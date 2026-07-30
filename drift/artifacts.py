"""Small bounded-memory CSV/JSON artifact writers for drift experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Iterable, Mapping


class CsvArtifactWriter:
    """Append dataclass or mapping rows directly to disk without retaining them in memory."""

    def __init__(self, path: str | Path, fieldnames: Iterable[str]) -> None:
        self.path = Path(path)
        self.fieldnames = tuple(fieldnames)
        self._lock = Lock()
        self._initialized = self.path.exists() and self.path.stat().st_size > 0

    @classmethod
    def for_dataclass(cls, path: str | Path, contract_type: type) -> "CsvArtifactWriter":
        return cls(path, (field.name for field in fields(contract_type)))

    def append(self, row: object) -> None:
        payload = asdict(row) if is_dataclass(row) else dict(row)
        unknown = sorted(set(payload) - set(self.fieldnames))
        if unknown:
            raise ValueError(f"artifact row contains unknown fields: {unknown}")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                if not self._initialized:
                    writer.writeheader()
                    self._initialized = True
                writer.writerow(payload)

    def reset(self) -> None:
        with self._lock:
            if self.path.exists():
                self.path.unlink()
            self._initialized = False


def write_csv(path: str | Path, rows: Iterable[object], contract_type: type) -> Path:
    writer = CsvArtifactWriter.for_dataclass(path, contract_type)
    writer.reset()
    for row in rows:
        writer.append(row)
    return writer.path


def write_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target
