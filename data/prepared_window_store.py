"""Fail-closed reader for offline-prepared forecasting samples.

The mitigation backend is deliberately narrower than the general raw timeline
stream.  It can read only manifest-approved, already-normalized X/y samples and
never has access to raw drift residuals or drift-generation configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


APPROVED_DATASET_ROLES = frozenset({
    "clean_train",
    "clean_validation",
    "clean_deployment",
    "drifted_deployment",
    "clean_final_holdout_counterfactual",
})

H_TRAINING_ROLES = frozenset({"clean_train", "clean_validation"})

# Provenance/bookkeeping fields written alongside every prepared-sample NPZ.
# These are excluded from the hashed payload, mirroring
# drift/mitigation/prepared/preparation.py::_write_samples, which hashes only
# {"X", "y", **metadata} before these fields are added at save time.
_PROVENANCE_FIELDS = frozenset({
    "dataset_role", "cluster_id", "client_id", "representation",
    "normalization_applied_count", "scaler_hash", "coordinate_manifest_hash",
    "source_timeline_hash", "prepared_payload_hash", "schema_version",
})


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_array_payload_hash(arrays: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedClientSamples:
    role: str
    cluster_id: str
    client_id: str
    X: np.ndarray
    y: np.ndarray
    metadata: dict[str, np.ndarray]

    @property
    def size(self) -> int:
        return int(self.y.shape[0])


class PreparedWindowStore:
    """Loads only manifest-approved, hash-verified, already-normalized samples."""

    def __init__(
        self,
        root: str | Path,
        *,
        allowed_roles: Iterable[str],
        expected_coordinate_manifest_hash: str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"prepared dataset root does not exist: {self.root}")
        self.allowed_roles = frozenset(str(role) for role in allowed_roles)
        if not self.allowed_roles:
            raise ValueError("PreparedWindowStore requires at least one allowed role")
        unknown_roles = self.allowed_roles - APPROVED_DATASET_ROLES
        if unknown_roles:
            raise ValueError(f"unapproved dataset role(s) requested: {sorted(unknown_roles)}")
        manifest_path = self.root / "manifests" / "prepared_dataset_hashes.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"prepared dataset manifest not found: {manifest_path}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.coordinate_manifest_hash = str(self.manifest.get("coordinate_manifest_hash", ""))
        if (
            expected_coordinate_manifest_hash is not None
            and self.coordinate_manifest_hash != str(expected_coordinate_manifest_hash)
        ):
            raise ValueError(
                f"prepared coordinate-manifest hash mismatch: expected "
                f"{expected_coordinate_manifest_hash!r}, got {self.coordinate_manifest_hash!r}"
            )
        self._cache: dict[tuple[str, str, str], PreparedClientSamples] = {}

    def load_client(self, role: str, cluster_id: str, client_id: str) -> PreparedClientSamples:
        role = str(role)
        if role not in self.allowed_roles:
            raise PermissionError(f"role is not approved for this prepared reader: {role}")
        cluster_id, client_id = str(cluster_id), str(client_id)
        key = (role, cluster_id, client_id)
        if key in self._cache:
            return self._cache[key]
        try:
            role_payload = self.manifest["roles"][role]
        except KeyError as exc:
            raise KeyError(f"prepared manifest has no role {role!r}") from exc
        client_key = f"{cluster_id}:{client_id}"
        try:
            entry = role_payload["clients"][client_key]
        except KeyError as exc:
            raise KeyError(f"prepared manifest has no client {client_key!r} for role {role!r}") from exc
        artifact_path = self.root / entry["path"]
        actual_artifact_hash = file_sha256(artifact_path)
        if actual_artifact_hash != entry["artifact_sha256"]:
            raise ValueError(
                f"prepared artifact hash mismatch for {role}/{client_key}: "
                f"expected {entry['artifact_sha256']!r}, got {actual_artifact_hash!r}"
            )
        with np.load(artifact_path, allow_pickle=False) as source:
            content = {name: np.asarray(source[name]) for name in source.files}
        normalization_applied_count = int(np.asarray(content["normalization_applied_count"]).item())
        if normalization_applied_count != 1:
            raise ValueError(
                f"prepared sample must have normalization_applied_count=1, "
                f"got {normalization_applied_count} for {role}/{client_key}"
            )
        representation = str(np.asarray(content["representation"]).item())
        if representation != "normalized":
            raise ValueError(
                f"prepared sample must have representation='normalized', "
                f"got {representation!r} for {role}/{client_key}"
            )
        payload = {name: value for name, value in content.items() if name not in _PROVENANCE_FIELDS}
        actual_payload_hash = canonical_array_payload_hash(payload)
        if actual_payload_hash != entry["payload_hash"]:
            raise ValueError(
                f"prepared payload hash mismatch for {role}/{client_key}: "
                f"expected {entry['payload_hash']!r}, got {actual_payload_hash!r}"
            )
        X = np.asarray(payload.pop("X"), dtype=np.float32)
        y = np.asarray(payload.pop("y"), dtype=np.float32).reshape(-1)
        samples = PreparedClientSamples(
            role=role, cluster_id=cluster_id, client_id=client_id, X=X, y=y, metadata=payload,
        )
        self._cache[key] = samples
        return samples
