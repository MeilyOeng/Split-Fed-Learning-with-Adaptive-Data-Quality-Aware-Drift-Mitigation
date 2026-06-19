"""
Client-side iterable dataset for cluster-aware streaming.
Supports modes: train, val, test.
"""

import pickle
import time
from typing import Any, Dict, Iterator, Optional, Tuple

import requests
import torch
from torch.utils.data import IterableDataset


class StreamingIterableDataset(IterableDataset):
    def __init__(
        self,
        api_url: str,
        client_id: str,
        cluster_id: str,
        mode: str,
        batch_size: int,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff: float = 1.0,
    ):
        super().__init__()
        self.api_url = api_url.rstrip("/")
        self.client_id = client_id
        self.cluster_id = cluster_id
        self.mode = mode
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._session: Optional[requests.Session] = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        session = self._get_session()
        while True:
            attempt = 0
            resp = None
            while True:
                try:
                    resp = session.post(
                        f"{self.api_url}/stream/next",
                        json={
                            "client_id": self.client_id,
                            "cluster_id": self.cluster_id,
                            "mode": self.mode,
                            "batch_size": self.batch_size,
                        },
                        timeout=self.timeout,
                    )
                    if resp.status_code == 204:
                        return
                    resp.raise_for_status()
                    break
                except requests.RequestException as e:
                    attempt += 1
                    if attempt >= self.max_retries:
                        status = getattr(resp, "status_code", "unknown")
                        body = getattr(resp, "text", "")
                        raise requests.HTTPError(
                            f"stream/next returned {status} for client_id={self.client_id}, "
                            f"cluster_id={self.cluster_id}, mode={self.mode}, body={body}"
                        ) from e
                    time.sleep(self.backoff * attempt)

            try:
                payload = resp.json()
                X_batch = payload["X"]
                y_batch = payload["y"]
            except Exception:
                payload = pickle.loads(resp.content)
                if not isinstance(payload, dict):
                    raise RuntimeError(f"Streaming payload must be dict, got {type(payload).__name__}")
                X_batch = payload["X"]
                y_batch = payload["y"]

            if not torch.is_tensor(X_batch):
                X_batch = torch.tensor(X_batch, dtype=torch.float32)
            if not torch.is_tensor(y_batch):
                y_batch = torch.tensor(y_batch, dtype=torch.float32)

            quality_scores = payload.get("quality_scores")
            if quality_scores is not None and not torch.is_tensor(quality_scores):
                quality_scores = torch.tensor(quality_scores, dtype=torch.float32)

            meta: Dict[str, Any] = {
                "cluster_id": payload.get("cluster_id"),
                "client_id": payload.get("client_id"),
                "mode": payload.get("mode", self.mode),
                "window_start_indices": payload.get("window_start_indices", []),
                "window_end_indices": payload.get("window_end_indices", []),
                "window_start_timestamps": payload.get("window_start_timestamps", []),
                "window_end_timestamps": payload.get("window_end_timestamps", []),
                "target_timestamps": payload.get("target_timestamps", []),
                "quality_scores": quality_scores,
                "quality_score_metadata": payload.get("quality_score_metadata", {}),
            }

            yield X_batch, y_batch, meta
