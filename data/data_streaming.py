"""
Client-side iterable dataset that pulls batches from a cluster-aware streaming service.

Filters dataset batches using `client_id` and `cluster_id`. Fetches only batches
that belong to the specific client and cluster.
"""

import pickle
from typing import Iterator, Tuple, Optional

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
        """Initialize the streaming dataset.

        Args:
            api_url: Base URL for the streaming API service.
            client_id: Unique identifier for the client.
            cluster_id: Cluster ID to which the client belongs.
            mode: Operation mode (`train`, `val`, `inference`).
            batch_size: Number of samples per batch.
            timeout: API response timeout (in seconds).
            max_retries: Maximum retry attempts for failed requests.
            backoff: Linear backoff time (in seconds) between retries.
        """
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
        """Create or retrieve an HTTP session for efficient API calls."""
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Streaming iterator to fetch batches from the server.

        Returns:
            Iterator of (X_batch, y_batch) tuples.
        """
        session = self._get_session()
        while True:
            attempt = 0
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
                    resp.raise_for_status()
                    break  # Success!
                except requests.RequestException as e:
                    # Retry mechanism with backoff
                    attempt += 1
                    if attempt >= self.max_retries:
                        status = getattr(resp, "status_code", "unknown") if "resp" in locals() else "no_response"
                        body = getattr(resp, "text", "")
                        raise requests.HTTPError(
                            f"stream/next returned {status} for client_id={self.client_id}, "
                            f"cluster_id={self.cluster_id}, mode={self.mode}, body={body}"
                        ) from e
                    # Apply linear backoff before retrying
                    import time
                    time.sleep(self.backoff * attempt)

            # Deserialize response payload
            payload = pickle.loads(resp.content)
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"Streaming payload must be a dict; got {type(payload).__name__}"
                )

            # Validate batch data
            if "X" not in payload or "y" not in payload or payload.get("y") is None:
                print(f"[StreamingIterableDataset] Skipping invalid batch; keys={list(payload.keys())}")
                continue

            X_batch = payload["X"]
            y_batch = payload["y"]
            if not torch.is_tensor(X_batch) or not torch.is_tensor(y_batch):
                raise RuntimeError(
                    f"Streaming payload must contain tensors; types: X={type(X_batch).__name__}, y={type(y_batch).__name__}"
                )

            # Yield the batch as (X, y)
            yield X_batch, y_batch