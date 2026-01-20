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
                    if resp.status_code == 204:
                        return
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

            # Try to deserialize response payload
            # First try JSON (minimal service), then fall back to pickle (original service)
            try:
                payload = resp.json()
                # Convert lists to tensors if using minimal service
                if "X" in payload and isinstance(payload["X"], list):
                    X_batch = torch.tensor(payload["X"], dtype=torch.float32)
                    y_batch = torch.tensor(payload["y"], dtype=torch.float32)
                else:
                    X_batch = payload["X"]
                    y_batch = payload["y"]
            except (ValueError, KeyError):
                # Fall back to pickle format
                payload = pickle.loads(resp.content)
                if not isinstance(payload, dict):
                    raise RuntimeError(
                        f"Streaming payload must be a dict; got {type(payload).__name__}"
                    )
                X_batch = payload["X"]
                y_batch = payload["y"]

            # Validate batch data
            if X_batch is None or y_batch is None:
                print(f"[StreamingIterableDataset] Skipping invalid batch")
                continue

            # Ensure tensors
            if not torch.is_tensor(X_batch):
                X_batch = torch.tensor(X_batch, dtype=torch.float32)
            if not torch.is_tensor(y_batch):
                y_batch = torch.tensor(y_batch, dtype=torch.float32)

            # Yield the batch as (X, y)
            yield X_batch, y_batch
