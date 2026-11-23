"""Single-step client optimizer contract shared by SplitFed update policies."""

from __future__ import annotations

import torch


def apply_client_optimizer_step(
    optimizer: torch.optim.Optimizer,
    *,
    update_allowed: bool,
) -> bool:
    """Apply exactly one client optimizer step when the policy permits it.

    A denied update is a strict no-op. Backward propagation, gradient clipping,
    and zeroing remain owned by the caller so their established ordering is
    unchanged.
    """
    if not update_allowed:
        return False
    optimizer.step()
    return True
