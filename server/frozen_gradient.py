"""Activation gradients for client-only adaptation with an immutable split server."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def frozen_server_activation_step(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    activations: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device | str,
) -> dict[str, Any]:
    """Return d(loss)/d(activation) without mutating server-owned state.

    Evaluation mode is intentional: it disables dropout and prevents BatchNorm
    or other training-mode buffers from changing. ``torch.autograd.grad`` asks
    only for the activation derivative, so parameter ``.grad`` fields and the
    optimizer are not touched.
    """
    model.eval()
    input_was_2d = activations.dim() == 2
    server_input = activations.detach().to(device)
    if input_was_2d:
        server_input = server_input.unsqueeze(1)
    if server_input.dim() != 3:
        raise ValueError(f"server activations must be [B,H] or [B,T,H], got {tuple(server_input.shape)}")
    server_input.requires_grad_(True)
    target_tensor = targets.detach().to(device)
    if target_tensor.dim() == 1:
        target_tensor = target_tensor.unsqueeze(1)
    # cuDNN RNN backward rejects evaluation-mode modules even when the only
    # requested derivative is with respect to the input. M2 must keep the
    # server in evaluation mode so dropout and persistent buffers cannot
    # change. On CUDA, use PyTorch's non-cuDNN RNN implementation for this
    # frozen activation-gradient-only call. The normal M1 path is untouched.
    backend = (
        torch.backends.cudnn.flags(enabled=False)
        if server_input.is_cuda and torch.backends.cudnn.is_available()
        else nullcontext()
    )
    with backend:
        predictions = model(server_input)
        loss = criterion(predictions, target_tensor)
        (gradient,) = torch.autograd.grad(
            loss,
            server_input,
            retain_graph=False,
            create_graph=False,
        )
    if input_was_2d and gradient.dim() == 3 and gradient.size(1) == 1:
        gradient = gradient.squeeze(1)
    return {
        "grad_activations": gradient.detach().cpu(),
        "predictions": predictions.detach().cpu(),
        "loss": float(loss.detach().cpu().item()),
        "server_mode": "eval_activation_gradient_only",
        "cudnn_rnn_disabled_for_input_gradient": bool(
            server_input.is_cuda and torch.backends.cudnn.is_available()
        ),
    }
