from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from utils.runtime_utils import QualityAnchorConfig


GradientTuple = Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ClientGradientContribution:
    client_id: str
    cluster_id: str
    global_round: int
    logical_round: int
    mean_quality: float
    sample_count: int
    server_gradients: GradientTuple
    server_gradient_norm: float


@dataclass(frozen=True)
class NonAnchorAlignmentRecord:
    client_id: str
    mean_quality: float
    gradient_norm: float
    cosine_with_anchor: float | None
    alignment_weight: float
    norm_clip_scale: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class AnchorAggregationResult:
    anchor_client_ids: Tuple[str, ...]
    anchor_qualities: Tuple[float, ...]
    non_anchor_client_ids: Tuple[str, ...]
    non_anchor_qualities: Tuple[float, ...]
    anchor_gradients: GradientTuple
    non_anchor_gradients: GradientTuple
    final_gradients: GradientTuple
    anchor_gradient_norm: float
    non_anchor_gradient_norm: float
    final_gradient_norm: float
    accepted_non_anchor_count: int
    rejected_non_anchor_count: int
    non_anchor_records: Tuple[NonAnchorAlignmentRecord, ...]


def _validate_gradient_structure(gradients: Sequence[GradientTuple]) -> None:
    if not gradients:
        raise ValueError("At least one gradient tuple is required.")
    reference = gradients[0]
    if not reference:
        raise ValueError("Gradient tuples must contain at least one tensor.")
    for gradient_tuple in gradients:
        if len(gradient_tuple) != len(reference):
            raise ValueError("Gradient tuple length mismatch.")
        for reference_tensor, tensor in zip(reference, gradient_tuple):
            if tensor.shape != reference_tensor.shape:
                raise ValueError("Gradient tensor shape mismatch.")
            if tensor.dtype != reference_tensor.dtype:
                raise ValueError("Gradient tensor dtype mismatch.")
            if tensor.device != reference_tensor.device:
                raise ValueError("Gradient tensor device mismatch.")


def gradient_dot(left: GradientTuple, right: GradientTuple) -> float:
    _validate_gradient_structure((left, right))
    total = 0.0
    for left_tensor, right_tensor in zip(left, right):
        total += float(
            (left_tensor.to(dtype=torch.float64) * right_tensor.to(dtype=torch.float64)).sum().item()
        )
    return total


def gradient_norm(gradients: GradientTuple) -> float:
    squared_norm = gradient_dot(gradients, gradients)
    return math.sqrt(max(0.0, squared_norm))


def gradient_cosine(left: GradientTuple, right: GradientTuple, eps: float) -> float:
    denominator = gradient_norm(left) * gradient_norm(right) + float(eps)
    cosine = gradient_dot(left, right) / denominator
    return max(-1.0, min(1.0, float(cosine)))


def scale_gradients(gradients: GradientTuple, scale: float) -> GradientTuple:
    return tuple(tensor * float(scale) for tensor in gradients)


def mean_gradients(gradients: Sequence[GradientTuple]) -> GradientTuple:
    _validate_gradient_structure(gradients)
    accumulated = [torch.zeros_like(tensor) for tensor in gradients[0]]
    for gradient_tuple in gradients:
        for index, tensor in enumerate(gradient_tuple):
            accumulated[index].add_(tensor)
    denominator = float(len(gradients))
    return tuple(tensor.div_(denominator) for tensor in accumulated)


def select_anchor_contributions(
    contributions: Sequence[ClientGradientContribution],
    top_fraction: float,
    min_clients: int,
) -> tuple[Tuple[ClientGradientContribution, ...], Tuple[ClientGradientContribution, ...]]:
    if not contributions:
        raise ValueError("Cannot select anchors from an empty contribution list.")
    client_ids = [contribution.client_id for contribution in contributions]
    if len(client_ids) != len(set(client_ids)):
        raise ValueError("Client gradient contributions contain duplicate client IDs.")
    for contribution in contributions:
        if not math.isfinite(contribution.mean_quality):
            raise ValueError(f"Client {contribution.client_id} has non-finite mean quality.")
        if not (0.0 <= contribution.mean_quality <= 1.0):
            raise ValueError(
                f"Client {contribution.client_id} mean quality must be within [0, 1], "
                f"got {contribution.mean_quality}."
            )

    anchor_count = min(
        len(contributions),
        max(int(min_clients), int(math.ceil(float(top_fraction) * len(contributions)))),
    )
    ordered = sorted(
        contributions,
        key=lambda contribution: (-contribution.mean_quality, str(contribution.client_id)),
    )
    return tuple(ordered[:anchor_count]), tuple(ordered[anchor_count:])


def _zeros_like(gradients: GradientTuple) -> GradientTuple:
    return tuple(torch.zeros_like(tensor) for tensor in gradients)


def aggregate_anchor_gradients(
    contributions: Sequence[ClientGradientContribution],
    config: QualityAnchorConfig,
) -> AnchorAggregationResult:
    anchor_contributions, non_anchor_contributions = select_anchor_contributions(
        contributions,
        config.top_fraction,
        config.min_clients,
    )
    anchor_gradients = mean_gradients(
        [contribution.server_gradients for contribution in anchor_contributions]
    )
    anchor_norm = gradient_norm(anchor_gradients)
    zero_non_anchor = _zeros_like(anchor_gradients)

    if not config.use_gradient_alignment_filter:
        records = tuple(
            NonAnchorAlignmentRecord(
                client_id=contribution.client_id,
                mean_quality=contribution.mean_quality,
                gradient_norm=contribution.server_gradient_norm,
                cosine_with_anchor=None,
                alignment_weight=0.0,
                norm_clip_scale=0.0,
                accepted=False,
                reason="alignment_filter_disabled",
            )
            for contribution in non_anchor_contributions
        )
        return AnchorAggregationResult(
            anchor_client_ids=tuple(
                contribution.client_id for contribution in anchor_contributions
            ),
            anchor_qualities=tuple(
                contribution.mean_quality for contribution in anchor_contributions
            ),
            non_anchor_client_ids=tuple(
                contribution.client_id for contribution in non_anchor_contributions
            ),
            non_anchor_qualities=tuple(
                contribution.mean_quality for contribution in non_anchor_contributions
            ),
            anchor_gradients=anchor_gradients,
            non_anchor_gradients=zero_non_anchor,
            final_gradients=anchor_gradients,
            anchor_gradient_norm=anchor_norm,
            non_anchor_gradient_norm=0.0,
            final_gradient_norm=anchor_norm,
            accepted_non_anchor_count=0,
            rejected_non_anchor_count=len(non_anchor_contributions),
            non_anchor_records=records,
        )

    weighted_sum = [torch.zeros_like(tensor) for tensor in anchor_gradients]
    alignment_weight_sum = 0.0
    records_list: list[NonAnchorAlignmentRecord] = []
    accepted_count = 0

    for contribution in non_anchor_contributions:
        cosine = gradient_cosine(
            contribution.server_gradients,
            anchor_gradients,
            config.eps,
        )
        alignment_weight = (
            float(cosine ** config.alignment_power)
            if cosine > 0.0
            else float(config.negative_alignment_weight)
        )
        clip_scale = min(
            1.0,
            (config.low_client_norm_cap * anchor_norm)
            / (contribution.server_gradient_norm + config.eps),
        )
        clipped_gradients = scale_gradients(contribution.server_gradients, clip_scale)
        accepted = alignment_weight > 0.0
        if accepted:
            accepted_count += 1
            alignment_weight_sum += alignment_weight
            for index, tensor in enumerate(clipped_gradients):
                weighted_sum[index].add_(tensor, alpha=alignment_weight)
        records_list.append(
            NonAnchorAlignmentRecord(
                client_id=contribution.client_id,
                mean_quality=contribution.mean_quality,
                gradient_norm=contribution.server_gradient_norm,
                cosine_with_anchor=cosine,
                alignment_weight=alignment_weight,
                norm_clip_scale=clip_scale,
                accepted=accepted,
                reason=(
                    "positive_alignment"
                    if cosine > 0.0
                    else "configured_negative_alignment"
                    if accepted
                    else "non_positive_alignment"
                ),
            )
        )

    if accepted_count == 0:
        non_anchor_gradients = zero_non_anchor
        final_gradients = anchor_gradients
    else:
        denominator = alignment_weight_sum + config.eps
        non_anchor_gradients = tuple(tensor.div_(denominator) for tensor in weighted_sum)
        final_gradients = tuple(
            config.anchor_update_share * anchor_tensor
            + (1.0 - config.anchor_update_share) * non_anchor_tensor
            for anchor_tensor, non_anchor_tensor in zip(
                anchor_gradients,
                non_anchor_gradients,
            )
        )

    non_anchor_norm = gradient_norm(non_anchor_gradients)
    final_norm = gradient_norm(final_gradients)
    return AnchorAggregationResult(
        anchor_client_ids=tuple(
            contribution.client_id for contribution in anchor_contributions
        ),
        anchor_qualities=tuple(
            contribution.mean_quality for contribution in anchor_contributions
        ),
        non_anchor_client_ids=tuple(
            contribution.client_id for contribution in non_anchor_contributions
        ),
        non_anchor_qualities=tuple(
            contribution.mean_quality for contribution in non_anchor_contributions
        ),
        anchor_gradients=anchor_gradients,
        non_anchor_gradients=non_anchor_gradients,
        final_gradients=final_gradients,
        anchor_gradient_norm=anchor_norm,
        non_anchor_gradient_norm=non_anchor_norm,
        final_gradient_norm=final_norm,
        accepted_non_anchor_count=accepted_count,
        rejected_non_anchor_count=len(non_anchor_contributions) - accepted_count,
        non_anchor_records=tuple(records_list),
    )
