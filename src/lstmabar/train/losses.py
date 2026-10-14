"""P6 losses (design §6). Stage 1: ``L_param`` = masked L1 on knobs + gate BCE.

Knob errors are masked by the **label's** gates (a disabled block's knobs had no effect on the
audio, ``RenderItem`` docstring); per-item weights (clamped grey-box items, P4 decision 10 /
R6) scale both terms.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def masked_l1(pred: Tensor, target: Tensor, mask: Tensor, weights: Tensor | None = None) -> Tensor:
    """Weighted mean of ``|pred - target|`` over the entries where ``mask`` is 1:
    ``sum(w_i m_ik |e_ik|) / sum(w_i m_ik)`` (0 when nothing is unmasked)."""
    w = mask if weights is None else mask * weights[:, None]
    denom = w.sum()
    if denom <= 0:
        return pred.sum() * 0.0
    return (w * (pred - target).abs()).sum() / denom


def gate_bce(logits: Tensor, target: Tensor, weights: Tensor | None = None) -> Tensor:
    """Weighted mean BCE over items and blocks."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if weights is None:
        return bce.mean()
    return (bce * weights[:, None]).sum() / (weights.sum() * bce.shape[1]).clamp_min(1e-12)


def param_loss(
    knob_pred: Tensor,
    gate_logits: Tensor,
    knob_target: Tensor,
    gate_target: Tensor,
    knob_mask: Tensor,
    weights: Tensor | None = None,
    param_weight: float = 1.0,
    gate_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """``L_param``: ``param_weight * masked_l1 + gate_weight * gate_bce``. ``knob_pred`` and
    ``knob_target`` must be in the same space (normalized for the physics mapping, linear
    physical for the unconstrained one). Returns the loss and its float parts."""
    l1 = masked_l1(knob_pred, knob_target, knob_mask, weights)
    bce = gate_bce(gate_logits, gate_target, weights)
    total = param_weight * l1 + gate_weight * bce
    return total, {"loss": total.item(), "param_l1": l1.item(), "gate_bce": bce.item()}


__all__ = ["gate_bce", "masked_l1", "param_loss"]
