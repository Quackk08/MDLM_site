"""Symmetric per-tensor activation fake quantization."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QuantizationRecord:
    """Activation quantization diagnostic values."""

    step: int
    bits: int
    block: int
    mse: float
    scale: float


def fake_quantize_symmetric(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake-quantize a tensor with signed symmetric integer levels."""

    if bits < 3 or bits > 8:
        raise ValueError("activation fake quantization supports integer bit widths from 3 to 8")
    if not x.is_floating_point():
        raise TypeError("fake quantization expects a floating-point tensor")
    qmax = (2 ** (bits - 1)) - 1
    max_abs = x.detach().abs().amax()
    if float(max_abs) == 0.0:
        return x.clone(), torch.tensor(1.0, device=x.device, dtype=x.dtype)
    scale = max_abs / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q * scale, scale


def quantization_mse(x: torch.Tensor, bits: int) -> float:
    """Return fake-quantization mean-squared error."""

    q, _ = fake_quantize_symmetric(x, bits)
    return float(torch.mean((q - x) ** 2).detach().cpu())

