"""Calibration of quantization error and local amplification."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .config import ExperimentConfig
from .data import TextSplits, iter_batches, mask_batch
from .model import MaskedDiffusionTransformer
from .tokenizer import ByteTokenizer


@dataclass
class CalibrationResult:
    """All scalar calibration outputs needed for schedules and reports."""

    quant_rows: list[dict[str, float | int]]
    fit_rows: list[dict[str, float | int]]
    amplification_raw_rows: list[dict[str, float | int]]
    amplification_summary_rows: list[dict[str, float | int]]
    gronwall_rows: list[dict[str, float | int]]
    theoretical_c: dict[int, float]
    fitted_c: dict[int, float]
    fitted_p: dict[int, float]
    mean_l: dict[int, float]
    g: dict[int, float]


def _fit_power_law(bits: list[int], eps: list[float]) -> tuple[float, float, float]:
    x = np.asarray(bits, dtype=np.float64)
    y = np.log2(np.asarray(eps, dtype=np.float64).clip(1e-30, None))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    p = -float(slope)
    c = float(2.0**intercept)
    return c, p, float(r2)


def _theoretical_c_from_bits(bits: list[int], eps: list[float]) -> float:
    values = [float(e) * (2.0 ** (2 * int(b))) for b, e in zip(bits, eps)]
    return float(np.median(values))


@torch.no_grad()
def estimate_quantization_error(
    model: MaskedDiffusionTransformer,
    splits: TextSplits,
    tokenizer: ByteTokenizer,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]], dict[int, float], dict[int, float], dict[int, float]]:
    """Estimate per-timestep activation quantization MSE and fit power laws."""

    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed + 500)
    rows: list[dict[str, float | int]] = []
    for batch_index, batch in enumerate(iter_batches(splits.calibration, config.train.batch_size, config.calibration.quantization_batches)):
        masked, _ = mask_batch(batch, tokenizer, config.train.mask_ratio_min, config.train.mask_ratio_max, generator)
        masked = masked.to(device)
        for step in range(1, config.model.timesteps + 1):
            timesteps = torch.full((masked.size(0),), step, dtype=torch.long, device=device)
            for bits in config.calibration.bits_for_fit:
                _, diagnostics = model(masked, timesteps, bit_schedule={step: int(bits)}, capture_diagnostics=True)
                assert diagnostics is not None
                if not diagnostics.quant_records:
                    mse = 0.0
                else:
                    mse = float(np.mean([record.mse for record in diagnostics.quant_records]))
                rows.append({"batch": batch_index, "step": step, "bits": int(bits), "activation_mse": mse})

    fit_rows: list[dict[str, float | int]] = []
    theoretical_c: dict[int, float] = {}
    fitted_c: dict[int, float] = {}
    fitted_p: dict[int, float] = {}
    for step in range(1, config.model.timesteps + 1):
        bits: list[int] = []
        eps: list[float] = []
        for bit in config.calibration.bits_for_fit:
            values = [float(row["activation_mse"]) for row in rows if int(row["step"]) == step and int(row["bits"]) == bit]
            bits.append(int(bit))
            eps.append(float(np.mean(values)) if values else 0.0)
        c_fit, p_fit, r2 = _fit_power_law(bits, eps)
        c_theory = _theoretical_c_from_bits(bits, eps)
        theoretical_c[step] = c_theory
        fitted_c[step] = c_fit
        fitted_p[step] = p_fit
        fit_rows.append(
            {
                "step": step,
                "C_theoretical_p2": c_theory,
                "C_fitted": c_fit,
                "p_fitted": p_fit,
                "r_squared": r2,
                "materially_differs_from_2": float(abs(p_fit - 2.0) > 0.25),
            }
        )
    return rows, fit_rows, theoretical_c, fitted_c, fitted_p


@torch.no_grad()
def estimate_amplification(
    model: MaskedDiffusionTransformer,
    splits: TextSplits,
    tokenizer: ByteTokenizer,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]], dict[int, float]]:
    """Estimate finite-perturbation local amplification ratios."""

    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed + 900)
    raw_rows: list[dict[str, float | int]] = []
    scale = config.calibration.perturbation_scale
    for batch_index, batch in enumerate(iter_batches(splits.calibration, config.train.batch_size, config.calibration.amplification_batches)):
        masked, _ = mask_batch(batch, tokenizer, config.train.mask_ratio_min, config.train.mask_ratio_max, generator)
        masked = masked.to(device)
        for step in range(1, config.model.timesteps + 1):
            timesteps = torch.full((masked.size(0),), step, dtype=torch.long, device=device)
            base_logits, _ = model(masked, timesteps)
            embedding_shape = (masked.size(0), masked.size(1), config.model.hidden_size)
            perturb = torch.randn(embedding_shape, generator=generator, device="cpu").to(device)
            perturb = perturb / perturb.pow(2).mean().sqrt().clamp_min(1e-12) * scale
            perturbed_logits, _ = model(masked, timesteps, input_hidden_perturbation=perturb)
            numerator = torch.mean((perturbed_logits - base_logits) ** 2).detach().cpu().item()
            denominator = torch.mean(perturb**2).detach().cpu().item()
            ratio = float(numerator / max(denominator, 1e-30))
            if not math.isfinite(ratio):
                ratio = 0.0
            raw_rows.append({"batch": batch_index, "step": step, "amplification": ratio})

    summary_rows: list[dict[str, float | int]] = []
    mean_l: dict[int, float] = {}
    for step in range(1, config.model.timesteps + 1):
        values = np.asarray([float(row["amplification"]) for row in raw_rows if int(row["step"]) == step], dtype=np.float64)
        if values.size == 0:
            values = np.asarray([0.0], dtype=np.float64)
        mean_value = float(np.mean(values))
        mean_l[step] = max(mean_value, 1e-12)
        summary_rows.append(
            {
                "step": step,
                "L_mean": mean_value,
                "L_median": float(np.median(values)),
                "L_p95": float(np.percentile(values, 95)),
                "num_estimates": int(values.size),
            }
        )
    return raw_rows, summary_rows, mean_l


def cumulative_gronwall(mean_l: dict[int, float], timesteps: int) -> tuple[dict[int, float], list[dict[str, float | int]]]:
    """Calculate cumulative amplification G_k in log space."""

    log_g: dict[int, float] = {}
    running = 0.0
    for step in range(timesteps, 0, -1):
        log_g[step] = running
        running += math.log(max(mean_l.get(step, 1.0), 1e-12))
    rows: list[dict[str, float | int]] = []
    g: dict[int, float] = {}
    for step in range(1, timesteps + 1):
        clipped_log = max(min(log_g[step], 60.0), -60.0)
        g_value = float(math.exp(clipped_log))
        g[step] = g_value
        rows.append({"step": step, "log_G": log_g[step], "G": g_value, "L_mean": mean_l.get(step, 1.0)})
    return g, rows


def run_calibration(
    model: MaskedDiffusionTransformer,
    splits: TextSplits,
    tokenizer: ByteTokenizer,
    config: ExperimentConfig,
    device: torch.device,
) -> CalibrationResult:
    """Run all calibration routines."""

    quant_rows, fit_rows, theoretical_c, fitted_c, fitted_p = estimate_quantization_error(model, splits, tokenizer, config, device)
    amp_raw, amp_summary, mean_l = estimate_amplification(model, splits, tokenizer, config, device)
    g, gronwall_rows = cumulative_gronwall(mean_l, config.model.timesteps)
    return CalibrationResult(
        quant_rows=quant_rows,
        fit_rows=fit_rows,
        amplification_raw_rows=amp_raw,
        amplification_summary_rows=amp_summary,
        gronwall_rows=gronwall_rows,
        theoretical_c=theoretical_c,
        fitted_c=fitted_c,
        fitted_p=fitted_p,
        mean_l=mean_l,
        g=g,
    )

