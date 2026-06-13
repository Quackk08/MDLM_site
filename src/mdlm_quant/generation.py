"""Deterministic multi-step denoising and trajectory-based evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import MaskedDiffusionTransformer
from .tokenizer import ByteTokenizer


@dataclass
class TrajectoryStep:
    """One deterministic reference denoising step."""

    seed: int
    sample_index: int
    step: int
    current_ids: torch.Tensor
    reveal_mask: torch.Tensor
    labels: torch.Tensor
    reference_logits: torch.Tensor
    reference_prediction: torch.Tensor


def _reveal_mask_for_step(masked_positions: torch.Tensor, confidences: torch.Tensor, remaining_steps: int) -> torch.Tensor:
    count = int(masked_positions.sum().item())
    reveal_count = max(1, (count + remaining_steps - 1) // remaining_steps)
    flat_positions = torch.nonzero(masked_positions, as_tuple=False).flatten()
    scores = confidences[flat_positions]
    order = torch.argsort(scores, descending=True, stable=True)
    chosen = flat_positions[order[:reveal_count]]
    reveal = torch.zeros_like(masked_positions)
    reveal[chosen] = True
    return reveal


@torch.no_grad()
def deterministic_unmask(
    model: MaskedDiffusionTransformer,
    initial_ids: torch.Tensor,
    tokenizer: ByteTokenizer,
    timesteps: int,
    device: torch.device,
    bit_schedule: dict[int, int] | None = None,
) -> torch.Tensor:
    """Generate a completed sequence with deterministic argmax unmasking."""

    model.eval()
    current = initial_ids.clone().to(device)
    for step in range(timesteps, 0, -1):
        masked_positions = current.eq(tokenizer.mask_token_id)
        if not bool(masked_positions.any()):
            break
        logits, _ = model(current.unsqueeze(0), torch.tensor([step], device=device), bit_schedule=bit_schedule)
        probs = torch.softmax(logits[0], dim=-1)
        confidence, pred = probs.max(dim=-1)
        reveal = _reveal_mask_for_step(masked_positions, confidence.detach().cpu(), step).to(device)
        current[reveal] = pred[reveal]
    return current.detach().cpu()


@torch.no_grad()
def build_reference_trajectories(
    model: MaskedDiffusionTransformer,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: ByteTokenizer,
    timesteps: int,
    seed: int,
    device: torch.device,
) -> list[TrajectoryStep]:
    """Build a reference trajectory used by every compared schedule."""

    model.eval()
    trajectories: list[TrajectoryStep] = []
    for sample_index in range(tokens.size(0)):
        current = tokens[sample_index].clone().to(device)
        target = labels[sample_index].clone().to(device)
        for step in range(timesteps, 0, -1):
            masked_positions = current.eq(tokenizer.mask_token_id)
            if not bool(masked_positions.any()):
                break
            logits, _ = model(current.unsqueeze(0), torch.tensor([step], device=device))
            logits_row = logits[0].detach().cpu()
            probs = torch.softmax(logits_row, dim=-1)
            confidence, pred_cpu = probs.max(dim=-1)
            reveal = _reveal_mask_for_step(masked_positions.detach().cpu(), confidence, step)
            trajectories.append(
                TrajectoryStep(
                    seed=seed,
                    sample_index=sample_index,
                    step=step,
                    current_ids=current.detach().cpu().clone(),
                    reveal_mask=reveal.clone(),
                    labels=target.detach().cpu().clone(),
                    reference_logits=logits_row,
                    reference_prediction=pred_cpu.clone(),
                )
            )
            current[reveal.to(device)] = pred_cpu.to(device)[reveal.to(device)]
    return trajectories


@torch.no_grad()
def evaluate_schedule_on_trajectories(
    model: MaskedDiffusionTransformer,
    trajectories: list[TrajectoryStep],
    tokenizer: ByteTokenizer,
    device: torch.device,
    method: str,
    bit_schedule: dict[int, int] | None,
    predicted_bound: float,
    avg_bits: float,
    budget: int,
) -> list[dict[str, float | int | str]]:
    """Evaluate a schedule on fixed reference trajectories."""

    del tokenizer
    model.eval()
    rows: list[dict[str, float | int | str]] = []
    grouped: dict[tuple[int, int], list[TrajectoryStep]] = {}
    for item in trajectories:
        grouped.setdefault((item.seed, item.sample_index), []).append(item)
    for (seed, sample_index), steps in grouped.items():
        token_correct = 0
        token_total = 0
        ce_sum = 0.0
        kl_sum = 0.0
        logit_mse_sum = 0.0
        agreement = 0
        final_predictions: dict[int, int] = {}
        for item in steps:
            input_ids = item.current_ids.unsqueeze(0).to(device)
            logits, _ = model(input_ids, torch.tensor([item.step], device=device), bit_schedule=bit_schedule)
            logits_cpu = logits[0].detach().cpu()
            reveal = item.reveal_mask
            if not bool(reveal.any()):
                continue
            chosen_logits = logits_cpu[reveal]
            labels = item.labels[reveal]
            valid = labels.ne(-100)
            if not bool(valid.any()):
                continue
            chosen_logits = chosen_logits[valid]
            labels = labels[valid]
            ref_logits = item.reference_logits[reveal][valid]
            pred = chosen_logits.argmax(dim=-1)
            ref_pred = ref_logits.argmax(dim=-1)
            ce = F.cross_entropy(chosen_logits, labels, reduction="sum")
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            log_probs = F.log_softmax(chosen_logits, dim=-1)
            ref_probs = ref_log_probs.exp()
            kl = F.kl_div(log_probs, ref_probs, reduction="batchmean", log_target=False) * labels.numel()
            mse = torch.mean((chosen_logits - ref_logits) ** 2) * labels.numel()
            ce_sum += float(ce)
            kl_sum += float(kl)
            logit_mse_sum += float(mse)
            token_correct += int(pred.eq(labels).sum())
            agreement += int(pred.eq(ref_pred).sum())
            reveal_positions = torch.nonzero(reveal, as_tuple=False).flatten()[valid]
            for pos, token in zip(reveal_positions.tolist(), pred.tolist()):
                final_predictions[int(pos)] = int(token)
            token_total += int(labels.numel())
        if token_total == 0:
            continue
        exact = 1.0
        for item in steps:
            initial_mask = item.labels.ne(-100)
            for pos in torch.nonzero(initial_mask, as_tuple=False).flatten().tolist():
                if final_predictions.get(int(pos), -1) != int(item.labels[pos]):
                    exact = 0.0
                    break
            break
        rows.append(
            {
                "method": method,
                "seed": seed,
                "sample_index": sample_index,
                "masked_cross_entropy": ce_sum / token_total,
                "masked_token_accuracy": token_correct / token_total,
                "sequence_reconstruction_accuracy": exact,
                "reference_token_agreement": agreement / token_total,
                "kl_divergence_from_reference": kl_sum / token_total,
                "final_logit_mse": logit_mse_sum / token_total,
                "predicted_gronwall_objective_p2": predicted_bound,
                "average_bit_width": avg_bits,
                "total_bit_budget": budget,
                "theoretical_activation_memory_ratio": avg_bits / 16.0,
            }
        )
    return rows
