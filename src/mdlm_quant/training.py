"""Training and checkpointing for the masked denoiser."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import TextSplits, mask_batch, sample_batch
from .model import MaskedDiffusionTransformer
from .tokenizer import ByteTokenizer
from .utils import write_csv


def checkpoint_path(config: ExperimentConfig) -> Path:
    """Return the checkpoint path for a configuration."""

    return Path(config.output.root) / "checkpoints" / config.output.checkpoint_name


def save_checkpoint(
    path: Path,
    model: MaskedDiffusionTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: list[dict[str, float]],
) -> None:
    """Save a resumable training checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "metrics": metrics}, path)


def load_checkpoint(
    path: Path,
    model: MaskedDiffusionTransformer,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, list[dict[str, float]]]:
    """Load a checkpoint if it exists."""

    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0)), list(payload.get("metrics", []))


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy only on masked positions."""

    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)


def train_model(
    model: MaskedDiffusionTransformer,
    splits: TextSplits,
    tokenizer: ByteTokenizer,
    config: ExperimentConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    """Train or resume the denoiser."""

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
    ckpt = checkpoint_path(config)
    start_step = 0
    metrics: list[dict[str, float]] = []
    if config.train.resume and ckpt.exists():
        start_step, metrics = load_checkpoint(ckpt, model, optimizer, map_location=device)
        if start_step >= config.train.train_steps:
            return metrics

    generator = torch.Generator(device="cpu").manual_seed(config.train.seed + start_step)
    model.train()
    for step in range(start_step + 1, config.train.train_steps + 1):
        batch = sample_batch(splits.train, config.train.batch_size, generator).to(device)
        masked, labels = mask_batch(
            batch.cpu(),
            tokenizer,
            config.train.mask_ratio_min,
            config.train.mask_ratio_max,
            generator,
        )
        masked = masked.to(device)
        labels = labels.to(device)
        mask_counts = labels.ne(-100).sum(dim=1).float()
        eligible_counts = batch.cpu().ne(tokenizer.pad_token_id).sum(dim=1).float().clamp_min(1)
        ratios = (mask_counts.cpu() / eligible_counts).clamp(0, 1)
        timesteps = torch.clamp(torch.ceil(ratios * config.model.timesteps), min=1, max=config.model.timesteps).long().to(device)
        logits, _ = model(masked, timesteps)
        loss = masked_cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics.append({"step": float(step), "loss": float(loss.detach().cpu())})
        if step % config.train.checkpoint_every == 0:
            save_checkpoint(ckpt, model, optimizer, step, metrics)

    save_checkpoint(ckpt, model, optimizer, config.train.train_steps, metrics)
    write_csv(Path(config.output.root) / "csv" / "training_log.csv", metrics)
    return metrics

