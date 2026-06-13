"""Compact encoder-style Transformer denoiser."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ModelConfig
from .quantization import QuantizationRecord, fake_quantize_symmetric


@dataclass
class ForwardDiagnostics:
    """Optional diagnostics captured during a forward pass."""

    quant_records: list[QuantizationRecord]
    final_hidden: torch.Tensor | None = None


class TransformerBlock(nn.Module):
    """A pre-norm Transformer encoder block."""

    def __init__(self, hidden_size: int, num_heads: int, ff_size: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Linear(ff_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply self-attention and feed-forward transformations."""

        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x


class MaskedDiffusionTransformer(nn.Module):
    """Small masked diffusion language model with timestep conditioning."""

    def __init__(self, config: ModelConfig, vocab_size: int, pad_token_id: int) -> None:
        super().__init__()
        self.config = config
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, config.hidden_size, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.timestep_embedding = nn.Embedding(config.timesteps + 1, config.hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config.hidden_size, config.num_heads, config.ff_size, config.dropout) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        bit_schedule: dict[int, int] | None = None,
        capture_diagnostics: bool = False,
        input_hidden_perturbation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ForwardDiagnostics | None]:
        """Run the denoiser and optionally fake-quantize activations per timestep."""

        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("input sequence length exceeds configured maximum")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        safe_steps = timesteps.clamp(min=0, max=self.config.timesteps)
        x = self.token_embedding(input_ids) + self.position_embedding(positions) + self.timestep_embedding(safe_steps).unsqueeze(1)
        if input_hidden_perturbation is not None:
            x = x + input_hidden_perturbation
        key_padding_mask = input_ids.eq(self.pad_token_id)
        records: list[QuantizationRecord] = []
        step_value = int(safe_steps[0].detach().cpu()) if safe_steps.numel() else 0
        bits = bit_schedule.get(step_value) if bit_schedule is not None else None
        for block_index, block in enumerate(self.blocks):
            x = block(x, key_padding_mask=key_padding_mask)
            if bits is not None:
                before = x
                x, scale = fake_quantize_symmetric(x, int(bits))
                if capture_diagnostics:
                    mse = torch.mean((x - before) ** 2).detach().cpu().item()
                    records.append(QuantizationRecord(step=step_value, bits=int(bits), block=block_index, mse=float(mse), scale=float(scale.detach().cpu())))
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)
        diagnostics = ForwardDiagnostics(quant_records=records, final_hidden=hidden.detach() if capture_diagnostics else None)
        return logits, diagnostics if capture_diagnostics else None

