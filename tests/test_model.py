import torch

from mdlm_quant.config import ModelConfig
from mdlm_quant.model import MaskedDiffusionTransformer
from mdlm_quant.tokenizer import ByteTokenizer


def test_model_input_output_shapes():
    tokenizer = ByteTokenizer()
    cfg = ModelConfig(num_layers=1, hidden_size=32, num_heads=4, ff_size=64, max_seq_len=16, timesteps=3)
    model = MaskedDiffusionTransformer(cfg, tokenizer.vocab_size, tokenizer.pad_token_id)
    input_ids = torch.randint(4, tokenizer.vocab_size, (2, 16))
    input_ids[:, 0] = tokenizer.bos_token_id
    timesteps = torch.tensor([1, 3])
    logits, diagnostics = model(input_ids, timesteps, bit_schedule={1: 4}, capture_diagnostics=True)
    assert logits.shape == (2, 16, tokenizer.vocab_size)
    assert diagnostics is not None
    assert diagnostics.final_hidden is not None

