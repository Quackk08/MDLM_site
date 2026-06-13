import torch

from mdlm_quant.config import ModelConfig
from mdlm_quant.generation import deterministic_unmask
from mdlm_quant.model import MaskedDiffusionTransformer
from mdlm_quant.tokenizer import ByteTokenizer
from mdlm_quant.utils import set_seed


def test_deterministic_inference_under_fixed_seed():
    set_seed(42)
    tokenizer = ByteTokenizer()
    cfg = ModelConfig(num_layers=1, hidden_size=32, num_heads=4, ff_size=64, max_seq_len=16, timesteps=3)
    model = MaskedDiffusionTransformer(cfg, tokenizer.vocab_size, tokenizer.pad_token_id)
    initial = torch.tensor(tokenizer.encode("a tiny test", 16))
    initial[3:8] = tokenizer.mask_token_id
    out1 = deterministic_unmask(model, initial, tokenizer, cfg.timesteps, torch.device("cpu"), bit_schedule={1: 4, 2: 4, 3: 4})
    out2 = deterministic_unmask(model, initial, tokenizer, cfg.timesteps, torch.device("cpu"), bit_schedule={1: 4, 2: 4, 3: 4})
    assert torch.equal(out1, out2)

