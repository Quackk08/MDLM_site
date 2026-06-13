import torch

from mdlm_quant.quantization import fake_quantize_symmetric, quantization_mse


def test_fake_quantizer_output_range_and_monotonic_error():
    torch.manual_seed(0)
    x = torch.randn(512) * 0.7
    errors = {}
    max_abs = float(x.abs().max())
    for bits in (3, 4, 6, 8):
        q, _ = fake_quantize_symmetric(x, bits)
        assert float(q.abs().max()) <= max_abs + 1e-6
        errors[bits] = quantization_mse(x, bits)
    assert errors[3] >= errors[4] >= errors[6] >= errors[8]


def test_fake_quantizer_rejects_invalid_bits():
    x = torch.ones(4)
    for bits in (2, 9):
        try:
            fake_quantize_symmetric(x, bits)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid bit width was accepted")

