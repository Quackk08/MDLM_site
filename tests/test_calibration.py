import math

from mdlm_quant.calibration import run_calibration
from mdlm_quant.config import CalibrationConfig, DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from mdlm_quant.data import tokenize_splits
from mdlm_quant.model import MaskedDiffusionTransformer
from mdlm_quant.tokenizer import ByteTokenizer
from mdlm_quant.utils import set_seed


def test_no_nan_or_infinity_in_sensitivity_calculations():
    set_seed(7)
    tokenizer = ByteTokenizer()
    cfg = ExperimentConfig(
        mode="test",
        device="cpu",
        model=ModelConfig(num_layers=1, hidden_size=32, num_heads=4, ff_size=64, max_seq_len=16, timesteps=3),
        data=DataConfig(source="builtin", train_size=8, calibration_size=6, validation_size=4, test_size=4, eval_size=2),
        train=TrainConfig(seed=7, batch_size=2, train_steps=1),
        calibration=CalibrationConfig(
            bits_for_fit=(3, 4, 6, 8),
            allowed_bits=(4, 6, 8),
            target_avg_bits=6,
            amplification_batches=1,
            quantization_batches=1,
            eval_seeds=(1, 2, 3, 4, 5),
            bootstrap_samples=10,
            oracle_max_candidates=4,
        ),
    )
    splits = tokenize_splits(cfg.data, tokenizer, cfg.model.max_seq_len, cfg.train.seed)
    model = MaskedDiffusionTransformer(cfg.model, tokenizer.vocab_size, tokenizer.pad_token_id)
    result = run_calibration(model, splits, tokenizer, cfg, device=__import__("torch").device("cpu"))
    numeric_values = []
    for rows in (result.fit_rows, result.amplification_summary_rows, result.gronwall_rows):
        for row in rows:
            numeric_values.extend(float(value) for value in row.values() if isinstance(value, (int, float)))
    assert numeric_values
    assert all(math.isfinite(value) for value in numeric_values)

