"""Experiment configuration definitions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Transformer denoiser hyperparameters."""

    num_layers: int = 2
    hidden_size: int = 192
    num_heads: int = 4
    ff_size: int = 768
    max_seq_len: int = 64
    dropout: float = 0.0
    timesteps: int = 8


@dataclass
class DataConfig:
    """Dataset and split settings."""

    source: str = "builtin"
    train_size: int = 96
    calibration_size: int = 24
    validation_size: int = 24
    test_size: int = 24
    eval_size: int = 16
    num_workers: int = 0


@dataclass
class TrainConfig:
    """Training settings."""

    seed: int = 1234
    batch_size: int = 8
    train_steps: int = 12
    learning_rate: float = 3e-4
    mask_ratio_min: float = 0.15
    mask_ratio_max: float = 0.75
    checkpoint_every: int = 50
    resume: bool = True


@dataclass
class CalibrationConfig:
    """Calibration, schedule, and evaluation settings."""

    bits_for_fit: tuple[int, ...] = (3, 4, 6, 8)
    allowed_bits: tuple[int, ...] = (4, 6, 8)
    target_avg_bits: int = 6
    perturbation_scale: float = 1e-3
    amplification_batches: int = 3
    quantization_batches: int = 3
    eval_seeds: tuple[int, ...] = (11, 17, 23, 31, 43)
    bootstrap_samples: int = 300
    oracle_max_candidates: int = 24


@dataclass
class OutputConfig:
    """Output paths."""

    root: str = "outputs/quick"
    checkpoint_name: str = "checkpoint.pt"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    mode: str = "quick"
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""

        return asdict(self)

    def save(self, path: Path) -> None:
        """Save configuration as JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def _research_config() -> ExperimentConfig:
    cfg = ExperimentConfig(mode="research")
    cfg.model = ModelConfig(num_layers=4, hidden_size=256, num_heads=4, ff_size=1024, max_seq_len=64, dropout=0.0, timesteps=12)
    cfg.data = DataConfig(source="tinystories", train_size=6000, calibration_size=512, validation_size=512, test_size=512, eval_size=128)
    cfg.train = TrainConfig(seed=2026, batch_size=32, train_steps=1500, learning_rate=3e-4, checkpoint_every=250, resume=True)
    cfg.calibration = CalibrationConfig(
        eval_seeds=(11, 17, 23, 31, 43),
        amplification_batches=8,
        quantization_batches=8,
        bootstrap_samples=1000,
        oracle_max_candidates=80,
    )
    cfg.output = OutputConfig(root="outputs/research")
    return cfg


def _quick_config() -> ExperimentConfig:
    cfg = ExperimentConfig(mode="quick")
    cfg.model = ModelConfig(num_layers=2, hidden_size=192, num_heads=4, ff_size=768, max_seq_len=64, dropout=0.0, timesteps=8)
    cfg.data = DataConfig(source="builtin", train_size=96, calibration_size=24, validation_size=24, test_size=24, eval_size=8)
    cfg.train = TrainConfig(seed=1234, batch_size=8, train_steps=12, learning_rate=5e-4, checkpoint_every=50, resume=True)
    cfg.calibration = CalibrationConfig(
        eval_seeds=(11, 17, 23, 31, 43),
        amplification_batches=2,
        quantization_batches=2,
        bootstrap_samples=200,
        oracle_max_candidates=18,
    )
    cfg.output = OutputConfig(root="outputs/quick")
    return cfg


def get_config(mode: str) -> ExperimentConfig:
    """Return a named default configuration."""

    normalized = mode.lower().strip()
    if normalized == "quick":
        return _quick_config()
    if normalized == "research":
        return _research_config()
    raise ValueError(f"unknown mode: {mode}")


def load_config(path: str | Path) -> ExperimentConfig:
    """Load configuration from a JSON file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = ExperimentConfig()
    cfg.mode = data.get("mode", cfg.mode)
    cfg.device = data.get("device", cfg.device)
    cfg.model = ModelConfig(**data.get("model", {}))
    cfg.data = DataConfig(**data.get("data", {}))
    cfg.train = TrainConfig(**data.get("train", {}))
    cfg.calibration = CalibrationConfig(**data.get("calibration", {}))
    cfg.output = OutputConfig(**data.get("output", {}))
    return cfg
