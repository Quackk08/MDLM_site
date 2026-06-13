"""Text loading, splitting, and random masking utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from .config import DataConfig
from .tokenizer import ByteTokenizer


BUILTIN_STORIES: tuple[str, ...] = (
    "Lina found a red cup under the old table. She washed it, filled it with water, and gave it to a thirsty flower.",
    "A small robot learned to say please. Every time it helped, the children clapped and the robot blinked its blue light.",
    "Ben lost his kite near the hill. The wind pushed it into a tree, so his sister used a long stick to bring it down.",
    "Mira baked tiny cakes for her toy bears. One cake fell, but she laughed and made a smaller cake for the smallest bear.",
    "The dog waited by the gate until noon. When Sam came home, the dog jumped twice and carried his shoe inside.",
    "A rainy morning made the street shine. Noor counted seven puddles and stepped around each one on the way to school.",
    "The moon looked like a silver button. Dad said the sky was a coat, and June imagined stars stitched across it.",
    "Oscar planted three beans in a blue pot. After many sunny days, one green sprout curled up like a question mark.",
    "The library cat slept on a map. When the map moved, the cat woke up and chose a new island for a nap.",
    "Tess built a tower from blocks. It leaned to the left, so she added a yellow block and made a bridge instead.",
    "A little train carried apples to town. At every stop, the driver waved and counted the baskets again.",
    "Ivy heard a bell in the garden. It was only a spoon in a glass, but it sounded like a tiny song.",
    "The snowman wore a green scarf. By sunset he was smaller, yet the scarf stayed bright on the white ground.",
    "Leo made a paper boat and set it in a bowl. The boat sailed around a spoon and reached the carrot island.",
    "Nina drew a door on cardboard. Behind the door she drew a forest, a path, and a house with warm windows.",
    "A sleepy dragon wanted soup, not treasure. The village cook made carrot soup, and the dragon warmed the kitchen.",
    "Pip the mouse found a crumb shaped like a star. He shared it with two friends under the cupboard.",
    "The class grew sunflowers in paper cups. Each child measured a stem and wrote the number beside a picture.",
    "A blue balloon followed Kira across the park. She tied it to her wrist and let it bob beside her.",
    "Grandpa fixed the clock with a gentle tap. The hands began to move, and the room sounded awake again.",
    "A shell on the beach held the sound of waves. Mina listened, smiled, and put it safely in her pocket.",
    "Tom made a nest from yarn for a wooden bird. The bird did not sing, but it looked very comfortable.",
    "Rae found a button in the grass. She sewed it onto a sock puppet and named the puppet Captain Dot.",
    "The bakery smelled like warm bread. A child chose the round loaf because it looked like a soft brown moon.",
)


@dataclass
class TextSplits:
    """Tokenized text splits for training and evaluation."""

    train: torch.Tensor
    calibration: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor
    source_name: str


def _repeat_to_size(texts: list[str], size: int) -> list[str]:
    if size <= 0:
        return []
    repeated: list[str] = []
    index = 0
    while len(repeated) < size:
        base = texts[index % len(texts)]
        repeated.append(f"{base} Story number {index + 1}.")
        index += 1
    return repeated[:size]


def _load_builtin(total: int) -> tuple[list[str], str]:
    return _repeat_to_size(list(BUILTIN_STORIES), total), "bundled_smoke_corpus"


def _load_tinystories(total: int) -> tuple[list[str], str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for TinyStories. Install requirements.txt or use source='builtin'.") from exc

    dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{total}]")
    texts = [str(item["text"]) for item in dataset]
    return texts, "roneneldan/TinyStories"


def load_texts(config: DataConfig) -> tuple[list[str], str]:
    """Load raw text according to the dataset configuration."""

    total = config.train_size + config.calibration_size + config.validation_size + config.test_size
    if config.source == "builtin":
        return _load_builtin(total)
    if config.source == "tinystories":
        return _load_tinystories(total)
    path = Path(config.source)
    if path.exists():
        texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return _repeat_to_size(texts, total), str(path)
    raise ValueError(f"unknown dataset source: {config.source}")


def tokenize_splits(config: DataConfig, tokenizer: ByteTokenizer, seq_len: int, seed: int) -> TextSplits:
    """Load, shuffle, tokenize, and split text data."""

    texts, source_name = load_texts(config)
    rng = random.Random(seed)
    rng.shuffle(texts)
    ids = torch.tensor([tokenizer.encode(text, seq_len) for text in texts], dtype=torch.long)
    a = config.train_size
    b = a + config.calibration_size
    c = b + config.validation_size
    d = c + config.test_size
    return TextSplits(train=ids[:a], calibration=ids[a:b], validation=ids[b:c], test=ids[c:d], source_name=source_name)


def sample_batch(tokens: torch.Tensor, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    """Sample rows with replacement."""

    indices = torch.randint(0, tokens.size(0), (batch_size,), generator=generator)
    return tokens[indices].clone()


def mask_batch(
    tokens: torch.Tensor,
    tokenizer: ByteTokenizer,
    mask_ratio_min: float,
    mask_ratio_max: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply random masking and return masked tokens plus masked-position labels."""

    if not 0.0 < mask_ratio_min <= mask_ratio_max <= 1.0:
        raise ValueError("mask ratio bounds must satisfy 0 < min <= max <= 1")
    masked = tokens.clone()
    labels = torch.full_like(tokens, fill_value=-100)
    special = torch.zeros_like(tokens, dtype=torch.bool)
    for special_id in (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id):
        special |= tokens.eq(special_id)
    eligible = ~special
    random_values = torch.rand(tokens.size(0), generator=generator)
    ratios = mask_ratio_min + (mask_ratio_max - mask_ratio_min) * random_values
    for row in range(tokens.size(0)):
        row_positions = torch.nonzero(eligible[row], as_tuple=False).flatten()
        if row_positions.numel() == 0:
            continue
        count = max(1, int(round(float(ratios[row]) * row_positions.numel())))
        perm = torch.randperm(row_positions.numel(), generator=generator)[:count]
        chosen = row_positions[perm]
        masked[row, chosen] = tokenizer.mask_token_id
        labels[row, chosen] = tokens[row, chosen]
    return masked, labels


def make_initial_mask(
    tokens: torch.Tensor,
    tokenizer: ByteTokenizer,
    mask_ratio: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic evaluation masks for a provided generator state."""

    return mask_batch(tokens, tokenizer, mask_ratio, mask_ratio, generator)


def iter_batches(tokens: torch.Tensor, batch_size: int, limit_batches: int | None = None) -> Iterable[torch.Tensor]:
    """Yield contiguous batches from a tensor."""

    count = 0
    for start in range(0, tokens.size(0), batch_size):
        if limit_batches is not None and count >= limit_batches:
            break
        yield tokens[start : start + batch_size].clone()
        count += 1

