"""Byte-level tokenizer with fixed special tokens."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ByteTokenizer:
    """A reversible UTF-8 byte tokenizer with four special tokens."""

    pad_token_id: int = 0
    mask_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    byte_offset: int = 4

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""

        return 256 + self.byte_offset

    def encode(self, text: str, max_length: int, add_special_tokens: bool = True) -> list[int]:
        """Encode text to a fixed-length token id list."""

        if max_length <= 0:
            raise ValueError("max_length must be positive")
        byte_ids = [b + self.byte_offset for b in text.encode("utf-8", errors="replace")]
        if add_special_tokens:
            ids = [self.bos_token_id] + byte_ids + [self.eos_token_id]
        else:
            ids = byte_ids
        ids = ids[:max_length]
        if ids and add_special_tokens and ids[-1] != self.eos_token_id:
            ids[-1] = self.eos_token_id
        ids += [self.pad_token_id] * (max_length - len(ids))
        return ids

    def decode(self, ids: list[int] | tuple[int, ...], skip_special_tokens: bool = True) -> str:
        """Decode token ids back to text."""

        data: list[int] = []
        for token_id in ids:
            if token_id == self.eos_token_id:
                if skip_special_tokens:
                    break
                continue
            if token_id in {self.pad_token_id, self.mask_token_id, self.bos_token_id}:
                if skip_special_tokens:
                    continue
                continue
            if token_id >= self.byte_offset:
                data.append(max(0, min(255, token_id - self.byte_offset)))
        return bytes(data).decode("utf-8", errors="replace")

