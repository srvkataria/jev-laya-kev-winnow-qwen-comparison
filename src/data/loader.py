from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def load_meta(data_dir: str | Path) -> dict:
    meta_path = Path(data_dir) / "meta.json"
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


class BatchLoader:
    """Infinite iterator of (B, T) token batches from a memmapped .bin file."""

    def __init__(
        self,
        data_path: str | Path,
        batch_size: int,
        seq_len: int,
        seed: int = 42,
    ):
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.window = seq_len + 1
        self.max_start = len(self.data) - self.window
        if self.max_start <= 0:
            raise ValueError(
                f"Not enough tokens in {data_path}: need at least {self.window}, got {len(self.data)}"
            )
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self) -> mx.array:
        starts = self.rng.integers(0, self.max_start, size=self.batch_size)
        batch = np.stack(
            [self.data[s : s + self.window].astype(np.int32) for s in starts]
        )
        return mx.array(batch)

    @property
    def num_tokens(self) -> int:
        return int(len(self.data))
