from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from src.model.gpt import GPT


def compute_loss(model: GPT, inputs: mx.array) -> float:
    x, y = inputs[..., :-1], inputs[..., 1:]
    logits = model(x)
    vocab = logits.shape[-1]
    loss = nn.losses.cross_entropy(
        logits.reshape(-1, vocab),
        y.reshape(-1),
        reduction="mean",
    )
    mx.eval(loss)
    return loss.item()


def compute_perplexity(model: GPT, inputs: mx.array) -> float:
    import math

    return math.exp(min(compute_loss(model, inputs), 20))
