from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import mlx.core as mx

from src.model.gpt import GPT


def _top_p_sample(logits: mx.array, top_p: float) -> int:
    probs = np.array(mx.softmax(logits, axis=-1))
    sorted_idx = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)
    cutoff = int(np.searchsorted(cumulative, top_p))
    kept = sorted_idx[: max(cutoff + 1, 1)]
    kept_probs = probs[kept]
    kept_probs = kept_probs / kept_probs.sum()
    return int(np.random.choice(kept, p=kept_probs))


def _sample_next(model: GPT, ids: list[int], temperature: float, top_p: float) -> int:
    context = ids[-model.seq_len :]
    x = mx.array([context])
    logits = model(x)
    next_logits = logits[0, -1, :] / max(temperature, 1e-6)
    if top_p < 1.0:
        return _top_p_sample(next_logits, top_p)
    return int(mx.random.categorical(next_logits).item())


def stream_generate(
    model: GPT,
    prompt_ids: list[int],
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    stop_ids: set[int] | None = None,
) -> Iterator[int]:
    """Yield newly sampled token ids. Stop before emitting any id in stop_ids."""
    ids = list(prompt_ids)
    stop = set(stop_ids or [])
    for _ in range(max_new_tokens):
        next_id = _sample_next(model, ids, temperature, top_p)
        if next_id in stop:
            return
        ids.append(next_id)
        yield next_id


def generate(
    model: GPT,
    prompt_ids: list[int],
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> list[int]:
    ids = list(prompt_ids)
    ids.extend(
        stream_generate(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    )
    return ids
