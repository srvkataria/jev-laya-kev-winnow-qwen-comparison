from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        return self.weight * (x * mx.rsqrt(variance + self.eps))


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        b, t, c = x.shape
        q = self.q_proj(x).reshape(b, t, self.n_head, self.head_dim)
        k = self.k_proj(x).reshape(b, t, self.n_head, self.head_dim)
        v = self.v_proj(x).reshape(b, t, self.n_head, self.head_dim)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = (q * self.head_dim**-0.5) @ k.transpose(0, 1, 3, 2)
        scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        if self.dropout > 0:
            weights = nn.Dropout(self.dropout)(weights)
        y = weights @ v
        y = y.transpose(0, 2, 1, 3).reshape(b, t, c)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        hidden = 4 * n_embd
        self.fc1 = nn.Linear(n_embd, hidden, bias=bias)
        self.fc2 = nn.Linear(hidden, n_embd, bias=bias)
        self.dropout = dropout

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu(x)
        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x)
        x = self.fc2(x)
        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
        bias: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.ln1 = RMSNorm(n_embd, eps=norm_eps)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout=dropout, bias=bias)
        self.ln2 = RMSNorm(n_embd, eps=norm_eps)
        self.mlp = MLP(n_embd, dropout=dropout, bias=bias)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layer: int = 10,
        n_embd: int = 512,
        n_head: int = 8,
        seq_len: int = 512,
        dropout: float = 0.0,
        bias: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.seq_len = seq_len

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.drop = nn.Dropout(dropout)
        for i in range(n_layer):
            setattr(
                self,
                f"block_{i}",
                Block(n_embd, n_head, dropout=dropout, bias=bias, norm_eps=norm_eps),
            )
        self.ln_f = RMSNorm(n_embd, eps=norm_eps)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=bias)

    @property
    def blocks(self) -> list[Block]:
        return [getattr(self, f"block_{i}") for i in range(self.n_layer)]

    def __call__(self, idx: mx.array) -> mx.array:
        b, t = idx.shape
        if t > self.seq_len:
            raise ValueError(f"Sequence length {t} exceeds model seq_len {self.seq_len}")

        pos = mx.arange(t)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        mask = nn.MultiHeadAttention.create_additive_causal_mask(t)
        mask = mask.reshape(1, 1, t, t)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        return self.lm_head(x)

    def sanitize(self, weights: dict) -> dict:
        """Tie embedding and output projection weights for loading."""
        if "token_emb.weight" in weights and "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["token_emb.weight"]
        return weights


def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def estimate_params(
    vocab_size: int,
    n_layer: int,
    n_embd: int,
) -> int:
    """Rough parameter estimate for logging before init."""
    embed = vocab_size * n_embd
    per_layer = 12 * n_embd * n_embd
    head = vocab_size * n_embd
    return embed + n_layer * per_layer + head
