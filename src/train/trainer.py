from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import psutil
from mlx.utils import tree_flatten, tree_map

from src.config import gpt_config
from src.data.loader import BatchLoader, load_meta
from src.model.gpt import GPT, count_parameters


def _get_peak_rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def _cosine_lr(step: int, warmup: int, max_steps: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * (step + 1) / max(warmup, 1)
    if step >= max_steps:
        return lr_min
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def _loss_fn(model: GPT, inputs: mx.array) -> mx.array:
    x, y = inputs[..., :-1], inputs[..., 1:]
    logits = model(x)
    vocab = logits.shape[-1]
    return nn.losses.cross_entropy(
        logits.reshape(-1, vocab),
        y.reshape(-1),
        reduction="mean",
    )


class Trainer:
    def __init__(self, config: dict, data_dir: str | Path, checkpoint_dir: str | Path):
        self.config = config
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_cfg = config["model"]
        train_cfg = config["train"]

        meta = load_meta(self.data_dir)
        model_cfg = gpt_config(config, vocab_size=meta["vocab_size"])

        mx.random.seed(train_cfg.get("seed", 42))

        self.model = GPT(**model_cfg)
        mx.eval(self.model.parameters())
        nparams = count_parameters(self.model)
        print(f"Model parameters: {nparams:,} ({nparams / 1e6:.2f}M)", flush=True)

        self.optimizer = optim.AdamW(
            learning_rate=train_cfg["learning_rate"],
            betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
            weight_decay=train_cfg["weight_decay"],
        )

        self.train_loader = BatchLoader(
            self.data_dir / "train.bin",
            batch_size=train_cfg["batch_size"],
            seq_len=model_cfg["seq_len"],
            seed=train_cfg.get("seed", 42),
        )
        self.val_loader = BatchLoader(
            self.data_dir / "val.bin",
            batch_size=train_cfg["batch_size"],
            seq_len=model_cfg["seq_len"],
            seed=train_cfg.get("seed", 42) + 1,
        )

        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.meta = meta
        self.step = 0
        self.loss_and_grad = nn.value_and_grad(self.model, _loss_fn)

        tok_src = self.data_dir / "tokenizer.json"
        tok_dst = self.checkpoint_dir / "tokenizer.json"
        if tok_src.exists() and not tok_dst.exists():
            shutil.copy2(tok_src, tok_dst)

    def _eval_loss(self, loader: BatchLoader, max_batches: int = 50) -> float:
        total = 0.0
        count = 0
        for batch in loader:
            loss = _loss_fn(self.model, batch)
            mx.eval(loss)
            total += loss.item()
            count += 1
            if count >= max_batches:
                break
        return total / max(count, 1)

    def _save_checkpoint(self, step: int) -> None:
        weights = dict(tree_flatten(self.model.parameters()))
        weights_path = self.checkpoint_dir / f"step_{step}.safetensors"
        mx.save_safetensors(str(weights_path), weights)

        meta = {
            "step": step,
            "model": self.model_cfg,
            "train": self.train_cfg,
            "parameters": count_parameters(self.model),
        }
        meta_path = self.checkpoint_dir / f"step_{step}.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved checkpoint -> {weights_path}", flush=True)

    def train(self) -> None:
        train_cfg = self.train_cfg
        grad_accum = train_cfg["grad_accum_steps"]
        max_steps = train_cfg["max_steps"]
        grad_clip = train_cfg.get("grad_clip", 1.0)

        train_iter = iter(self.train_loader)
        running_loss = 0.0
        log_count = 0
        tic = time.perf_counter()

        print(
            f"Training for {max_steps:,} steps | "
            f"effective batch={train_cfg['batch_size'] * grad_accum} | "
            f"seq_len={self.model_cfg['seq_len']}",
            flush=True,
        )

        while self.step < max_steps:
            accum_grads = None
            accum_loss = 0.0

            for _ in range(grad_accum):
                batch = next(train_iter)
                loss, grads = self.loss_and_grad(self.model, batch)
                accum_loss += loss.item()
                if accum_grads is None:
                    accum_grads = grads
                else:
                    accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)

            accum_grads = tree_map(lambda g: g / grad_accum, accum_grads)
            if grad_clip and grad_clip > 0:
                accum_grads, _ = optim.clip_grad_norm(accum_grads, grad_clip)

            lr = _cosine_lr(
                self.step,
                train_cfg["warmup_steps"],
                max_steps,
                train_cfg["learning_rate"],
                train_cfg.get("lr_min", 1e-5),
            )
            self.optimizer.learning_rate = lr
            self.optimizer.update(self.model, accum_grads)
            mx.eval(self.model.parameters(), self.optimizer.state)

            self.step += 1
            step_loss = accum_loss / grad_accum
            running_loss += step_loss
            log_count += 1

            if self.step % train_cfg["log_every"] == 0:
                avg = running_loss / log_count
                toc = time.perf_counter()
                tok_per_step = (
                    train_cfg["batch_size"] * grad_accum * self.model_cfg["seq_len"]
                )
                print(
                    f"step {self.step:>6} | loss {avg:.4f} | lr {lr:.2e} | "
                    f"{tok_per_step / (toc - tic):,.0f} tok/s | "
                    f"ram {_get_peak_rss_gb():.2f} GB",
                    flush=True,
                )
                running_loss = 0.0
                log_count = 0
                tic = time.perf_counter()

            if self.step % train_cfg["eval_every"] == 0:
                val_loss = self._eval_loss(self.val_loader)
                ppl = math.exp(min(val_loss, 20))
                print(
                    f"step {self.step:>6} | val_loss {val_loss:.4f} | "
                    f"val_ppl {ppl:.2f} | ram {_get_peak_rss_gb():.2f} GB",
                    flush=True,
                )

            if self.step % train_cfg["save_every"] == 0:
                self._save_checkpoint(self.step)

        self._save_checkpoint(self.step)
        print("Training complete.", flush=True)
