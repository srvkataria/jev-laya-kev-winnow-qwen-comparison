#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx

from src.config import load_config, gpt_config
from src.data.loader import BatchLoader
from src.eval.metrics import compute_loss
from src.model.gpt import GPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on validation data.")
    parser.add_argument("--config", type=str, default="config/model_50m.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=100)
    return parser.parse_args()


def _load_model(config: dict, checkpoint: Path) -> GPT:
    meta_path = checkpoint.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_cfg = gpt_config({"model": meta["model"]})
    else:
        model_cfg = gpt_config(config)

    model = GPT(**model_cfg)
    weights = mx.load(str(checkpoint))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model


def main() -> None:
    args = parse_args()
    config = load_config(ROOT / args.config)
    data_dir = Path(args.data_dir or config["data"]["output_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint

    model = _load_model(config, checkpoint)
    loader = BatchLoader(
        data_dir / "val.bin",
        batch_size=config["train"]["batch_size"],
        seq_len=config["model"]["seq_len"],
    )

    total = 0.0
    for i, batch in enumerate(loader):
        total += compute_loss(model, batch)
        if i + 1 >= args.max_batches:
            break

    val_loss = total / max(i + 1, 1)
    ppl = math.exp(min(val_loss, 20))
    print(f"val_loss={val_loss:.4f}  perplexity={ppl:.2f}  batches={i + 1}")


if __name__ == "__main__":
    main()
