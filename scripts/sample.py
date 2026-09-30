#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

from src.config import load_config, gpt_config
from src.eval.generate import generate
from src.model.gpt import GPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("--config", type=str, default="config/model_50m.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
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
    eval_cfg = config["eval"]

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint

    tokenizer_path = checkpoint.parent / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = ROOT / config["data"]["output_dir"] / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.decoder is None:
        tokenizer.decoder = ByteLevel()

    model = _load_model(config, checkpoint)
    prompt_ids = tokenizer.encode(args.prompt).ids
    output_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_tokens or eval_cfg["max_new_tokens"],
        temperature=args.temperature or eval_cfg["temperature"],
        top_p=args.top_p or eval_cfg["top_p"],
    )
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
