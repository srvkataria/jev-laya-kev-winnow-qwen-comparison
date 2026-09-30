#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, merge_cli_overrides
from src.train.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyStories GPT (~50M) with MLX.")
    parser.add_argument("--config", type=str, default="config/model_50m.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(ROOT / args.config)
    config = merge_cli_overrides(
        config,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
    )

    data_dir = args.data_dir or config["data"]["output_dir"]
    checkpoint_dir = args.checkpoint_dir or config["train"]["checkpoint_dir"]

    trainer = Trainer(
        config=config,
        data_dir=ROOT / data_dir,
        checkpoint_dir=ROOT / checkpoint_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
