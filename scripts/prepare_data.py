#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.prepare import prepare_tinystories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and tokenize TinyStories.")
    parser.add_argument("--config", type=str, default="config/model_50m.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(ROOT / args.config)
    data_cfg = config["data"]

    output_dir = args.output_dir or data_cfg["output_dir"]
    max_samples = args.max_samples if args.max_samples is not None else data_cfg["max_samples"]
    vocab_size = args.vocab_size or data_cfg["vocab_size"]

    prepare_tinystories(
        output_dir=ROOT / output_dir,
        dataset_id=data_cfg["dataset_id"],
        vocab_size=vocab_size,
        val_ratio=data_cfg["val_ratio"],
        max_samples=max_samples,
        min_chars=data_cfg.get("min_chars", 10),
        max_chars=data_cfg.get("max_chars", 0),
    )


if __name__ == "__main__":
    main()
