from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm


def _train_tokenizer(texts: list[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<|endoftext|>"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def _encode_split(tokenizer: Tokenizer, texts: list[str]) -> np.ndarray:
    ids: list[int] = []
    eot = tokenizer.token_to_id("<|endoftext|>")
    for text in tqdm(texts, desc="Tokenizing", leave=False):
        encoded = tokenizer.encode(text).ids
        ids.extend(encoded)
        ids.append(eot)
    return np.array(ids, dtype=np.uint16)


def prepare_tinystories(
    output_dir: str | Path,
    dataset_id: str = "roneneldan/TinyStories",
    vocab_size: int = 8000,
    val_ratio: float = 0.01,
    max_samples: int = 0,
    min_chars: int = 10,
    max_chars: int = 0,
    tokenizer_sample_size: int = 500_000,
) -> dict:
    """Download TinyStories, train BPE, write memmapped token shards."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split="train", streaming=False)

    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    texts: list[str] = []
    for row in tqdm(ds, desc="Cleaning"):
        text = row["text"].strip()
        if len(text) < min_chars:
            continue
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        texts.append(text)

    if len(texts) < 100:
        raise ValueError(f"Too few samples after filtering: {len(texts)}")

    split_idx = max(1, int(len(texts) * (1.0 - val_ratio)))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:] or texts[-max(1, len(texts) // 100) :]

    print(f"Training BPE tokenizer (vocab={vocab_size}) on {min(len(train_texts), tokenizer_sample_size):,} stories")
    sample_for_tok = train_texts[: min(len(train_texts), tokenizer_sample_size)]
    tokenizer = _train_tokenizer(sample_for_tok, vocab_size=vocab_size)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    print(f"Encoding train split ({len(train_texts):,} stories)")
    train_ids = _encode_split(tokenizer, train_texts)
    print(f"Encoding val split ({len(val_texts):,} stories)")
    val_ids = _encode_split(tokenizer, val_texts)

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    meta = {
        "dataset_id": dataset_id,
        "vocab_size": vocab_size,
        "train_tokens": int(train_ids.size),
        "val_tokens": int(val_ids.size),
        "train_stories": len(train_texts),
        "val_stories": len(val_texts),
        "val_ratio": val_ratio,
        "max_samples": max_samples,
        "dtype": "uint16",
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved tokenizer -> {tokenizer_path}")
    print(f"Saved train.bin   -> {train_path} ({meta['train_tokens']:,} tokens)")
    print(f"Saved val.bin     -> {val_path} ({meta['val_tokens']:,} tokens)")
    return meta
