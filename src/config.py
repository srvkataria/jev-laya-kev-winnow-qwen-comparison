from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_cli_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Apply non-None CLI overrides into nested config sections."""
    field_map = {
        "data_dir": ("data", "output_dir"),
        "checkpoint_dir": ("train", "checkpoint_dir"),
        "max_samples": ("data", "max_samples"),
        "max_steps": ("train", "max_steps"),
        "batch_size": ("train", "batch_size"),
        "grad_accum_steps": ("train", "grad_accum_steps"),
    }
    for key, value in overrides.items():
        if value is None:
            continue
        if key in field_map:
            section, field = field_map[key]
            config.setdefault(section, {})[field] = value
        elif key in config and isinstance(config[key], dict):
            config[key] = value
        elif "." in key:
            section, field = key.split(".", 1)
            config.setdefault(section, {})[field] = value
        else:
            config[key] = value
    return config


def gpt_config(config: dict, vocab_size: int | None = None) -> dict:
    """Return model kwargs accepted by src.model.GPT."""
    gpt_fields = {
        "vocab_size",
        "n_layer",
        "n_embd",
        "n_head",
        "seq_len",
        "dropout",
        "bias",
        "norm_eps",
    }
    model_cfg = dict(config.get("model", config))
    if vocab_size is not None:
        model_cfg["vocab_size"] = vocab_size
    return {k: v for k, v in model_cfg.items() if k in gpt_fields}
