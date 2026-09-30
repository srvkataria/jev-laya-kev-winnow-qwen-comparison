# TinyStories-50M — Small Language Model Research Lab

A local research workspace for training and experimenting with a **~50M parameter** decoder-only language model on **Apple Silicon** (Mac Mini), using **MLX** and the **TinyStories** dataset.

Designed for research, ablations, and proof-of-concepts — not production-scale pretraining.

## Goals

- Train a small GPT-style model from scratch on TinyStories
- Stay within a **10 GB RAM** budget during training
- Provide a reproducible loop: data → train → eval → sample
- Support fast iteration for architecture and hyperparameter experiments

## Requirements

| Requirement | Minimum |
|---|---|
| Machine | Mac with Apple Silicon (M-series) |
| RAM | 16 GB unified memory recommended; training capped at ~10 GB |
| OS | macOS 14+ |
| Python | 3.10+ (tested with 3.14 + MLX 0.31) |
| Disk | ~5 GB free (dataset cache + checkpoints) |

## Project Structure

```text
.
├── README.md              # This file — overview and run instructions
├── requirements.txt       # Python dependencies
├── config/
│   └── model_50m.yaml     # Model + training configuration
├── src/
│   ├── config.py          # YAML loader + CLI override helpers
│   ├── data/              # Dataset download, tokenization, batch loader
│   ├── model/             # GPT-style transformer (MLX)
│   ├── train/             # Training loop, checkpointing, logging
│   └── eval/              # Perplexity + text generation
├── scripts/
│   ├── prepare_data.py    # Download and tokenize TinyStories
│   ├── train.py           # Main training entrypoint
│   ├── sample.py          # Generate text from a checkpoint
│   ├── eval.py            # Validation loss / perplexity
│   └── serve.py           # Local story page
├── web/
│   └── index.html         # Story page markup and styles
├── data/                  # Local tokenized shards (gitignored)
├── checkpoints/           # Saved model weights (gitignored)
└── .cache/                # Hugging Face cache (gitignored)
```

## Quick Start

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Optional — keep Hugging Face cache inside the project:

```bash
export HF_HOME="$(pwd)/.cache/huggingface"
```

### 2. Prepare TinyStories

**Smoke test (500 stories):**

```bash
python scripts/prepare_data.py --max-samples 500
```

**Full dataset:**

```bash
python scripts/prepare_data.py --max-samples 0
```

Outputs under `data/tinystories/`:

- `tokenizer.json` — 8k BPE tokenizer
- `train.bin` / `val.bin` — memmapped token shards
- `meta.json` — token counts and split info

### 3. Train (~50M params)

**Smoke test (10 steps):**

```bash
python scripts/train.py --max-steps 10 --checkpoint-dir checkpoints/smoke-test
```

**Default run (100k steps, config-driven):**

```bash
python scripts/train.py \
  --config config/model_50m.yaml \
  --data-dir data/tinystories \
  --checkpoint-dir checkpoints/run-001
```

Training logs loss, learning rate, tokens/sec, and process RSS. Checkpoints saved as `step_{N}.safetensors` + `step_{N}.json`.

### 4. Generate samples

```bash
python scripts/sample.py \
  --checkpoint checkpoints/run-001/step_50000.safetensors \
  --prompt "Once upon a time" \
  --max-tokens 128
```

### 5. Evaluate

```bash
python scripts/eval.py \
  --checkpoint checkpoints/run-001/step_50000.safetensors \
  --data-dir data/tinystories
```

### 6. Open the story page

Loads the Phase 2 checkpoint once and streams a story in the browser. Listens on localhost only.

```bash
source .venv/bin/activate
python scripts/serve.py
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Training Phases (Recommended)

| Phase | Purpose | Dataset | Steps |
|---|---|---|---|
| **0 — Smoke test** | Validate pipeline and RAM usage | 500–10k samples | 10–1k |
| **1 — Short run** | Confirm loss decrease and coherent samples | 1M samples | ~10k |
| **2 — Full run** | Research baseline checkpoint | Full TinyStories | 50k–200k |

Start with Phase 0 before committing to a long run.

**Phase 1 complete:** `checkpoints/run-001/step_10000.safetensors` — val perplexity **5.24**.

**Phase 2 complete:** `checkpoints/run-full-20k/step_20000.safetensors` — val perplexity **4.25** on full TinyStories.

## Memory Budget

Training targets **≤10 GB** peak RAM:

- ~49M parameters (10 layers × 576 dim × 8 heads)
- Context length 512
- Micro-batch 2 × grad accumulation 8 (effective batch 16)
- Memmapped `.bin` loader — dataset not loaded into RAM

## References

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — Ronen Eldan et al.
- [MLX](https://github.com/ml-explore/mlx) — Apple machine learning framework
- [nanoGPT](https://github.com/karpathy/nanoGPT) — conceptual baseline for small GPT training

## License

Research and experimentation use. Dataset subject to [TinyStories terms on Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories).
