#!/usr/bin/env python3
"""Local page for writing TinyStories with the trained checkpoint.

The model loads once at startup. Each request streams new tokens as
server-sent events and stops when the model emits <|endoftext|>.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

from src.config import gpt_config, load_config
from src.eval.generate import stream_generate
from src.model.gpt import GPT

WEB_DIR = ROOT / "web"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "run-full-20k" / "step_20000.safetensors"
MIN_TOKENS = 32
MAX_TOKENS = 256
MIN_TEMPERATURE = 0.2
MAX_TEMPERATURE = 1.2
MAX_PROMPT_CHARS = 4_000
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([.,!?;:])")


class StoryApp:
    def __init__(self, model: GPT, tokenizer: Tokenizer, top_p: float) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.top_p = top_p
        self.eot_id = tokenizer.token_to_id("<|endoftext|>")
        self.lock = threading.Lock()


APP: StoryApp | None = None


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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _tidy(text: str) -> str:
    """Drop the space this BPE vocab puts before punctuation."""
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text)


def _continuation(tokenizer: Tokenizer, prompt_ids: list[int], all_ids: list[int]) -> str:
    base = tokenizer.decode(prompt_ids)
    full = tokenizer.decode(all_ids)
    if full.startswith(base):
        text = full[len(base) :]
    else:
        text = full
    return _tidy(text)


class StoryHandler(BaseHTTPRequestHandler):
    server_version = "TinyStories/1.0"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_bytes((WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path != "/api/generate":
            self._send_json(404, {"error": "Not found."})
            return
        if APP is None:
            self._send_json(503, {"error": "The model is not loaded."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Missing request body."})
            return
        if length <= 0 or length > 64_000:
            self._send_json(400, {"error": "Request body is empty or too large."})
            return

        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Request body must be JSON."})
            return

        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            self._send_json(400, {"error": "Write a prompt first."})
            return
        if len(prompt) > MAX_PROMPT_CHARS:
            self._send_json(400, {"error": "That prompt is too long."})
            return

        try:
            max_tokens = int(payload.get("max_tokens", 128))
            temperature = float(payload.get("temperature", 0.8))
        except (TypeError, ValueError):
            self._send_json(400, {"error": "Length and temperature must be numbers."})
            return

        max_tokens = int(_clamp(max_tokens, MIN_TOKENS, MAX_TOKENS))
        temperature = _clamp(temperature, MIN_TEMPERATURE, MAX_TEMPERATURE)

        if not APP.lock.acquire(blocking=False):
            self._send_json(
                409,
                {"error": "The model is already writing a story. Wait for it to finish."},
            )
            return

        try:
            self._stream_story(prompt, max_tokens, temperature)
        finally:
            APP.lock.release()

    def _stream_story(self, prompt: str, max_tokens: int, temperature: float) -> None:
        assert APP is not None
        prompt_ids = APP.tokenizer.encode(prompt).ids
        if not prompt_ids:
            self._send_json(400, {"error": "The tokenizer could not read that prompt."})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        all_ids = list(prompt_ids)
        previous = ""
        stop_ids = {APP.eot_id} if APP.eot_id is not None else set()
        try:
            for token_id in stream_generate(
                APP.model,
                prompt_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=APP.top_p,
                stop_ids=stop_ids,
            ):
                all_ids.append(token_id)
                text = _continuation(APP.tokenizer, prompt_ids, all_ids)
                if text != previous:
                    previous = text
                    self._write_event({"text": text})
            self._write_event({"done": True})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            self._write_event({"error": str(exc)})

    def _write_event(self, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local TinyStories writing page.")
    parser.add_argument("--config", type=str, default="config/model_50m.yaml")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    global APP
    args = parse_args()
    config = load_config(ROOT / args.config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    tokenizer_path = checkpoint.parent / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = ROOT / config["data"]["output_dir"] / "tokenizer.json"
    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer not found next to checkpoint or in data dir: {tokenizer_path}")

    print(f"Loading {checkpoint}", flush=True)
    model = _load_model(config, checkpoint)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.decoder is None:
        tokenizer.decoder = ByteLevel()
    APP = StoryApp(model, tokenizer, top_p=float(config["eval"]["top_p"]))

    server = ThreadingHTTPServer((args.host, args.port), StoryHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Ready at {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
