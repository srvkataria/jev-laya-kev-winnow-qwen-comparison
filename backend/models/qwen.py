"""Qwen instruct via the Ollama chat API. Confidence is the number the model writes."""

from __future__ import annotations

import json
import re
import time

import httpx

from backend.config import QueueSpec, Settings
from backend.models.decision import RunAbort
from backend.schemas import RouteCall


def build_prompt(text: str, queues: dict[str, QueueSpec]) -> str:
    lines = "\n".join(f"- {key}: {spec.description}" for key, spec in queues.items())
    return (
        "You route customer support tickets to exactly one queue.\n\n"
        f"Queues:\n{lines}\n\n"
        'Reply with JSON only: {"queue": "<key>", "confidence": <number from 0 to 1>, "reason": "<short>"}\n'
        "Use a queue key from the list. confidence is how sure you are, from 0 to 1.\n\n"
        "Ticket:\n"
        '"""\n'
        f"{text}\n"
        '"""'
    )


def parse_qwen(content: str, criteria_keys: list[str]) -> tuple[str | None, float, bool]:
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match is None:
        return None, 0.0, False
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, 0.0, False
    if not isinstance(payload, dict):
        return None, 0.0, False
    queue = payload.get("queue")
    if not isinstance(queue, str) or queue not in criteria_keys:
        return None, 0.0, False
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        return None, 0.0, False
    if confidence < 0 or confidence > 1:
        return None, 0.0, False
    return queue, confidence, True


class QwenClient:
    def __init__(self, settings: Settings, queues: dict[str, QueueSpec]) -> None:
        self.settings = settings
        self.queues = queues
        self.criteria_keys = list(queues)
        self.base_url = settings.qwen_base_url.rstrip("/")
        self.model = settings.ollama_style_qwen_model
        self._client: httpx.AsyncClient | None = None

    async def _client_or_new(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0), trust_env=False)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def warmup(self) -> None:
        client = await self._client_or_new()
        try:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "ok",
                    "stream": False,
                    "keep_alive": -1,
                    "think": False,
                    "options": {"num_predict": 1, "num_ctx": 4096, "temperature": 0},
                },
                timeout=180.0,
            )
        except (httpx.ConnectError, httpx.TimeoutException):
            raise RunAbort(self._down_message()) from None
        if response.status_code >= 400:
            raise RunAbort(_ollama_error(response))

    async def shutdown(self) -> None:
        client = await self._client_or_new()
        try:
            await client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "keep_alive": 0},
            )
        except httpx.HTTPError:
            pass
        await self.aclose()

    async def route(self, text: str) -> RouteCall:
        started = time.perf_counter()
        client = await self._client_or_new()
        try:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": build_prompt(text, self.queues)}],
                    "stream": False,
                    "format": "json",
                    "keep_alive": -1,
                    "think": False,
                    "options": {"temperature": 0, "num_ctx": 4096},
                },
            )
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise RunAbort(self._down_message()) from None
        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - started) * 1000
            return RouteCall(
                predicted_queue=None,
                confidence=0,
                valid_output=False,
                latency_ms=latency_ms,
                cost_usd=self._cost(latency_ms),
                raw_response="Timed out after 30 seconds.",
            )

        latency_ms = (time.perf_counter() - started) * 1000
        if response.status_code >= 400:
            if response.status_code in {401, 404, 422}:
                raise RunAbort(_ollama_error(response))
            return RouteCall(
                predicted_queue=None,
                confidence=0,
                valid_output=False,
                latency_ms=latency_ms,
                cost_usd=self._cost(latency_ms),
                raw_response=_ollama_error(response),
            )
        try:
            payload = response.json()
            content = payload["message"]["content"]
        except (json.JSONDecodeError, KeyError, TypeError):
            content = response.text
        if not isinstance(content, str):
            content = json.dumps(content)
        predicted, confidence, valid = parse_qwen(content, self.criteria_keys)
        return RouteCall(
            predicted_queue=predicted,
            confidence=confidence,
            valid_output=valid,
            latency_ms=latency_ms,
            cost_usd=self._cost(latency_ms),
            raw_response=content[:12000],
        )

    def _cost(self, latency_ms: float) -> float:
        return (latency_ms / 1000 / 3600) * self.settings.local_hourly_cost_usd

    def _down_message(self) -> str:
        return f"Ollama is not running at {self.base_url}. Start it, then press Run Qwen again."


def _ollama_error(response: httpx.Response) -> str:
    try:
        body = response.json()
    except Exception:
        body = None
    if isinstance(body, dict) and body.get("error"):
        return str(body["error"])
    text = response.text.strip()
    return text[:500] if text else f"Ollama returned HTTP {response.status_code}."
