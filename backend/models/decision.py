"""Shared System One client for Jev, Laya, and Kev.

Stored confidence is the probability of the chosen queue. The API also returns
a spread score named confidence, (K * p_max - 1) / (K - 1). That score is kept
in the raw response and is not the number used for AI Resolution Rate.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from backend.config import QueueSpec, Settings
from backend.schemas import RouteCall

QUESTION_ID = "queue"
INSTRUCTIONS = "Which support queue should handle this ticket?"


class RunAbort(Exception):
    pass


def choice_body(model: str, text: str, queues: dict[str, QueueSpec]) -> dict[str, Any]:
    return {
        "model": model,
        "state": text,
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": {key: spec.description for key, spec in queues.items()},
            }
        },
    }


def probability_of_choice(probabilities: Any, choice: str, criteria_keys: list[str]) -> float | None:
    if isinstance(probabilities, dict):
        if choice not in probabilities:
            return None
        try:
            return float(probabilities[choice])
        except (TypeError, ValueError):
            return None
    if isinstance(probabilities, list):
        if choice not in criteria_keys:
            return None
        index = criteria_keys.index(choice)
        if index >= len(probabilities):
            return None
        try:
            return float(probabilities[index])
        except (TypeError, ValueError):
            return None
    return None


def parse_choice(payload: Any, criteria_keys: list[str]) -> tuple[str | None, float, bool]:
    if not isinstance(payload, dict):
        return None, 0.0, False
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        return None, 0.0, False
    answer = answers.get(QUESTION_ID)
    if not isinstance(answer, dict):
        return None, 0.0, False
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in criteria_keys:
        return None, 0.0, False
    probability = probability_of_choice(answer.get("probabilities"), choice, criteria_keys)
    if probability is None or probability < 0 or probability > 1:
        return None, 0.0, False
    return choice, probability, True


def error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except Exception:
        body = None
    if isinstance(body, dict) and body.get("error"):
        return str(body["error"])
    text = response.text.strip()
    if text:
        return text[:500]
    return f"The model returned HTTP {response.status_code}."


class DecisionClient:
    def __init__(
        self,
        *,
        settings: Settings,
        queues: dict[str, QueueSpec],
        display_name: str,
        model: str,
        base_url: str,
        api_key: str | None,
        per_call_usd: float | None,
        hourly_usd: float | None,
        runtime_name: str,
    ) -> None:
        self.settings = settings
        self.queues = queues
        self.criteria_keys = list(queues)
        self.display_name = display_name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.per_call_usd = per_call_usd
        self.hourly_usd = hourly_usd
        self.runtime_name = runtime_name
        self._client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _client_or_new(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0), trust_env=False)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def warmup(self) -> None:
        if self.runtime_name != "ollaya":
            return
        client = await self._client_or_new()
        try:
            response = await client.post(
                f"{self.base_url}/api/decide",
                headers=self._headers(),
                json={"model": self.model, "keep_alive": -1},
                timeout=180.0,
            )
        except (httpx.ConnectError, httpx.TimeoutException):
            raise RunAbort(self._down_message()) from None
        if response.status_code >= 400:
            raise RunAbort(self._http_abort(response))

    async def shutdown(self) -> None:
        if self.runtime_name != "ollaya":
            await self.aclose()
            return
        client = await self._client_or_new()
        try:
            await client.post(
                f"{self.base_url}/api/decide",
                headers=self._headers(),
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
                f"{self.base_url}/v1/systemone",
                headers=self._headers(),
                json=choice_body(self.model, text, self.queues),
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise RunAbort(f"{self._down_message()} {exc}") from None
        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - started) * 1000
            return RouteCall(
                predicted_queue=None,
                confidence=0,
                valid_output=False,
                latency_ms=latency_ms,
                cost_usd=0,
                raw_response="Timed out after 30 seconds.",
            )

        latency_ms = (time.perf_counter() - started) * 1000
        raw = response.text
        if response.status_code in {401, 403, 404, 422}:
            raise RunAbort(self._http_abort(response))
        if response.status_code >= 400:
            return RouteCall(
                predicted_queue=None,
                confidence=0,
                valid_output=False,
                latency_ms=latency_ms,
                cost_usd=self._cost(latency_ms, charged=False),
                raw_response=error_message(response),
            )
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = None
        predicted, confidence, valid = parse_choice(payload, self.criteria_keys)
        return RouteCall(
            predicted_queue=predicted,
            confidence=confidence,
            valid_output=valid,
            latency_ms=latency_ms,
            cost_usd=self._cost(latency_ms, charged=True),
            raw_response=raw[:12000],
        )

    def _cost(self, latency_ms: float, charged: bool) -> float:
        if self.per_call_usd is not None:
            return self.per_call_usd if charged else 0.0
        hourly = self.hourly_usd or 0.0
        return (latency_ms / 1000 / 3600) * hourly

    def _down_message(self) -> str:
        if self.runtime_name == "ollaya":
            return (
                f"Ollaya is not running at {self.base_url}. "
                f"Start it, then press Run {self.display_name} again."
            )
        return (
            f"Jev did not respond at {self.base_url}. "
            "Check the network and JEV_API_KEY, then press Run Jev again."
        )

    def _http_abort(self, response: httpx.Response) -> str:
        message = error_message(response)
        if response.status_code == 401:
            if self.runtime_name == "jev":
                return "Jev rejected JEV_API_KEY. Check the key in .env."
            return f"Ollaya rejected the request: {message}"
        return message
