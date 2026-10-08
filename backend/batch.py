"""One HTTP request that carries every ticket.

Decision models get one System One call: the state is the list of tickets, and
each ticket is its own choice question. Qwen gets one chat call whose prompt
lists every ticket.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from backend.config import BATCH_RESULTS_DIR, ROOT, SYSTEM_ORDER, QueueSpec, Settings
from backend.metrics import auto_resolved
from backend.models.decision import INSTRUCTIONS, QUESTION_ID, parse_choice
from backend.models.qwen import parse_qwen
from backend.schemas import BatchResult, Metrics, SystemId, Ticket


def decision_payload(model: str, tickets: list[Ticket], queues: dict[str, QueueSpec]) -> dict[str, Any]:
    criteria = {key: spec.description for key, spec in queues.items()}
    return {
        "model": model,
        "state": [{"id": ticket.id, "text": ticket.text} for ticket in tickets],
        "questions": {
            ticket.id: {
                "type": "choice",
                "instructions": f"{INSTRUCTIONS} This question is only about ticket {ticket.id}.",
                "criteria": criteria,
            }
            for ticket in tickets
        },
    }


def qwen_payload(model: str, tickets: list[Ticket], queues: dict[str, QueueSpec]) -> dict[str, Any]:
    lines = "\n".join(f"- {key}: {spec.description}" for key, spec in queues.items())
    ticket_lines = "\n".join(f"{ticket.id}: {ticket.text}" for ticket in tickets)
    prompt = (
        "You route customer support tickets to exactly one queue.\n\n"
        f"Queues:\n{lines}\n\n"
        "Reply with JSON only, one object whose key \"routes\" is an array. "
        "Each item is {\"id\": \"<ticket id>\", \"queue\": \"<key>\", \"confidence\": <number from 0 to 1>}. "
        "Include every ticket, in the same order.\n\n"
        "Tickets:\n"
        f"{ticket_lines}"
    )
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "keep_alive": -1,
        "think": False,
        "options": {"temperature": 0, "num_ctx": 8192},
    }


def _qwen_predictions(output: Any, criteria_keys: list[str]) -> dict[str, tuple[str | None, float, bool]] | None:
    if not isinstance(output, dict):
        return None
    message = output.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        body = json.loads(content)
    except json.JSONDecodeError:
        return None
    routes = body.get("routes") if isinstance(body, dict) else None
    if not isinstance(routes, list):
        return None
    found: dict[str, tuple[str | None, float, bool]] = {}
    for item in routes:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        raw = json.dumps({"queue": item.get("queue"), "confidence": item.get("confidence")})
        found[item["id"]] = parse_qwen(raw, criteria_keys)
    return found or None


def _decision_predictions(output: Any, criteria_keys: list[str]) -> dict[str, tuple[str | None, float, bool]] | None:
    if not isinstance(output, dict):
        return None
    answers = output.get("answers")
    if not isinstance(answers, dict) or not answers:
        return None
    found: dict[str, tuple[str | None, float, bool]] = {}
    for ticket_id, answer in answers.items():
        if not isinstance(ticket_id, str):
            continue
        found[ticket_id] = parse_choice({"answers": {QUESTION_ID: answer}}, criteria_keys)
    return found or None


def score_batch(
    system_id: str,
    saved: dict[str, Any],
    tickets: list[Ticket],
    criteria_keys: list[str],
    cutoff: float,
    *,
    hourly_usd: float,
    jev_cost_per_call_usd: float,
) -> Metrics | None:
    output = saved.get("output")
    predictions = _qwen_predictions(output, criteria_keys) if system_id == "qwen" else _decision_predictions(output, criteria_keys)
    if not tickets:
        return None
    # A finished batch with no usable routes still counts: every ticket is wrong.
    if predictions is None:
        predictions = {}
    correct = 0
    resolved = 0
    for ticket in tickets:
        predicted, confidence, valid = predictions.get(ticket.id, (None, 0.0, False))
        if valid and predicted == ticket.gold_queue:
            correct += 1
        if auto_resolved(valid, confidence, cutoff):
            resolved += 1
    count = len(tickets)
    latency_ms = float(saved.get("latency_ms") or 0)
    if system_id == "jev":
        batch_cost = jev_cost_per_call_usd if saved.get("status_code") == 200 else 0.0
    else:
        batch_cost = (latency_ms / 1000 / 3600) * hourly_usd
    return Metrics(
        latency_p95_ms=round(latency_ms, 1),
        accuracy=round(correct / count, 4),
        ai_resolution_rate=round(resolved / count, 4),
        cost_per_thousand_usd=round((batch_cost / count) * 1000, 6),
    )


def batch_summaries(
    tickets: list[Ticket],
    criteria_keys: list[str],
    cutoff: float,
    *,
    hourly_usd: float,
    jev_cost_per_call_usd: float,
) -> dict[str, dict[str, Any] | None]:
    summaries: dict[str, dict[str, Any] | None] = {}
    for system_id in SYSTEM_ORDER:
        path = BATCH_RESULTS_DIR / f"{system_id}.json"
        if not path.exists():
            summaries[system_id] = None
            continue
        saved = json.loads(path.read_text())
        metrics = score_batch(
            system_id,
            saved,
            tickets,
            criteria_keys,
            cutoff,
            hourly_usd=hourly_usd,
            jev_cost_per_call_usd=jev_cost_per_call_usd,
        )
        summaries[system_id] = None if metrics is None else {"system": system_id, "metrics": metrics.model_dump()}
    return summaries


def write_batch_result(result: BatchResult, directory: Path | None = None) -> Path:
    folder = directory or BATCH_RESULTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{result.system}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(result.model_dump_json(indent=2) + "\n")
    temporary.replace(path)
    return path


def _output_body(response: httpx.Response) -> dict | list | str:
    try:
        body = response.json()
    except json.JSONDecodeError:
        return response.text
    if isinstance(body, (dict, list)):
        return body
    return response.text


async def send_batch(
    system_id: SystemId,
    settings: Settings,
    model: str,
    tickets: list[Ticket],
    queues: dict[str, QueueSpec],
) -> BatchResult:
    if system_id == "qwen":
        url = f"{settings.qwen_base_url.rstrip('/')}/api/chat"
        payload = qwen_payload(model, tickets, queues)
        headers = {"Content-Type": "application/json"}
    else:
        base = settings.jev_base_url if system_id == "jev" else settings.ollaya_base_url
        url = f"{base.rstrip('/')}/v1/systemone"
        payload = decision_payload(model, tickets, queues)
        headers = {"Content-Type": "application/json"}
        if system_id == "jev" and settings.jev_api_key.strip():
            headers["Authorization"] = f"Bearer {settings.jev_api_key.strip()}"

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=5.0), trust_env=False) as client:
            response = await client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        result = BatchResult(
            system=system_id,
            method="POST",
            url=url,
            status_code=None,
            latency_ms=round(latency_ms, 1),
            ticket_count=len(tickets),
            payload=payload,
            output=str(exc).strip() or type(exc).__name__,
        )
    else:
        latency_ms = (time.perf_counter() - started) * 1000
        result = BatchResult(
            system=system_id,
            method="POST",
            url=url,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 1),
            ticket_count=len(tickets),
            payload=payload,
            output=_output_body(response),
        )
    result.saved_path = str((BATCH_RESULTS_DIR / f"{system_id}.json").relative_to(ROOT))
    write_batch_result(result)
    return result
