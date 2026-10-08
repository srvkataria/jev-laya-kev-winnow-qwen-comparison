from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import httpx

from backend.config import (
    RESULTS_DIR,
    SYSTEM_ORDER,
    BenchConfig,
    Settings,
    load_bench,
)
from backend.metrics import auto_resolved, summarize
from backend.models import jev
from backend.models.decision import DecisionClient, RunAbort
from backend.models.qwen import QwenClient
from backend.batch import send_batch
from backend.schemas import BatchResult, RouteCall, RunFile, RunStatus, SystemId, SystemInfo, Ticket, TicketResult
from backend.tickets import load_tickets


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class Runner:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.state = RunStatus(status="idle", tickets_total=0)
        self._task: asyncio.Task[None] | None = None
        self._batching = False

    async def start(self, system_id: SystemId) -> None:
        async with self._lock:
            if self.state.status == "running" or self._batching:
                raise BusyError("A run is already in progress.")
            settings = load_settings()
            bench = load_bench()
            tickets = load_tickets()
            message = preflight(system_id, settings)
            if message:
                raise PreflightError(message)
            self.state = RunStatus(
                status="running",
                system=system_id,
                tickets_done=0,
                tickets_total=len(tickets),
                error=None,
            )
            self._task = asyncio.create_task(self._execute(system_id, settings, bench, tickets))

    async def run_batch(self, system_id: SystemId) -> BatchResult:
        async with self._lock:
            if self.state.status == "running" or self._batching:
                raise BusyError("A run is already in progress.")
            settings = load_settings()
            bench = load_bench()
            message = preflight(system_id, settings)
            if message:
                raise PreflightError(message)
            tickets = load_tickets()
            spec = bench.systems[system_id]
            model = spec.model or settings.ollama_style_qwen_model
            self._batching = True
        try:
            return await send_batch(system_id, settings, model, tickets, bench.queues)
        finally:
            async with self._lock:
                self._batching = False

    async def _execute(
        self,
        system_id: SystemId,
        settings: Settings,
        bench: BenchConfig,
        tickets: list[Ticket],
    ) -> None:
        client = None
        started_at = _now()
        try:
            client = _build_client(system_id, settings, bench)
            await client.warmup()
            rows: list[TicketResult] = []
            for ticket in tickets:
                call = await _route_with_retry(client, ticket.text)
                if call.abort:
                    raise RunAbort(call.abort)
                rows.append(_to_result(ticket, call, bench.confidence_cutoff))
                self.state.tickets_done += 1
            finished_at = _now()
            result = RunFile(
                system=system_id,
                started_at=started_at,
                finished_at=finished_at,
                ticket_count=len(rows),
                confidence_cutoff=bench.confidence_cutoff,
                local_hourly_cost_usd=settings.local_hourly_cost_usd,
                metrics=summarize(rows),
                tickets=rows,
            )
            _write_result(result)
            self.state = RunStatus(
                status="done",
                system=system_id,
                tickets_done=len(rows),
                tickets_total=len(tickets),
                error=None,
            )
        except RunAbort as exc:
            self.state = RunStatus(
                status="error",
                system=system_id,
                tickets_done=self.state.tickets_done,
                tickets_total=len(tickets),
                error=str(exc),
            )
        except Exception as exc:
            self.state = RunStatus(
                status="error",
                system=system_id,
                tickets_done=self.state.tickets_done,
                tickets_total=len(tickets),
                error=f"The run stopped: {exc}",
            )
        finally:
            if client is not None:
                await client.shutdown()


class BusyError(Exception):
    pass


class PreflightError(Exception):
    pass


def load_settings() -> Settings:
    from backend.config import get_settings

    return get_settings()


def preflight(system_id: SystemId, settings: Settings) -> str | None:
    if system_id == "jev" and not settings.jev_api_key.strip():
        return "Jev needs JEV_API_KEY in .env."
    return None


def _build_client(system_id: SystemId, settings: Settings, bench: BenchConfig):
    spec = bench.systems[system_id]
    queues = bench.queues
    if system_id == "qwen":
        return QwenClient(settings, queues)
    if system_id == "jev":
        return jev.build(settings, queues)
    if not spec.model:
        raise RuntimeError(f"{spec.name} has no model tag in configs/systems.yaml.")
    return DecisionClient(
        settings=settings,
        queues=queues,
        display_name=spec.name,
        model=spec.model,
        base_url=settings.ollaya_base_url,
        api_key=None,
        per_call_usd=None,
        hourly_usd=settings.local_hourly_cost_usd,
        runtime_name="ollaya",
    )


async def _route_with_retry(client, text: str) -> RouteCall:
    last: RouteCall | None = None
    for _ in range(2):
        last = await client.route(text)
        if last.abort or last.valid_output:
            return last
    assert last is not None
    return last


def _to_result(ticket: Ticket, call: RouteCall, cutoff: float) -> TicketResult:
    valid = call.valid_output and call.predicted_queue is not None
    predicted = call.predicted_queue if valid else None
    return TicketResult(
        id=ticket.id,
        gold_queue=ticket.gold_queue,
        predicted_queue=predicted,
        correct=valid and predicted == ticket.gold_queue,
        confidence=call.confidence if valid else 0,
        auto_resolved=auto_resolved(valid, call.confidence, cutoff),
        latency_ms=round(call.latency_ms, 1),
        cost_usd=call.cost_usd,
        valid_output=valid,
        raw_response=call.raw_response,
    )


def _write_result(result: RunFile) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{result.system}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(result.model_dump_json(indent=2) + "\n")
    temporary.replace(path)


def read_result(system_id: str) -> dict | None:
    path = RESULTS_DIR / f"{system_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


async def probe(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError:
        return "not running"
    return None


def system_infos(settings: Settings, bench: BenchConfig, probes: dict[str, str | None]) -> list[SystemInfo]:
    from backend.schemas import LastRun

    infos: list[SystemInfo] = []
    for system_id in SYSTEM_ORDER:
        spec = bench.systems[system_id]
        load_error = probes.get(system_id)
        saved = read_result(system_id)
        last_run = None
        if saved and "metrics" in saved:
            last_run = LastRun(
                finished_at=saved["finished_at"],
                ticket_count=saved["ticket_count"],
                metrics=saved["metrics"],
            )
        infos.append(
            SystemInfo(
                id=system_id,  # type: ignore[arg-type]
                name=spec.name,
                color=spec.color,
                loaded=load_error is None,
                load_error=load_error,
                remote=spec.remote,
                note=spec.note,
                est_ram_gb=spec.est_ram_gb,
                jev_cost_per_call_usd=settings.jev_cost_per_call_usd if system_id == "jev" else None,
                last_run=last_run,
            )
        )
    return infos
