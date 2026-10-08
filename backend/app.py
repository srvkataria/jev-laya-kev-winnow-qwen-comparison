from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.batch import batch_summaries
from backend.config import SYSTEM_ORDER, get_settings, load_bench
from backend.runner import BusyError, PreflightError, Runner, probe, read_result, system_infos
from backend.schemas import RunRequest, RunStatus
from backend.tickets import load_tickets

app = FastAPI(title="Ticket routing POC")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

runner = Runner()


@app.get("/api/systems")
async def systems():
    settings = get_settings()
    bench = load_bench()
    qwen_error, ollaya_error = await asyncio.gather(
        probe(f"{settings.qwen_base_url.rstrip('/')}/api/tags"),
        probe(f"{settings.ollaya_base_url.rstrip('/')}/"),
    )
    ollaya_message = (
        f"Ollaya is not running at {settings.ollaya_base_url}." if ollaya_error else None
    )
    probes: dict[str, str | None] = {}
    for system_id in SYSTEM_ORDER:
        if system_id == "qwen":
            probes[system_id] = (
                f"Ollama is not running at {settings.qwen_base_url}." if qwen_error else None
            )
        elif system_id == "jev":
            probes[system_id] = None if settings.jev_api_key.strip() else "Jev needs JEV_API_KEY in .env."
        else:
            probes[system_id] = ollaya_message
    return {
        "ticket_count": len(load_tickets()),
        "queue_count": len(bench.queues),
        "local_hourly_cost_usd": settings.local_hourly_cost_usd,
        "systems": [info.model_dump() for info in system_infos(settings, bench, probes)],
    }


@app.post("/api/batch")
async def batch_job(body: RunRequest):
    try:
        result = await runner.run_batch(body.system)
    except BusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PreflightError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump()


@app.post("/api/runs", status_code=202)
async def start_run(body: RunRequest):
    try:
        await runner.start(body.system)
    except BusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PreflightError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return runner.state


@app.get("/api/runs/current", response_model=RunStatus)
async def current_run():
    return runner.state


@app.get("/api/batch/results")
async def batch_results():
    settings = get_settings()
    bench = load_bench()
    return batch_summaries(
        load_tickets(),
        list(bench.queues),
        bench.confidence_cutoff,
        hourly_usd=settings.local_hourly_cost_usd,
        jev_cost_per_call_usd=settings.jev_cost_per_call_usd,
    )


@app.get("/api/results")
async def results():
    summaries = {}
    for system_id in SYSTEM_ORDER:
        saved = read_result(system_id)
        if saved is None:
            summaries[system_id] = None
            continue
        summaries[system_id] = {
            "system": saved["system"],
            "finished_at": saved["finished_at"],
            "ticket_count": saved["ticket_count"],
            "metrics": saved["metrics"],
        }
    return summaries


@app.get("/api/results/{system_id}")
async def result_file(system_id: str):
    if system_id not in SYSTEM_ORDER:
        raise HTTPException(status_code=404, detail="Unknown system.")
    saved = read_result(system_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="No run saved for this system yet.")
    return saved
