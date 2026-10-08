from typing import Literal

from pydantic import BaseModel


SystemId = Literal[
    "qwen",
    "jev",
    "laya",
    "laya-typed-decisions",
    "kev-0.8b",
    "kev-4b",
    "winnow-e4b",
    "winnow-12b",
]


class Ticket(BaseModel):
    id: str
    text: str
    gold_queue: str


class TicketResult(BaseModel):
    id: str
    gold_queue: str
    predicted_queue: str | None
    correct: bool
    confidence: float
    auto_resolved: bool
    latency_ms: float
    cost_usd: float
    valid_output: bool
    raw_response: str


class Metrics(BaseModel):
    latency_p95_ms: float
    accuracy: float
    ai_resolution_rate: float
    cost_per_thousand_usd: float


class RunFile(BaseModel):
    system: SystemId
    started_at: str
    finished_at: str
    ticket_count: int
    confidence_cutoff: float
    local_hourly_cost_usd: float
    metrics: Metrics
    tickets: list[TicketResult]


class RunRequest(BaseModel):
    system: SystemId


class BatchResult(BaseModel):
    system: SystemId
    method: str
    url: str
    status_code: int | None
    latency_ms: float
    ticket_count: int
    payload: dict
    output: dict | list | str
    saved_path: str | None = None


class RunStatus(BaseModel):
    status: Literal["idle", "running", "done", "error"]
    system: SystemId | None = None
    tickets_done: int = 0
    tickets_total: int = 0
    error: str | None = None


class LastRun(BaseModel):
    finished_at: str
    ticket_count: int
    metrics: Metrics


class SystemInfo(BaseModel):
    id: SystemId
    name: str
    color: str
    loaded: bool
    load_error: str | None = None
    remote: bool
    note: str | None = None
    est_ram_gb: float
    jev_cost_per_call_usd: float | None = None
    last_run: LastRun | None = None


class RouteCall(BaseModel):
    predicted_queue: str | None
    confidence: float = 0
    valid_output: bool
    latency_ms: float
    cost_usd: float
    raw_response: str
    abort: str | None = None
