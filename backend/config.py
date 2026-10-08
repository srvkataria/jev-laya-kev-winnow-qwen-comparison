from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tickets.jsonl"
RESULTS_DIR = ROOT / "results"
BATCH_RESULTS_DIR = RESULTS_DIR / "batch"
SYSTEMS_PATH = ROOT / "configs" / "systems.yaml"

SYSTEM_ORDER = (
    "qwen",
    "jev",
    "laya",
    "laya-typed-decisions",
    "kev-0.8b",
    "kev-4b",
    "winnow-e4b",
    "winnow-12b",
)


class QueueSpec(BaseModel):
    label: str
    description: str


class SystemSpec(BaseModel):
    name: str
    color: str
    est_ram_gb: float
    remote: bool
    note: str | None = None
    model: str | None = None


class BenchConfig(BaseModel):
    confidence_cutoff: float = 0.8
    queues: dict[str, QueueSpec]
    systems: dict[str, SystemSpec]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    jev_api_key: str = ""
    jev_cost_per_call_usd: float = 0.0
    qwen_base_url: str = "http://127.0.0.1:11434"
    ollama_style_qwen_model: str = "qwen3:8b"
    ollaya_base_url: str = "http://127.0.0.1:11435"
    ram_budget_gb: float = 18
    local_hourly_cost_usd: float = 0.05
    jev_base_url: str = "https://api.typesafe.ai"


def load_bench() -> BenchConfig:
    raw = yaml.safe_load(SYSTEMS_PATH.read_text())
    bench = BenchConfig.model_validate(raw)
    missing = [system_id for system_id in SYSTEM_ORDER if system_id not in bench.systems]
    if missing:
        raise RuntimeError(f"configs/systems.yaml is missing {', '.join(missing)}")
    return bench


def get_settings() -> Settings:
    return Settings()
