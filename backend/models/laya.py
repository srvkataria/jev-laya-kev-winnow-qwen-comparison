from backend.config import QueueSpec, Settings, load_bench
from backend.models.decision import DecisionClient


def build(settings: Settings, queues: dict[str, QueueSpec]) -> DecisionClient:
    model = load_bench().systems["laya"].model or "laya:en"
    return DecisionClient(
        settings=settings,
        queues=queues,
        display_name="Laya",
        model=model,
        base_url=settings.ollaya_base_url,
        api_key=None,
        per_call_usd=None,
        hourly_usd=settings.local_hourly_cost_usd,
        runtime_name="ollaya",
    )
