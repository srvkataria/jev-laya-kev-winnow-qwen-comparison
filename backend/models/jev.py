from backend.config import QueueSpec, Settings, load_bench
from backend.models.decision import DecisionClient


def build(settings: Settings, queues: dict[str, QueueSpec]) -> DecisionClient:
    model = load_bench().systems["jev"].model or "jev-latest"
    return DecisionClient(
        settings=settings,
        queues=queues,
        display_name="Jev",
        model=model,
        base_url=settings.jev_base_url,
        api_key=settings.jev_api_key,
        per_call_usd=settings.jev_cost_per_call_usd,
        hourly_usd=None,
        runtime_name="jev",
    )
