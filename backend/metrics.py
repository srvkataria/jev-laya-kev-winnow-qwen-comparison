import math

from backend.schemas import Metrics, TicketResult


def auto_resolved(valid_output: bool, confidence: float, cutoff: float) -> bool:
    return valid_output and confidence >= cutoff


def latency_p95_ms(values: list[float]) -> float:
    """Nearest-rank 95th percentile. Rank is ceil(0.95 * n), 1-based."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(0.95 * len(ordered))
    index = min(max(rank, 1), len(ordered)) - 1
    return float(ordered[index])


def summarize(tickets: list[TicketResult]) -> Metrics:
    count = len(tickets)
    if count == 0:
        return Metrics(
            latency_p95_ms=0,
            accuracy=0,
            ai_resolution_rate=0,
            cost_per_thousand_usd=0,
        )
    correct = sum(1 for ticket in tickets if ticket.correct)
    resolved = sum(1 for ticket in tickets if ticket.auto_resolved)
    cost = sum(ticket.cost_usd for ticket in tickets)
    return Metrics(
        latency_p95_ms=round(latency_p95_ms([ticket.latency_ms for ticket in tickets]), 1),
        accuracy=round(correct / count, 4),
        ai_resolution_rate=round(resolved / count, 4),
        cost_per_thousand_usd=round((cost / count) * 1000, 6),
    )
