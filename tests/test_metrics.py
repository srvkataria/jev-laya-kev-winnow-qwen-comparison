from backend.metrics import auto_resolved, latency_p95_ms, summarize
from backend.schemas import TicketResult


def _ticket(correct: bool, resolved: bool, latency: float, cost: float) -> TicketResult:
    return TicketResult(
        id="T-0001",
        gold_queue="billing",
        predicted_queue="billing" if correct else "bugs",
        correct=correct,
        confidence=0.9 if resolved else 0.4,
        auto_resolved=resolved,
        latency_ms=latency,
        cost_usd=cost,
        valid_output=True,
        raw_response="",
    )


def test_nearest_rank_p95_on_twenty_values():
    assert latency_p95_ms(list(range(1, 21))) == 19


def test_auto_resolved_uses_the_cutoff():
    assert auto_resolved(True, 0.8, 0.8) is True
    assert auto_resolved(True, 0.79, 0.8) is False
    assert auto_resolved(False, 0.95, 0.8) is False


def test_summarize_four_metrics():
    tickets = [
        _ticket(True, True, 100, 0.01),
        _ticket(True, False, 200, 0.02),
        _ticket(False, True, 300, 0.03),
        _ticket(False, False, 400, 0.04),
    ]
    metrics = summarize(tickets)
    assert metrics.accuracy == 0.5
    assert metrics.ai_resolution_rate == 0.5
    assert metrics.latency_p95_ms == 400
    assert metrics.cost_per_thousand_usd == 25.0
