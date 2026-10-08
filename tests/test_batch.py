import json

from backend.batch import decision_payload, qwen_payload, score_batch, write_batch_result
from backend.config import load_bench
from backend.schemas import BatchResult
from backend.tickets import load_tickets


def test_decision_batch_is_one_body_with_every_ticket():
    bench = load_bench()
    tickets = load_tickets()
    payload = decision_payload("laya:en", tickets, bench.queues)

    assert payload["model"] == "laya:en"
    assert len(payload["state"]) == 200
    assert list(payload["questions"]) == [ticket.id for ticket in tickets]
    assert "gold_queue" not in payload["state"][0]
    first = payload["questions"]["T-0001"]
    assert first["type"] == "choice"
    assert set(first["criteria"]) == set(bench.queues)


def test_qwen_batch_lists_every_ticket_once():
    bench = load_bench()
    tickets = load_tickets()
    payload = qwen_payload("qwen3:8b", tickets, bench.queues)
    content = payload["messages"][0]["content"]

    assert payload["stream"] is False
    assert "gold_queue" not in content
    for ticket in tickets:
        assert f"{ticket.id}: {ticket.text}" in content


def test_batch_file_keeps_payload_and_output(tmp_path):
    result = BatchResult(
        system="qwen",
        method="POST",
        url="http://127.0.0.1:11434/api/chat",
        status_code=200,
        latency_ms=12.5,
        ticket_count=200,
        payload={"model": "qwen3:8b"},
        output="",
        saved_path="results/batch/qwen.json",
    )
    path = write_batch_result(result, tmp_path)
    saved = json.loads(path.read_text())

    assert path.name == "qwen.json"
    assert saved["payload"] == {"model": "qwen3:8b"}
    assert saved["output"] == ""
    assert saved["status_code"] == 200
    assert saved["latency_ms"] == 12.5


def test_batch_score_uses_the_same_four_metrics():
    tickets = load_tickets()
    saved = {
        "latency_ms": 1000,
        "status_code": 200,
        "output": {
            "message": {
                "content": json.dumps(
                    {
                        "routes": [
                            {"id": ticket.id, "queue": ticket.gold_queue, "confidence": 0.9}
                            for ticket in tickets
                        ]
                    }
                )
            }
        },
    }
    metrics = score_batch(
        "qwen",
        saved,
        tickets,
        list(load_bench().queues),
        0.8,
        hourly_usd=0.05,
        jev_cost_per_call_usd=0,
    )
    assert metrics is not None
    assert metrics.accuracy == 1
    assert metrics.ai_resolution_rate == 1
    assert metrics.latency_p95_ms == 1000
    assert metrics.cost_per_thousand_usd > 0


def test_unusable_batch_output_counts_every_ticket_wrong():
    metrics = score_batch(
        "laya",
        {"latency_ms": 10, "status_code": 422, "output": {"error": "STATE_TRUNCATED"}},
        load_tickets(),
        list(load_bench().queues),
        0.8,
        hourly_usd=0.05,
        jev_cost_per_call_usd=0,
    )
    assert metrics is not None
    assert metrics.accuracy == 0
    assert metrics.ai_resolution_rate == 0
    assert metrics.latency_p95_ms == 10
