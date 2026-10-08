from backend.config import load_bench
from backend.models.decision import parse_choice, probability_of_choice
from backend.models.qwen import parse_qwen


def keys() -> list[str]:
    return list(load_bench().queues)


def test_choice_uses_probability_of_the_picked_queue():
    payload = {
        "answers": {
            "queue": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.6,
                "probabilities": {"billing": 0.8, "refunds": 0.2},
            }
        }
    }
    predicted, confidence, valid = parse_choice(payload, ["billing", "refunds"])
    assert valid is True
    assert predicted == "billing"
    assert confidence == 0.8


def test_choice_rejects_an_unknown_queue():
    payload = {"answers": {"queue": {"choice": "legal", "probabilities": {"legal": 1}}}}
    predicted, confidence, valid = parse_choice(payload, keys())
    assert valid is False
    assert predicted is None
    assert confidence == 0


def test_probability_list_follows_criteria_order():
    assert probability_of_choice([0.2, 0.8], "refunds", ["billing", "refunds"]) == 0.8


def test_qwen_json_inside_extra_text():
    raw = 'Sure.\n{"queue": "bugs", "confidence": 0.91, "reason": "blank page"}'
    predicted, confidence, valid = parse_qwen(raw, keys())
    assert (predicted, confidence, valid) == ("bugs", 0.91, True)


def test_qwen_rejects_confidence_above_one():
    raw = '{"queue": "sales", "confidence": 91, "reason": "price"}'
    assert parse_qwen(raw, keys()) == (None, 0.0, False)
