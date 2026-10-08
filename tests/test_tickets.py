import json
from collections import Counter

from backend.config import DATA_PATH


def test_two_hundred_tickets_are_balanced():
    rows = [json.loads(line) for line in DATA_PATH.read_text().splitlines() if line.strip()]
    assert len(rows) == 200
    assert len({row["id"] for row in rows}) == 200
    counts = Counter(row["gold_queue"] for row in rows)
    assert counts["billing"] == 34
    assert counts["bugs"] == 34
    assert counts["refunds"] == 33
    assert counts["api_access"] == 33
    assert counts["sales"] == 33
    assert counts["other"] == 33
