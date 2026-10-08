import json

from backend.config import DATA_PATH, load_bench
from backend.schemas import Ticket


def load_tickets() -> list[Ticket]:
    queues = set(load_bench().queues)
    tickets: list[Ticket] = []
    for line_number, line in enumerate(DATA_PATH.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        ticket = Ticket.model_validate(json.loads(line))
        if ticket.gold_queue not in queues:
            raise RuntimeError(f"{DATA_PATH.name}:{line_number} has unknown queue {ticket.gold_queue}")
        tickets.append(ticket)
    if not tickets:
        raise RuntimeError(f"{DATA_PATH} has no tickets")
    return tickets
