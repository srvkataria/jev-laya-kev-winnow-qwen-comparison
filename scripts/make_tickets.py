"""Write 200 synthetic tickets with a clear gold queue on every line."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "tickets.jsonl"

PLANS = ["Starter", "Pro", "Team", "Business"]
MONTHS = ["March", "April", "May", "June", "July", "August", "September"]
AMOUNTS = [19, 29, 49, 79, 99, 149, 199]
CARDS = [4242, 1881, 9033, 5510, 2208, 7741]


def slot(i: int) -> dict[str, str | int]:
    return {
        "i": i + 1,
        "plan": PLANS[i % len(PLANS)],
        "month": MONTHS[i % len(MONTHS)],
        "amount": AMOUNTS[i % len(AMOUNTS)],
        "card": CARDS[i % len(CARDS)],
    }


def expand(frames: list[str], count: int) -> list[str]:
    texts: list[str] = []
    i = 0
    while len(texts) < count:
        text = frames[len(texts) % len(frames)].format(**slot(i))
        i += 1
        texts.append(text)
    if len(set(texts)) != count:
        raise SystemExit("ticket texts are not unique")
    return texts


BILLING = [
    "The {plan} invoice for {month} failed on the card ending {card}. I did not change the plan.",
    "Please send the {month} receipt for the ${amount} {plan} charge on card {card}.",
    "I updated card {card} and the ${amount} {plan} renewal in {month} still shows as a failed payment.",
    "We need invoice {i} for the {plan} workspace. The {month} payment of ${amount} did not clear.",
    "Switch the {plan} plan payment method to the card ending {card}. The {month} charge was ${amount}.",
    "The ${amount} charge on card {card} in {month} does not match the {plan} price on invoice {i}.",
]

REFUNDS = [
    "Please refund the ${amount} {plan} charge from {month}. I want the money returned to card {card}.",
    "I asked for a refund of invoice {i} and the ${amount} has not come back to card {card}.",
    "Cancel the {plan} plan from {month} and return the ${amount} you took.",
    "The extra ${amount} charge in {month} on card {card} should be refunded. Reference {i}.",
    "We were billed ${amount} for {plan} in {month} after we already paid. Please return that money.",
]

API = [
    "My API key for the {plan} workspace stopped working in {month}. The dashboard will not show the key.",
    "Where do I create a new API key for workspace {i}? The {plan} dashboard hides the keys page.",
    "The API has been returning rate limit errors since {month}. Request {i} was rejected.",
    "Please reset the API secret for workspace {i}. I can sign in, but the key panel is locked.",
    "Our dashboard says the {plan} account has no API access. We need the key we created in {month}.",
]

BUGS = [
    "The export button on the {plan} reports page does nothing. It has been broken since {month}.",
    "Opening project {i} shows a blank page and a 500 error. This started in {month}.",
    "Search returns records from the wrong workspace. Broken since {month}, example id {i}.",
    "The save button overlaps the footer and the form will not submit. Seen on account {i} in {month}.",
    "The {plan} settings page crashes when I open notifications. Error started in {month}, case {i}.",
]

SALES = [
    "Before we buy, what is the price of the {plan} plan for a team of {i}?",
    "I am evaluating tools this {month}. Can you send a quote for the {plan} plan?",
    "We have not signed up. Do you discount {plan} for {i} people?",
    "What is included in {plan} at ${amount} a month? We are still comparing vendors.",
    "Can a prospect get a demo of {plan}? We would decide in {month}. Reference {i}.",
]

OTHER = [
    "Please change the email address on account {i}. Login and billing both work.",
    "What are your support hours? I am asking in {month} and nothing is broken.",
    "I want to delete account {i}. I do not need money returned.",
    "Is the office address on the {month} page still current? Account {i} is fine.",
    "Can I rename workspace {i}? Nothing is broken and I am not asking about price.",
    "Thanks for the help on request {i} in {month}. No further action.",
]


def main() -> None:
    groups = [
        ("billing", expand(BILLING, 34)),
        ("refunds", expand(REFUNDS, 33)),
        ("api_access", expand(API, 33)),
        ("bugs", expand(BUGS, 34)),
        ("sales", expand(SALES, 33)),
        ("other", expand(OTHER, 33)),
    ]
    rows: list[dict[str, str]] = []
    longest = max(len(texts) for _, texts in groups)
    for index in range(longest):
        for queue, texts in groups:
            if index < len(texts):
                rows.append({"text": texts[index], "gold_queue": queue})
    if len(rows) != 200:
        raise SystemExit(f"expected 200 tickets, got {len(rows)}")
    texts = [row["text"] for row in rows]
    if len(set(texts)) != 200:
        raise SystemExit("duplicate ticket text")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for number, row in enumerate(rows, start=1):
        lines.append(
            json.dumps(
                {"id": f"T-{number:04d}", "text": row["text"], "gold_queue": row["gold_queue"]},
                ensure_ascii=True,
            )
        )
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} tickets to {OUT}")


if __name__ == "__main__":
    main()
