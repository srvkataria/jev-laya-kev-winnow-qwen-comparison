# Ticket routing POC

A small local proof of concept for a video. Four systems read the same support tickets and pick a queue. One screen runs each system and shows four numbers.

The long research bench is described in `ticket-routing-poc-spec.md`. This POC is the cut we are building.

## The question

Of Qwen, Jev, Laya, and Kev 0.8B, which one should file support tickets, and how many of those tickets still need a person?

## Queues

A queue is the inbox a ticket is sent to. The model reads the ticket and picks one. With one queue there is nothing to decide.

| Key | Queue | What lands there |
|---|---|---|
| `billing` | Billing | Payments, invoices, plan changes, failed charges |
| `refunds` | Refunds | Requests to return money |
| `api_access` | API access | API keys, dashboard access, rate limits |
| `bugs` | Bugs | Errors or broken features |
| `sales` | Sales | Pricing and plan questions from prospects |
| `other` | Other | Anything that does not clearly fit |

## Tickets

200 tickets, the same set for every system. About 30–35 tickets in each queue, so one busy inbox cannot carry the score.

Each ticket has an id, the text, and the correct queue (`gold_queue`). Synthetic tickets are fine for the video if the video says they are synthetic. They are cleaner than real support mail, so the numbers demonstrate the method.

## Systems

| System | Where it runs | Confidence |
|---|---|---|
| Qwen | Local instruct model | A number it writes in its answer, from 0 to 1 |
| Jev | TypeSafe API (`jev-latest`) | Probability of the queue it picked. Ticket text leaves the machine |
| Laya | Local, Ollaya tag `laya:en` | Probability of the queue it picked |
| Laya typed | Local, Ollaya tag `laya:typed-decisions` | Probability of the queue it picked |
| Kev 0.8B | Local | Probability of the queue it picked |
| Kev 4B | Local, Ollaya tag `kev:4b` | Probability of the queue it picked |
| Winnow e4b | Local, Ollaya tag `winnow:e4b` | Probability of the queue it picked |
| Winnow 12B | Local, Ollaya tag `winnow:12b` | Probability of the queue it picked |

Jev, Laya, Kev, and Winnow score every queue and return those probabilities. Qwen is asked for the queue and a confidence.

## The four metrics

Reported in this order, for each system, over the same 200 tickets.

| Order | Metric | What it is |
|---|---|---|
| 1 | Latency | p95, in milliseconds. Nineteen out of twenty tickets finish faster than this. |
| 2 | Accuracy | Share of all 200 tickets sent to the correct queue. An invalid answer counts as wrong. |
| 3 | AI Resolution Rate | Share scored 0.8 or higher. The computer files those. Every ticket under 0.8 goes to a human. |
| 4 | Cost per thousand tickets | The bill scaled to 1,000 tickets, in USD. |

The 0.8 cutoff is fixed before any scores are looked at, and it is the same for all four systems.

On 100 tickets, if 60 score 0.8 or higher, the AI Resolution Rate is 60%. A person reads the other 40. Those 60 still include mistakes. Accuracy is what shows those mistakes.

At 200 tickets, a gap of a few points is noise. A gap of about 10 points is enough to say on camera.

### Cost

Jev’s cost comes from its API price. Check the price for 200 calls before that run.

Qwen, Laya, and Kev are priced as time on the Mac: `(total compute seconds / 3600) × 0.05`. The $0.05 per hour figure is an assumption for hardware and electricity. Show it next to the local cost numbers.

## What one run stores

Each system writes one file when its button is pressed:

```
results/qwen.json
results/jev.json
results/laya.json
results/laya-typed-decisions.json
results/kev-0.8b.json
results/kev-4b.json
results/winnow-e4b.json
results/winnow-12b.json
```

The file has the four aggregate metrics and one record per ticket. Per ticket, store the fields the aggregates are built from: predicted queue, correct or not, confidence, whether it was auto-resolved (`confidence >= 0.8`), latency, cost, and the raw reply. The shape is in `Tech.md`.

A batch job writes `results/batch/<system>.json` instead. That file keeps the single request payload, the reply, the HTTP status, and the latency.

## The screen

One page. One button per system: **Run Qwen**, **Run Jev**, **Run Laya**, **Run Laya typed**, **Run Kev 0.8B**, **Run Kev 4B**, **Run Winnow e4b**, **Run Winnow 12B**. Each button sends all 200 tickets through that system only. A run shows progress (`34 / 200`). When it finishes, that system’s column fills in with Latency, Accuracy, AI Resolution Rate, and Cost per thousand tickets.

The page uses a warm paper background and a strong contrasting color for each system, on its button and in its results column. Colors are listed in `Tech.md`.

Run one system at a time. On a Mac mini, model weights share memory with the browser and the OS.

## How to read a result

Latency tells you whether a ticket can be filed while the person is still looking at it. Accuracy tells you how often the queue is right. AI Resolution Rate tells you how much of the pile the computer takes. Cost tells you what 1,000 tickets would cost, with Jev as the only system that sends text off the machine.

## Results

Same 200 synthetic tickets, cutoff 0.8. A high-confidence mistake is a wrong queue the system would still file on its own.

Jev’s cost is $0 here because `JEV_COST_PER_CALL_USD` is still 0. The API is not free. Local cost is model time at $0.05 per hour.

| System | Latency p95 | Accuracy | AI Resolution Rate | Cost / 1,000 | High-confidence mistakes |
|---|---:|---:|---:|---:|---:|
| Qwen | 1,917 ms | 94.0% (188/200) | 100% (200/200) | $0.0224 | 12 |
| Jev | 367 ms | 100% (200/200) | 97.0% (194/200) | $0.00 | 0 |
| Laya | 50 ms | 70.0% (140/200) | 62.0% (124/200) | $0.0007 | 9 |
| Laya typed | 95 ms | 75.5% (151/200) | 27.0% (54/200) | $0.0011 | 0 |
| Kev 0.8B | 170 ms | 84.0% (168/200) | 6.5% (13/200) | $0.0023 | 0 |
| Kev 4B | 2,009 ms | 92.0% (184/200) | 11.5% (23/200) | $0.0269 | 0 |
| Winnow e4b | 614 ms | 96.5% (193/200) | 77.0% (154/200) | $0.0083 | 7 |
| Winnow 12B | 2,341 ms | 96.5% (193/200) | 100% (200/200) | $0.0240 | 7 |

Correct tickets by the queue they belonged to. Each column is correct / tickets in that queue.

| System | Billing | Refunds | API access | Bugs | Sales | Other |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 34/34 | 33/33 | 33/33 | 34/34 | 33/33 | 21/33 |
| Jev | 34/34 | 33/33 | 33/33 | 34/34 | 33/33 | 33/33 |
| Laya | 34/34 | 30/33 | 26/33 | 34/34 | 14/33 | 2/33 |
| Laya typed | 34/34 | 23/33 | 26/33 | 34/34 | 13/33 | 21/33 |
| Kev 0.8B | 34/34 | 30/33 | 33/33 | 20/34 | 18/33 | 33/33 |
| Kev 4B | 29/34 | 22/33 | 33/33 | 34/34 | 33/33 | 33/33 |
| Winnow e4b | 34/34 | 26/33 | 33/33 | 34/34 | 33/33 | 33/33 |
| Winnow 12B | 34/34 | 26/33 | 33/33 | 34/34 | 33/33 | 33/33 |

Where the misses went:

| System | Misses | Pattern |
|---|---:|---|
| Qwen | 12 | `other` → billing (9), `other` → bugs (3). All twelve were filed at confidence 0.95. |
| Jev | 0 | Every queue was perfect. Six correct refunds stayed under 0.8 and went to a person. |
| Laya | 60 | Sales 14/33 and `other` 2/33. Nine mistakes were still auto-filed, mostly API access called bugs. |
| Laya typed | 49 | `other` rises to 21/33. Refunds fall to 23/33. No high-confidence mistake. |
| Kev 0.8B | 32 | Bugs → `other` (14) and sales → `other` (15). |
| Kev 4B | 16 | Refunds → billing (11) and billing → `other` (5), all under 0.8. |
| Winnow e4b | 7 | The same seven refunds sent to billing, all auto-filed (0.81–0.85). |
| Winnow 12B | 7 | Those same seven refunds, auto-filed with higher confidence (0.88–0.96). |

Jev is the only run that is fully correct and still files most of the pile. Winnow e4b is the closest local system. Winnow 12B ties it on accuracy and is worse on speed, cost, and confidence: the larger model became more sure of the same refund-versus-billing mistake.

## Run

Copy `.env.example` to `.env`. Then, from the project root:

```
uv sync
uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Each button runs all 200 tickets through that one system and writes `results/<system>.json`. Qwen needs Ollama. Laya and Kev 0.8B need Ollaya. Jev needs `JEV_API_KEY`. Details are in `Tech.md`.
