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
