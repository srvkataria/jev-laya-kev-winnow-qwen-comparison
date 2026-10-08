# Tech

Stack, screen, and result files for the ticket routing POC. Product rules are in `README.md`.

## Stack

Python runs the models. A small web page starts each run and reads the result files. Results are JSON on disk, one file per system. No database.

| Piece | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Model calls, metrics, and file writes stay in one process |
| Dependencies | `uv` | One lockfile, fast installs |
| API | FastAPI + Uvicorn | Four “start run” endpoints, one status endpoint |
| HTTP client | `httpx` | Async calls to Qwen, Jev, Laya, and Kev |
| Models | Pydantic v2 | Request and result shapes |
| UI | Vite, React, TypeScript | One page, four buttons, live progress |
| Styles | CSS variables | Warm paper tokens, no component library |
| Storage | `results/*.json` | One file per system, easy to open on camera |

Confirm each model’s current HTTP schema before writing its client. Tags and fields move quickly. Pin the versions you actually run in `configs/systems.yaml`.

## Repository

```
yt-04-project/
  README.md
  Tech.md
  ticket-routing-poc-spec.md
  pyproject.toml
  .env.example
  configs/systems.yaml
  data/tickets.jsonl          # 200 tickets
  results/
    qwen.json
    jev.json
    laya.json
    laya-typed-decisions.json
    kev-0.8b.json
    kev-4b.json
    winnow-e4b.json
    winnow-12b.json
  backend/
    app.py
    runner.py                 # one run at a time, writes the JSON file
    metrics.py
    models/
      qwen.py
      jev.py
      laya.py
      kev.py
  frontend/                   # Vite + React + TypeScript
```

## Model runtimes

Load one system for its run, then unload it before the next button.

| System | Runtime | Notes |
|---|---|---|
| Qwen | Ollama or `llama-server` | Instruct model. Ask for JSON: `queue`, `confidence` (0–1), `reason`. Temperature 0. |
| Jev | `POST https://api.typesafe.ai/v1/systemone` | Model `jev-latest`. API key in `.env` as `JEV_API_KEY`. Send redacted text only. |
| Laya | Ollaya, tag `laya:en` | Choice question over the six queues. Map returned probabilities onto the queue keys. |
| Laya typed | Ollaya, tag `laya:typed-decisions` | Same choice-question call as Laya. Fine-tuned checkpoint, same size class. |
| Kev 0.8B | Ollaya, tag `kev:0.8b` | Same choice-question call as Laya. The bare tag `kev` is the 4B model, so this POC pins `kev:0.8b`. |
| Kev 4B | Ollaya, tag `kev:4b` | Same choice-question call. About 10 GB. |
| Winnow e4b | Ollaya, tag `winnow:e4b` | Same choice-question call. About 8 GB. |
| Winnow 12B | Ollaya, tag `winnow:12b` | Same choice-question call. About 14 GB free. The bare tag `winnow` is this model. |

Memory on the Mac mini is shared with macOS and the browser. Laya and Kev 0.8B are small. Keep the Qwen context window at 4096 tokens. Unload the previous heavy model before starting the next one.

Timeout is 30 seconds per ticket. Retry once on an invalid queue key, then record the ticket as wrong.

Local cost for one ticket:

```
cost_usd = (latency_ms / 1000 / 3600) * 0.05
```

Jev cost is the API price for that call. If the API returns no dollar amount, use a per-call rate set in `configs/systems.yaml`.

## API

```
GET  /api/systems                 ticket count, hourly cost, and each system's name, color, loaded state, and last run
POST /api/runs                    body: { "system": "qwen" | "jev" | "laya" | "laya-typed-decisions" | "kev-0.8b" | "kev-4b" | "winnow-e4b" | "winnow-12b" }
GET  /api/runs/current            status, tickets_done, tickets_total, error
GET  /api/results                 the four summaries, for the table
GET  /api/results/{system}        that system’s JSON file
```

`POST /api/runs` returns 409 if a run is already in progress. The UI disables all four buttons until the current run finishes or fails.

Progress can be polled every second. Server-Sent Events are optional and not required for this POC.

## Screen

One page. Warm paper, no sidebar.

Top: title “Ticket routing” and the line “200 tickets, six queues.”

Then four buttons in a row:

- Run Qwen
- Run Jev
- Run Laya
- Run Kev 0.8B

Each button is filled with that system’s color and labeled in white. Under the button, a quiet status line: “Not run yet”, “34 / 200”, “Done”, or the error text.

Below the buttons, a table with one column per system, in this row order:

1. Latency
2. Accuracy
3. AI Resolution Rate
4. Cost per thousand tickets

Empty cells say “—” until that system’s file exists. Latency is shown in milliseconds. Accuracy and AI Resolution Rate are shown as percents. Cost is shown in USD.

Jev’s button carries a fixed note: “Sends ticket text to TypeSafe.”

### Warm paper theme

Page background is warm paper. Text is warm ink. Each system color is used on its button, column header, and progress text.

| Token | Hex | Use |
|---|---|---|
| `--paper` | `#F6EFE4` | Page background |
| `--sheet` | `#FFF9F0` | Table and cards |
| `--rule` | `#E4D5C3` | Borders |
| `--ink` | `#2A2118` | Text |
| `--ink-muted` | `#6B5E52` | Status lines, notes |
| `--qwen` | `#C4501A` | Qwen |
| `--jev` | `#1F4E9B` | Jev |
| `--laya` | `#6E2F8A` | Laya |
| `--laya-typed` | `#7C3AED` | Laya typed |
| `--kev` | `#0E7A4B` | Kev 0.8B |
| `--kev-4b` | `#0A5C44` | Kev 4B |
| `--winnow` | `#9F1239` | Winnow e4b |
| `--winnow-12b` | `#831843` | Winnow 12B |

Buttons use white text on the system color. Body text uses IBM Plex Sans, with `system-ui` as the fallback. Numbers use tabular figures so the columns line up.

Corners are slightly rounded (6 px) on buttons. The table sits on the sheet color with a 1 px `--rule` border. No drop shadows.

## Result files

Written by the runner to `results/`. One file per system, replaced on each new run of that system. Pretty-printed JSON.

```json
{
  "system": "laya",
  "started_at": "2026-10-04T08:00:00Z",
  "finished_at": "2026-10-04T08:04:10Z",
  "ticket_count": 200,
  "confidence_cutoff": 0.8,
  "local_hourly_cost_usd": 0.05,
  "metrics": {
    "latency_p95_ms": 480,
    "accuracy": 0.86,
    "ai_resolution_rate": 0.71,
    "cost_per_thousand_usd": 0.012
  },
  "tickets": [
    {
      "id": "T-0001",
      "gold_queue": "billing",
      "predicted_queue": "billing",
      "correct": true,
      "confidence": 0.91,
      "auto_resolved": true,
      "latency_ms": 420,
      "cost_usd": 0.0000058,
      "valid_output": true,
      "raw_response": ""
    }
  ]
}
```

`predicted_queue` is `null` when the output is still invalid after the one retry. `correct` is false in that case. `auto_resolved` is true only when `confidence >= 0.8` and `valid_output` is true.

### How the four metrics are computed

From the `tickets` array, with `n = 200`:

- **Latency.** 95th percentile of `latency_ms`, nearest-rank. Report as `latency_p95_ms`.
- **Accuracy.** Count of `correct == true`, divided by `n`.
- **AI Resolution Rate.** Count of `auto_resolved == true`, divided by `n`.
- **Cost per thousand tickets.** `(sum of cost_usd / n) * 1000`.

The UI reads `metrics` and does not recompute them. The ticket rows are what make the four numbers auditable.

## Tickets file

`data/tickets.jsonl`, one ticket per line:

```json
{"id": "T-0001", "text": "I was charged twice for the Pro plan.", "gold_queue": "billing"}
```

Keys are the six queues in the README. The runner sends every line to the selected system, in file order, one ticket at a time.

## Environment

`.env.example`:

```
JEV_API_KEY=
JEV_COST_PER_CALL_USD=0
QWEN_BASE_URL=http://127.0.0.1:11434
OLLAMA_STYLE_QWEN_MODEL=qwen3:8b
OLLAYA_BASE_URL=http://127.0.0.1:11435
RAM_BUDGET_GB=18
LOCAL_HOURLY_COST_USD=0.05
```

Ports and the Qwen model tag are placeholders until the local install is checked. `JEV_COST_PER_CALL_USD` is the dollar price of one Jev call. Leave it at 0 until you set it from TypeSafe’s pricing, and the cost cell for Jev stays $0.

Stored confidence for Jev, Laya, and Kev is the probability of the queue they picked. The API also returns a separate spread score under the name `confidence`. That score stays in `raw_response`.

## Run

From the project root, in two terminals:

```
uv sync
uv run python scripts/make_tickets.py
uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

```
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Copy `.env.example` to `.env` and fill in `JEV_API_KEY` before Run Jev. Qwen needs Ollama on port 11434. Laya and Kev need Ollaya on port 11435, with `laya:en` and `kev:0.8b` pulled.
