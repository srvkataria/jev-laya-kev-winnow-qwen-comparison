import { useCallback, useEffect, useState } from "react";
import { BatchPage } from "./BatchPage";

type Metrics = {
  latency_p95_ms: number;
  accuracy: number;
  ai_resolution_rate: number;
  cost_per_thousand_usd: number;
};

type SystemInfo = {
  id: string;
  name: string;
  color: string;
  loaded: boolean;
  load_error: string | null;
  note: string | null;
  jev_cost_per_call_usd: number | null;
  last_run: { metrics: Metrics } | null;
};

type Catalog = {
  ticket_count: number;
  queue_count: number;
  local_hourly_cost_usd: number;
  systems: SystemInfo[];
};

type RunState = {
  status: "idle" | "running" | "done" | "error";
  system: string | null;
  tickets_done: number;
  tickets_total: number;
  error: string | null;
};

type Summary = { metrics: Metrics } | null;

const METRICS: { key: keyof Metrics; label: string; format: (value: number) => string }[] = [
  {
    key: "latency_p95_ms",
    label: "Latency",
    format: (value) => `${Math.round(value).toLocaleString("en-US")} ms`,
  },
  {
    key: "accuracy",
    label: "Accuracy",
    format: (value) => `${(value * 100).toFixed(1)}%`,
  },
  {
    key: "ai_resolution_rate",
    label: "AI Resolution Rate",
    format: (value) => `${(value * 100).toFixed(1)}%`,
  },
  {
    key: "cost_per_thousand_usd",
    label: "Cost per thousand tickets",
    format: (value) => (value >= 1 ? `$${value.toFixed(2)}` : `$${value.toFixed(4)}`),
  },
];

const IDLE_RUN: RunState = {
  status: "idle",
  system: null,
  tickets_done: 0,
  tickets_total: 0,
  error: null,
};

async function readJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}).`);
  }
  return response.json() as Promise<T>;
}

function useHash(): string {
  const [hash, setHash] = useState(() => window.location.hash);
  useEffect(() => {
    const onChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return hash;
}

export function App() {
  const hash = useHash();
  if (hash === "#batch") return <BatchPage />;
  return <ScorePage />;
}

function ScorePage() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [results, setResults] = useState<Record<string, Summary>>({});
  const [run, setRun] = useState<RunState>(IDLE_RUN);
  const [startError, setStartError] = useState<{ system: string; message: string } | null>(null);
  const [pageError, setPageError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const [nextCatalog, nextResults] = await Promise.all([
      readJson<Catalog>("/api/systems"),
      readJson<Record<string, Summary>>("/api/results"),
    ]);
    setCatalog(nextCatalog);
    setResults(nextResults);
  }, []);

  useEffect(() => {
    let cancelled = false;
    Promise.all([refresh(), readJson<RunState>("/api/runs/current")])
      .then(([, current]) => {
        if (!cancelled) setRun(current);
      })
      .catch(() => {
        if (!cancelled) setPageError("The API is not running. Start it, then reload this page.");
      });
    return () => {
      cancelled = true;
    };
  }, [refresh]);

  useEffect(() => {
    if (run.status !== "running") return;
    const timer = window.setInterval(() => {
      readJson<RunState>("/api/runs/current")
        .then((current) => {
          setRun(current);
          if (current.status === "done" || current.status === "error") {
            refresh().catch(() => setPageError("The results could not be loaded."));
          }
        })
        .catch(() => setPageError("Lost contact with the API during the run."));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [run.status, refresh]);

  async function start(system: string) {
    setStartError(null);
    setPageError(null);
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ system }),
    });
    if (!response.ok) {
      const body = (await response.json().catch(() => ({}))) as { detail?: unknown };
      const message = typeof body.detail === "string" ? body.detail : "The run could not start.";
      setStartError({ system, message });
      return;
    }
    setRun((await response.json()) as RunState);
  }

  const running = run.status === "running";
  const hourly = catalog?.local_hourly_cost_usd ?? 0.05;
  const jev = catalog?.systems.find((system) => system.id === "jev");
  const jevPriced = (jev?.jev_cost_per_call_usd ?? 0) > 0;

  return (
    <main className="page">
      <div className="top">
        <h1>Ticket routing</h1>
        <a className="navlink" href="#batch">
          Batch jobs
        </a>
      </div>
      <p className="lede">
        {catalog ? `${catalog.ticket_count} tickets, six queues.` : "200 tickets, six queues."}
      </p>
      {pageError ? <p className="footnote">{pageError}</p> : null}

      <section className="buttons" aria-label="Run a system">
        {(catalog?.systems ?? placeholderSystems()).map((system) => {
          const message = statusMessage(system, run, startError);
          return (
            <div className="action" key={system.id}>
              <button
                className="run"
                style={{ background: system.color }}
                disabled={running}
                onClick={() => start(system.id)}
              >
                Run {system.name}
              </button>
              <p
                className={messageTone(system, run, startError) === "error" ? "status error" : "status"}
                style={messageTone(system, run, startError) === "live" ? { color: system.color } : undefined}
              >
                {message}
              </p>
              {system.note ? <p className="note">{system.note}</p> : null}
            </div>
          );
        })}
      </section>

      <div className="sheet">
        <table>
          <thead>
            <tr>
              <th />
              {(catalog?.systems ?? placeholderSystems()).map((system) => (
                <th key={system.id} style={{ color: system.color, ["--header" as string]: system.color }}>
                  {system.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {METRICS.map((metric) => (
              <tr key={metric.key}>
                <td>{metric.label}</td>
                {(catalog?.systems ?? placeholderSystems()).map((system) => {
                  const value = results[system.id]?.metrics?.[metric.key];
                  return <td key={system.id}>{value === undefined ? "—" : metric.format(value)}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="footnote">
        Local cost uses ${hourly.toFixed(2)} per hour of model time.
        {jevPriced
          ? " Jev uses its per-call price."
          : " Jev’s cost stays at $0 until you set JEV_COST_PER_CALL_USD."}
      </p>
    </main>
  );
}

function messageTone(
  system: SystemInfo,
  run: RunState,
  startError: { system: string; message: string } | null,
): "live" | "error" | "quiet" {
  if (run.status === "running" && run.system === system.id) return "live";
  if (startError?.system === system.id) return "error";
  if (run.status === "error" && run.system === system.id && run.error) return "error";
  if (run.status === "done" && run.system === system.id) return "live";
  if (system.last_run) return "live";
  if (system.load_error) return "error";
  return "quiet";
}

function statusMessage(
  system: SystemInfo,
  run: RunState,
  startError: { system: string; message: string } | null,
): string {
  if (run.status === "running" && run.system === system.id) {
    return `${run.tickets_done} / ${run.tickets_total}`;
  }
  if (startError?.system === system.id) return startError.message;
  if (run.status === "error" && run.system === system.id && run.error) return run.error;
  if (system.last_run || run.status === "done") {
    if (run.status === "done" && run.system === system.id) return "Done";
    if (system.last_run) return "Done";
  }
  if (system.load_error) return system.load_error;
  return "Not run yet";
}

function placeholderSystems(): SystemInfo[] {
  return [
    { id: "qwen", name: "Qwen", color: "#C4501A", loaded: false, load_error: null, note: null, jev_cost_per_call_usd: null, last_run: null },
    { id: "jev", name: "Jev", color: "#1F4E9B", loaded: false, load_error: null, note: "Sends ticket text to TypeSafe", jev_cost_per_call_usd: 0, last_run: null },
    { id: "laya", name: "Laya", color: "#6E2F8A", loaded: false, load_error: null, note: null, jev_cost_per_call_usd: null, last_run: null },
    { id: "laya-typed-decisions", name: "Laya typed", color: "#7C3AED", loaded: false, load_error: null, note: null, jev_cost_per_call_usd: null, last_run: null },
    { id: "kev-0.8b", name: "Kev 0.8B", color: "#0E7A4B", loaded: false, load_error: null, note: null, jev_cost_per_call_usd: null, last_run: null },
    { id: "kev-4b", name: "Kev 4B", color: "#0A5C44", loaded: false, load_error: null, note: null, jev_cost_per_call_usd: null, last_run: null },
    { id: "winnow-e4b", name: "Winnow e4b", color: "#9F1239", loaded: false, load_error: null, note: null, jev_cost_per_call_usd: null, last_run: null },
    { id: "winnow-12b", name: "Winnow 12B", color: "#831843", loaded: false, load_error: null, note: null, jev_cost_per_call_usd: null, last_run: null },
  ];
}
