import { useEffect, useState } from "react";

type SystemInfo = {
  id: string;
  name: string;
  color: string;
  note: string | null;
  jev_cost_per_call_usd?: number | null;
};

type Catalog = {
  ticket_count: number;
  local_hourly_cost_usd: number;
  systems: SystemInfo[];
};

type Metrics = {
  latency_p95_ms: number;
  accuracy: number;
  ai_resolution_rate: number;
  cost_per_thousand_usd: number;
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

type BatchResult = {
  system: string;
  method: string;
  url: string;
  status_code: number | null;
  latency_ms: number;
  ticket_count: number;
  payload: unknown;
  output: unknown;
  saved_path: string | null;
};

const PLACEHOLDERS: SystemInfo[] = [
  { id: "qwen", name: "Qwen", color: "#C4501A", note: null },
  { id: "jev", name: "Jev", color: "#1F4E9B", note: "Sends ticket text to TypeSafe" },
  { id: "laya", name: "Laya", color: "#6E2F8A", note: null },
  { id: "laya-typed-decisions", name: "Laya typed", color: "#7C3AED", note: null },
  { id: "kev-0.8b", name: "Kev 0.8B", color: "#0E7A4B", note: null },
  { id: "kev-4b", name: "Kev 4B", color: "#0A5C44", note: null },
  { id: "winnow-e4b", name: "Winnow e4b", color: "#9F1239", note: null },
  { id: "winnow-12b", name: "Winnow 12B", color: "#831843", note: null },
];

function show(value: unknown): string {
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

export function BatchPage() {
  const [systems, setSystems] = useState<SystemInfo[]>(PLACEHOLDERS);
  const [ticketCount, setTicketCount] = useState(200);
  const [hourly, setHourly] = useState(0.05);
  const [jevPriced, setJevPriced] = useState(false);
  const [scores, setScores] = useState<Record<string, Summary>>({});
  const [pageError, setPageError] = useState<string | null>(null);
  const [active, setActive] = useState<string | null>(null);
  const [error, setError] = useState<{ system: string; message: string } | null>(null);
  const [result, setResult] = useState<BatchResult | null>(null);

  useEffect(() => {
    fetch("/api/systems")
      .then(async (response) => {
        if (!response.ok) throw new Error("The API is not running.");
        return response.json() as Promise<Catalog>;
      })
      .then((catalog) => {
        setSystems(catalog.systems);
        setTicketCount(catalog.ticket_count);
        setHourly(catalog.local_hourly_cost_usd);
        const jev = catalog.systems.find((system) => system.id === "jev");
        setJevPriced((jev?.jev_cost_per_call_usd ?? 0) > 0);
      })
      .catch(() => setPageError("The API is not running. Start it, then reload this page."));
    fetch("/api/batch/results")
      .then(async (response) => {
        if (!response.ok) throw new Error("The batch scores could not be loaded.");
        return response.json() as Promise<Record<string, Summary>>;
      })
      .then(setScores)
      .catch(() => undefined);
  }, []);

  async function start(system: string) {
    setError(null);
    setPageError(null);
    setActive(system);
    setResult(null);
    try {
      const response = await fetch("/api/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ system }),
      });
      const body = (await response.json().catch(() => ({}))) as BatchResult & { detail?: unknown };
      if (!response.ok) {
        const message = typeof body.detail === "string" ? body.detail : "The batch could not start.";
        setError({ system, message });
        return;
      }
      setResult(body);
      const scored = await fetch("/api/batch/results");
      if (scored.ok) setScores((await scored.json()) as Record<string, Summary>);
    } catch {
      setError({ system, message: "Lost contact with the API." });
    } finally {
      setActive(null);
    }
  }

  return (
    <main className="page">
      <div className="top">
        <h1>Batch jobs</h1>
        <a className="navlink" href="#scores">
          Scores
        </a>
      </div>
      <p className="lede">One request sends all {ticketCount} tickets. The payload and the reply stay on this page.</p>
      {pageError ? <p className="footnote">{pageError}</p> : null}

      <section className="buttons" aria-label="Run a batch">
        {systems.map((system) => {
          const sending = active === system.id;
          const finished = scores[system.id]?.metrics != null;
          const message = sending
            ? `Sending ${ticketCount} tickets…`
            : error?.system === system.id
              ? error.message
              : result?.system === system.id || finished
                ? "Done"
                : "Not run yet";
          const tone = error?.system === system.id ? "error" : result?.system === system.id || sending || finished ? "live" : "quiet";
          return (
            <div className="action" key={system.id}>
              <button
                className="run"
                style={{ background: system.color }}
                disabled={active !== null}
                onClick={() => start(system.id)}
              >
                Run {system.name}
              </button>
              <p className={tone === "error" ? "status error" : "status"} style={tone === "live" ? { color: system.color } : undefined}>
                {message}
              </p>
              {system.note ? <p className="note">{system.note}</p> : null}
            </div>
          );
        })}
      </section>

      {result ? (
        <>
          <p className="meta">
            {result.method} {result.url}
            {" · "}
            {result.status_code === null ? "No response" : `HTTP ${result.status_code}`}
            {" · "}
            {Math.round(result.latency_ms).toLocaleString("en-US")} ms
            {" · "}
            {result.ticket_count} tickets
            {result.saved_path ? ` · Saved to ${result.saved_path}` : ""}
          </p>
          <div className="panels">
            <section className="sheet panel">
              <h2>Payload</h2>
              <pre>{show(result.payload)}</pre>
            </section>
            <section className="sheet panel">
              <h2>Output</h2>
              <pre>{show(result.output) || "The model returned no text."}</pre>
            </section>
          </div>
        </>
      ) : null}

      <div className="sheet">
        <table>
          <thead>
            <tr>
              <th />
              {systems.map((system) => (
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
                {systems.map((system) => {
                  const value = scores[system.id]?.metrics?.[metric.key];
                  return <td key={system.id}>{value === undefined ? "—" : metric.format(value)}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="footnote">
        Latency is the one batch request. Local cost uses ${hourly.toFixed(2)} per hour of that request.
        {jevPriced
          ? " Jev uses its per-call price."
          : " Jev’s cost stays at $0 until you set JEV_COST_PER_CALL_USD."}
      </p>
    </main>
  );
}
