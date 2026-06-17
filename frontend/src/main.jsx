import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { BarChart3, Bot, Database, FileUp, Loader2, Send } from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function App() {
  const [messages, setMessages] = useState([
    { role: "assistant", text: "Ask a question about the AWS Customer Agreement." },
  ]);
  const [query, setQuery] = useState("");
  const [sources, setSources] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");

  useEffect(() => {
    loadAnalytics();
  }, []);

  async function ingest() {
    setBusy(true);
    setNotice("");
    try {
      const response = await fetch(`${API_BASE}/ingest`, { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Ingestion failed");
      setNotice(`Ingested ${data.chunks} chunks from ${data.pages} pages using ${data.embedding_backend}.`);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function ask(event) {
    event.preventDefault();
    const clean = query.trim();
    if (!clean) return;
    setMessages((items) => [...items, { role: "user", text: clean }]);
    setQuery("");
    setBusy(true);
    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: clean }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Request failed");
      setMessages((items) => [
        ...items,
        { role: "assistant", text: data.answer, meta: `${data.latency_ms} ms · ${data.llm_provider}` },
      ]);
      setSources(data.sources || []);
      loadAnalytics();
    } catch (error) {
      setMessages((items) => [...items, { role: "assistant", text: error.message }]);
    } finally {
      setBusy(false);
    }
  }

  async function loadAnalytics() {
    try {
      const response = await fetch(`${API_BASE}/analytics`);
      if (response.ok) setAnalytics(await response.json());
    } catch {
      setAnalytics(null);
    }
  }

  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <span className="eyebrow">RAG Document Q&A</span>
          <h1>AWS Customer Agreement Assistant</h1>
        </div>
        <button className="iconButton" onClick={ingest} disabled={busy} title="Ingest PDF">
          {busy ? <Loader2 className="spin" size={18} /> : <FileUp size={18} />}
          Ingest
        </button>
      </section>

      {notice && <div className="notice">{notice}</div>}

      <div className="layout">
        <section className="panel chatPanel">
          <div className="panelTitle">
            <Bot size={18} />
            Chat
          </div>
          <div className="messages">
            {messages.map((message, index) => (
              <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
                <p>{message.text}</p>
                {message.meta && <small>{message.meta}</small>}
              </div>
            ))}
          </div>
          <form className="composer" onSubmit={ask}>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="What does the agreement say about termination?"
            />
            <button className="send" disabled={busy || !query.trim()} title="Send question">
              {busy ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
            </button>
          </form>
        </section>

        <section className="panel">
          <div className="panelTitle">
            <Database size={18} />
            Sources
          </div>
          <div className="sources">
            {sources.length === 0 && <p className="muted">Sources from the latest answer will appear here.</p>}
            {sources.map((source) => (
              <article className="source" key={source.chunk_id}>
                <div>
                  <strong>Page {source.page}</strong>
                  <span>{source.score.toFixed(3)}</span>
                </div>
                <p>{source.text}</p>
              </article>
            ))}
          </div>
        </section>
      </div>

      <section className="panel analytics">
        <div className="panelTitle">
          <BarChart3 size={18} />
          Analytics
        </div>
        <div className="stats">
          <Metric label="Total queries" value={analytics?.summary?.total_queries ?? 0} />
          <Metric label="Answered" value={analytics?.summary?.answered_count ?? 0} />
          <Metric label="No answer" value={analytics?.summary?.no_answer_count ?? 0} />
          <Metric label="Avg latency" value={`${Math.round(analytics?.summary?.average_latency_ms ?? 0)} ms`} />
        </div>
        <div className="tables">
          <Table
            title="Most frequent questions"
            rows={analytics?.most_frequent_questions || []}
            columns={["question", "count", "avg_latency_ms"]}
          />
          <Table
            title="No-answer queries"
            rows={analytics?.no_answer_queries || []}
            columns={["question", "top_similarity", "created_at"]}
          />
        </div>
      </section>
    </main>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Table({ title, rows, columns }) {
  return (
    <div className="tableWrap">
      <h2>{title}</h2>
      <table>
        <thead>
          <tr>{columns.map((column) => <th key={column}>{column.replaceAll("_", " ")}</th>)}</tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr>
              <td colSpan={columns.length}>No data yet</td>
            </tr>
          )}
          {rows.map((row, index) => (
            <tr key={`${title}-${index}`}>
              {columns.map((column) => (
                <td key={column}>{formatCell(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatCell(value) {
  if (typeof value === "number") return Number.isInteger(value) ? value : value.toFixed(2);
  return value || "-";
}

createRoot(document.getElementById("root")).render(<App />);
