"use client";

import { FormEvent, useState } from "react";
import {
  AgentGatewayResponse,
  postAgentGateway,
  UserPersonaPayload
} from "@/lib/agent-api";

function StarRow({ rating }: { rating: number }) {
  const full = Math.round(rating);
  return (
    <div className="flex items-center gap-1 text-amber-400" aria-label={`Rating ${rating} of 5`}>
      {Array.from({ length: 5 }).map((_, i) => (
        <span key={i}>{i < full ? "★" : "☆"}</span>
      ))}
      <span className="ml-2 text-sm text-slate-400">{rating.toFixed(1)} / 5</span>
    </div>
  );
}

export default function UnifiedAgentPage() {
  const [userId, setUserId] = useState("unified_user_1");
  const [location, setLocation] = useState("Lagos");
  const [interests, setInterests] = useState("food, tech, books");
  const [sentiment, setSentiment] = useState("balanced");
  const [toneNotes, setToneNotes] = useState("Use clear, natural English. Keep slang minimal unless requested.");
  const [history, setHistory] = useState("");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AgentGatewayResponse | null>(null);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const persona: UserPersonaPayload = {
      user_id: userId.trim() || "unified_user_1",
      location,
      interests: interests
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      sentiment_bias: sentiment,
      tone_notes: toneNotes.trim() || undefined,
      history: history.trim() || undefined
    };
    try {
      const res = await postAgentGateway({ user_persona: persona, query: q, top_k: 5 });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <div>
        <p className="text-xs uppercase tracking-[0.2em] text-brand-500">NaijaSense AI</p>
        <h2 className="text-2xl font-semibold text-slate-100">Single Chat Agent</h2>
        <p className="mt-1 text-sm text-slate-400">
          One input like ChatGPT/Gemini. The agent auto-routes to review or recommendations.
        </p>
      </div>

      <form onSubmit={onSubmit} className="glass space-y-4 rounded-2xl p-6">
        <textarea
          className="field min-h-28"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask anything: e.g. “Review Amala Spot — quick service, hot food.” or “What should I eat tonight in Lagos?”"
          required
        />
        <details className="rounded-xl border border-slate-800 bg-slate-950/60 p-3 text-sm">
          <summary className="cursor-pointer text-slate-400">Optional persona settings</summary>
          <div className="mt-3 grid gap-3">
            <div className="grid gap-3 sm:grid-cols-2">
              <input className="field" value={userId} onChange={(e) => setUserId(e.target.value)} placeholder="User ID" />
              <input className="field" value={location} onChange={(e) => setLocation(e.target.value)} placeholder="Location" />
            </div>
            <input
              className="field"
              value={interests}
              onChange={(e) => setInterests(e.target.value)}
              placeholder="Interests (comma-separated)"
            />
            <div className="grid gap-3 sm:grid-cols-2">
              <select className="field" value={sentiment} onChange={(e) => setSentiment(e.target.value)}>
                <option value="positive">Sentiment: positive</option>
                <option value="balanced">Sentiment: balanced</option>
                <option value="critical">Sentiment: critical</option>
              </select>
              <input
                className="field"
                value={toneNotes}
                onChange={(e) => setToneNotes(e.target.value)}
                placeholder="Tone / persona notes"
              />
            </div>
            <textarea
              className="field min-h-20"
              value={history}
              onChange={(e) => setHistory(e.target.value)}
              placeholder="Optional: paste history or profile narrative"
            />
          </div>
        </details>
        <button className="btn w-full" disabled={loading || !query.trim()}>
          {loading ? "Routing…" : "Send"}
        </button>
        {error && <p className="text-sm text-red-300">{error}</p>}
      </form>

      {result && (
        <div className="space-y-4">
          <div className="rounded-xl border border-slate-800 bg-slate-950/80 px-4 py-3 text-sm text-slate-400">
            <p>
              <span className="text-brand-500">Task:</span> {result.task}{" "}
              <span className="text-slate-600">·</span>{" "}
              <span className="text-brand-500">Router:</span> {result.routing_source}
            </p>
            <p className="mt-1 text-slate-300">{result.orchestrator_rationale}</p>
          </div>

          {result.task === "review" && result.review && (
            <article className="glass rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-slate-100">Simulated review</h3>
              <div className="mt-3">
                <StarRow rating={result.review.rating} />
              </div>
              <p className="mt-4 whitespace-pre-wrap text-slate-200">{result.review.review_text}</p>
            </article>
          )}

          {result.task === "recommend" && result.recommendation && (
            <article className="glass rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-slate-100">Recommendations</h3>
              {result.recommendation.conversational_response && (
                <p className="mt-3 rounded-xl bg-brand-500/10 p-3 text-sm text-brand-500">
                  {result.recommendation.conversational_response}
                </p>
              )}
              <ul className="mt-4 space-y-3">
                {result.recommendation.recommendations.map((item) => (
                  <li key={item.item_name} className="rounded-xl bg-slate-900/80 p-4">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium text-slate-100">{item.item_name}</span>
                      <span className="text-sm text-brand-500">{item.score.toFixed(2)}</span>
                    </div>
                    <p className="mt-2 text-sm text-slate-400">{item.explanation}</p>
                  </li>
                ))}
              </ul>
            </article>
          )}


          <details className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 text-sm">
            <summary className="cursor-pointer text-slate-400">Reasoning trace</summary>
            <ol className="mt-3 list-decimal space-y-1 pl-5 text-slate-500">
              {result.reasoning_steps.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ol>
          </details>
        </div>
      )}
    </div>
  );
}
