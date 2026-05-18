"use client";

import Link from "next/link";
import { FormEvent, useState } from "react";
import { BackendStatus } from "@/components/BackendStatus";
import { publicTaskUrl } from "@/lib/api-root";
import { TaskBResponse, postTaskB } from "@/lib/task-api";

export default function TaskBPage() {
  const [userId, setUserId] = useState("demo_user");
  const [location, setLocation] = useState("Yaba, Lagos");
  const [interests, setInterests] = useState("food, street food");
  const [sentiment, setSentiment] = useState("balanced");
  const [context, setContext] = useState("Cheap weekend food spots, not too far from campus.");
  const [topK, setTopK] = useState(5);

  const [result, setResult] = useState<TaskBResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const endpoint = publicTaskUrl("/task-b/recommendation");

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await postTaskB({
        user_persona: {
          user_id: userId.trim(),
          location,
          interests: interests
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
          sentiment_bias: sentiment,
        },
        context,
        top_k: topK,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-[11px] uppercase tracking-[0.22em] text-brand-500">Task B</p>
          <h2 className="text-xl font-semibold text-slate-100">Recommendation</h2>
          <p className="mt-1 text-sm text-slate-400">
            Persona (+ optional query) → ranked list with Reason-Before-Recommend trace.
          </p>
          <p className="mt-2 break-all font-mono text-[10px] text-slate-500">
            POST {endpoint}
          </p>
        </div>
        <BackendStatus />
      </div>

      <section className="grid gap-6 lg:grid-cols-2">
        <form onSubmit={onSubmit} className="glass space-y-3 rounded-2xl p-6">
          <h3 className="font-medium text-slate-200">User persona</h3>
          <input className="field" value={userId} onChange={(e) => setUserId(e.target.value)} placeholder="User ID" required />
          <input className="field" value={location} onChange={(e) => setLocation(e.target.value)} placeholder="Location" />
          <input className="field" value={interests} onChange={(e) => setInterests(e.target.value)} placeholder="Interests (comma-separated)" />
          <select className="field" value={sentiment} onChange={(e) => setSentiment(e.target.value)}>
            <option value="positive">Positive</option>
            <option value="balanced">Balanced</option>
            <option value="critical">Critical</option>
          </select>
          <textarea
            className="field min-h-20"
            value={context}
            onChange={(e) => setContext(e.target.value)}
            placeholder="What are you looking for? (optional context query)"
          />
          <label className="block text-xs text-slate-400">
            Top K
            <input
              className="field mt-1"
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value) || 5)}
            />
          </label>

          <button type="submit" className="btn w-full" disabled={loading}>
            {loading ? "Ranking…" : "Run Task B"}
          </button>
          {error && <p className="rounded-lg bg-red-500/15 p-2 text-sm text-red-300">{error}</p>}
        </form>

        <div className="glass space-y-4 rounded-2xl p-6">
          <h3 className="text-lg font-semibold">Output</h3>
          {!result && (
            <p className="text-sm text-slate-400">Run Task B to see ranked recommendations here.</p>
          )}
          {result && (
            <>
              {result.scenario_flags && (
                <div className="flex flex-wrap gap-2 text-[10px] uppercase tracking-wide text-slate-400">
                  {result.scenario_flags.cold_start && (
                    <span className="rounded bg-slate-800 px-2 py-0.5">cold-start</span>
                  )}
                  {result.scenario_flags.cross_domain && (
                    <span className="rounded bg-slate-800 px-2 py-0.5">cross-domain</span>
                  )}
                </div>
              )}
              <ul className="space-y-3">
                {result.recommendations.map((item) => (
                  <li
                    key={item.rank}
                    className="rounded-xl border border-slate-800 bg-slate-900/80 p-3"
                  >
                    <p className="text-sm font-medium text-slate-100">
                      #{item.rank} {item.item_name}
                      <span className="ml-2 text-brand-400">score {item.score}</span>
                    </p>
                    <p className="mt-1 text-xs text-slate-400">{item.explanation}</p>
                  </li>
                ))}
              </ul>
              {result.chain_of_thought?.length > 0 && (
                <details className="rounded-xl border border-slate-800 bg-slate-900/50 p-3" open>
                  <summary className="cursor-pointer text-sm font-medium text-slate-300">
                    Reason-Before-Recommend
                  </summary>
                  <ol className="mt-2 list-decimal space-y-1 pl-5 text-xs text-slate-400">
                    {result.chain_of_thought.map((line, i) => (
                      <li key={i}>{line}</li>
                    ))}
                  </ol>
                </details>
              )}
            </>
          )}
        </div>
      </section>

      <p className="text-center text-xs text-slate-500">
        <Link href="/" className="text-brand-500 hover:underline">
          ← Submission home
        </Link>
        {" · "}
        <Link href="/task-a" className="text-brand-500 hover:underline">
          Task A demo
        </Link>
        {" · "}
        <Link href="/unified" className="text-brand-500 hover:underline">
          Unified hub
        </Link>
      </p>
    </div>
  );
}
