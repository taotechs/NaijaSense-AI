"use client";

import Link from "next/link";
import { FormEvent, useState } from "react";
import { BackendStatus } from "@/components/BackendStatus";
import { publicTaskUrl } from "@/lib/api-root";
import { TaskAResponse, postTaskA } from "@/lib/task-api";

function StarRow({ rating }: { rating: number }) {
  const full = Math.round(rating);
  return (
    <div className="flex items-center gap-1 text-amber-400" aria-label={`Rating ${rating} of 5`}>
      {Array.from({ length: 5 }).map((_, i) => (
        <span key={i}>{i < full ? "\u2605" : "\u2606"}</span>
      ))}
      <span className="ml-2 text-sm text-slate-400">{rating.toFixed(1)} / 5</span>
    </div>
  );
}

export default function TaskAPage() {
  const [userId, setUserId] = useState("demo_user");
  const [location, setLocation] = useState("Lagos");
  const [interests, setInterests] = useState("street food, amala, suya");
  const [sentiment, setSentiment] = useState("balanced");
  const [toneNotes, setToneNotes] = useState("Value for money; honest Nigerian tone.");
  const [itemName, setItemName] = useState("Iya Eba Amala Spot");
  const [itemContext, setItemContext] = useState(
    "Saturday lunch with a friend; amala soft, egusi rich, about 2k each, 20 min wait."
  );

  const [result, setResult] = useState<TaskAResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const endpoint = publicTaskUrl("/task-a/user-modeling");

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await postTaskA({
        user_persona: {
          user_id: userId.trim(),
          location,
          interests: interests
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
          sentiment_bias: sentiment,
          tone_notes: toneNotes,
        },
        product_details: {
          item_name: itemName.trim(),
          item_context: itemContext,
        },
        persona_style: "nigerian_twitter",
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
          <p className="text-[11px] uppercase tracking-[0.22em] text-brand-500">Task A</p>
          <h2 className="text-xl font-semibold text-slate-100">User modeling</h2>
          <p className="mt-1 text-sm text-slate-400">
            Persona + product → star rating + written review (hackathon endpoint).
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
          <textarea className="field min-h-16" value={toneNotes} onChange={(e) => setToneNotes(e.target.value)} placeholder="Tone / style notes" />

          <h3 className="pt-2 font-medium text-slate-200">Product details</h3>
          <input className="field" value={itemName} onChange={(e) => setItemName(e.target.value)} placeholder="Item name" required />
          <textarea className="field min-h-24" value={itemContext} onChange={(e) => setItemContext(e.target.value)} placeholder="What happened? Price, wait, taste…" />

          <button type="submit" className="btn w-full" disabled={loading}>
            {loading ? "Generating review…" : "Run Task A"}
          </button>
          {error && <p className="rounded-lg bg-red-500/15 p-2 text-sm text-red-300">{error}</p>}
        </form>

        <div className="glass space-y-4 rounded-2xl p-6">
          <h3 className="text-lg font-semibold">Output</h3>
          {!result && (
            <p className="text-sm text-slate-400">Fill the form and run Task A to see rating + review here.</p>
          )}
          {result && (
            <>
              <StarRow rating={result.rating} />
              <details className="rounded-xl border border-slate-800 bg-slate-900/50 p-3" open>
                <summary className="cursor-pointer text-sm font-medium text-slate-300">
                  Pass 1 — review_reasoning
                </summary>
                <p className="mt-2 text-xs text-slate-400">{result.review_reasoning}</p>
              </details>
              <div className="rounded-xl bg-slate-900/80 p-4 text-sm leading-relaxed text-slate-200">
                <p className="mb-1 text-[10px] uppercase tracking-wide text-slate-500">Pass 2 — review_text</p>
                {result.review_text}
              </div>
            </>
          )}
        </div>
      </section>

      <p className="text-center text-xs text-slate-500">
        <Link href="/" className="text-brand-500 hover:underline">
          ← Submission home
        </Link>
        {" · "}
        <Link href="/task-b" className="text-brand-500 hover:underline">
          Task B demo
        </Link>
        {" · "}
        <Link href="/unified" className="text-brand-500 hover:underline">
          Unified hub
        </Link>
      </p>
    </div>
  );
}
