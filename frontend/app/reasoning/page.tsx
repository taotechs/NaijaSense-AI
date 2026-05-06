"use client";

import { FormEvent, useState } from "react";
import { recommendItems, simulateReview } from "@/lib/api";

type ReasoningResult = {
  title: string;
  steps: string[];
  meta?: Record<string, unknown>;
};

export default function ReasoningPage() {
  const [userId, setUserId] = useState("user_demo_1");
  const [itemName, setItemName] = useState("Jollof Bowl");
  const [candidateItems, setCandidateItems] = useState("Foodie Hub, Budget Earbuds, Smartwatch");
  const [mode, setMode] = useState<"review" | "recommend">("review");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ReasoningResult | null>(null);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      if (mode === "review") {
        const res = await simulateReview({
          user_profile: {
            user_id: userId,
            location: "Lagos",
            interests: ["food", "lifestyle"],
            sentiment_bias: "balanced"
          },
          item_data: { item_name: itemName, item_context: "Wanted a quick and tasty meal." },
          persona_style: "nigerian_twitter"
        });
        setResult({
          title: "Review Simulation Reasoning",
          steps: res.reasoning_steps,
          meta: {
            rating: res.rating,
            review: res.review_text
          }
        });
      } else {
        const res = await recommendItems({
          user_profile: {
            user_id: userId,
            location: "Lagos",
            interests: ["food", "tech"]
          },
          candidate_items: candidateItems
            .split(",")
            .map((i) => i.trim())
            .filter(Boolean),
          context: "Need practical options",
          top_k: 3,
          recommender_personality: "analyst",
          conversational_mode: true
        });
        setResult({
          title: "Recommendation Reasoning",
          steps: res.reasoning_steps,
          meta: {
            conversational_response: res.conversational_response,
            explainability: res.explainability
          }
        });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to fetch reasoning logs.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="grid gap-6 lg:grid-cols-2">
      <form onSubmit={onSubmit} className="glass space-y-4 rounded-2xl p-6">
        <h2 className="text-xl font-semibold">Agent Reasoning Viewer</h2>
        <p className="text-sm text-slate-400">Pull step-by-step orchestration logs directly from backend responses.</p>
        <select className="field" value={mode} onChange={(e) => setMode(e.target.value as "review" | "recommend")}>
          <option value="review">Review Flow</option>
          <option value="recommend">Recommendation Flow</option>
        </select>
        <input className="field" value={userId} onChange={(e) => setUserId(e.target.value)} placeholder="User ID" required />
        {mode === "review" ? (
          <input className="field" value={itemName} onChange={(e) => setItemName(e.target.value)} placeholder="Item name" required />
        ) : (
          <input
            className="field"
            value={candidateItems}
            onChange={(e) => setCandidateItems(e.target.value)}
            placeholder="Candidate items (comma-separated)"
            required
          />
        )}
        <button className="btn w-full" disabled={loading}>
          {loading ? "Fetching logs..." : "Load Reasoning"}
        </button>
        {error && <p className="rounded-lg bg-red-500/15 p-2 text-sm text-red-300">{error}</p>}
      </form>

      <div className="glass rounded-2xl p-6">
        <h3 className="text-lg font-semibold">{result?.title ?? "Reasoning Steps"}</h3>
        {!result && <p className="mt-3 text-sm text-slate-400">Run any flow to inspect backend reasoning steps here.</p>}
        {result && (
          <div className="mt-4 space-y-3">
            {result.steps.map((step, index) => (
              <div key={`${step}-${index}`} className="rounded-xl bg-slate-900 p-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-brand-500">Step {index + 1}</p>
                <p className="mt-1 text-sm text-slate-300">{step}</p>
              </div>
            ))}
            {result.meta && (
              <pre className="overflow-auto rounded-xl bg-slate-900 p-3 text-xs text-slate-400">
                {JSON.stringify(result.meta, null, 2)}
              </pre>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
