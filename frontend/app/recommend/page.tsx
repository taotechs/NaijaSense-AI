"use client";

import { FormEvent, useState } from "react";
import { RecommendationResponse, recommendItems } from "@/lib/api";

export default function RecommendPage() {
  const [userId, setUserId] = useState("user_demo_1");
  const [location, setLocation] = useState("Abuja");
  const [interests, setInterests] = useState("tech, food");
  const [candidateItems, setCandidateItems] = useState("Budget Earbuds, Foodie Hub, Formal Shoes");
  const [context, setContext] = useState("I want practical daily options.");
  const [topK, setTopK] = useState(3);
  const [personality, setPersonality] = useState<"analyst" | "coach" | "friend" | "nigerian_twitter">("friend");

  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await recommendItems({
        user_profile: {
          user_id: userId.trim(),
          location,
          interests: interests
            .split(",")
            .map((i) => i.trim())
            .filter(Boolean)
        },
        candidate_items: candidateItems
          .split(",")
          .map((i) => i.trim())
          .filter(Boolean),
        context,
        top_k: topK,
        recommender_personality: personality,
        conversational_mode: true
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load recommendations.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="grid gap-6 lg:grid-cols-2">
      <form onSubmit={onSubmit} className="glass space-y-4 rounded-2xl p-6">
        <h2 className="text-xl font-semibold">Recommendation Engine</h2>
        <input className="field" value={userId} onChange={(e) => setUserId(e.target.value)} required placeholder="User ID" />
        <input className="field" value={location} onChange={(e) => setLocation(e.target.value)} placeholder="Location" />
        <input className="field" value={interests} onChange={(e) => setInterests(e.target.value)} placeholder="Interests (comma-separated)" />
        <input className="field" value={candidateItems} onChange={(e) => setCandidateItems(e.target.value)} placeholder="Candidate items (comma-separated)" />
        <textarea className="field min-h-20" value={context} onChange={(e) => setContext(e.target.value)} placeholder="Context for recommendation" />
        <div className="grid grid-cols-2 gap-3">
          <input className="field" type="number" min={1} max={10} value={topK} onChange={(e) => setTopK(Number(e.target.value || 3))} />
          <select className="field" value={personality} onChange={(e) => setPersonality(e.target.value as typeof personality)}>
            <option value="analyst">Analyst</option>
            <option value="coach">Coach</option>
            <option value="friend">Friend</option>
            <option value="nigerian_twitter">Nigerian Twitter</option>
          </select>
        </div>
        <button className="btn w-full" disabled={loading}>
          {loading ? "Ranking..." : "Get Recommendations"}
        </button>
        {error && <p className="rounded-lg bg-red-500/15 p-2 text-sm text-red-300">{error}</p>}
      </form>

      <div className="glass rounded-2xl p-6">
        <h3 className="text-lg font-semibold">Recommended Items</h3>
        {!result && <p className="mt-3 text-sm text-slate-400">Run recommendation to view ranked items with explanations.</p>}
        {result && (
          <div className="mt-4 space-y-3">
            {result.conversational_response && (
              <p className="rounded-xl bg-brand-500/10 p-3 text-sm text-brand-500">{result.conversational_response}</p>
            )}
            {result.recommendations.map((item) => (
              <div key={item.item_name} className="rounded-xl bg-slate-900 p-4">
                <div className="flex items-center justify-between">
                  <p className="font-semibold">{item.item_name}</p>
                  <span className="text-sm text-brand-500">{item.score.toFixed(2)}</span>
                </div>
                <p className="mt-2 text-sm text-slate-400">{item.explanation}</p>
              </div>
            ))}
            {result.explainability && (
              <pre className="overflow-auto rounded-xl bg-slate-900 p-3 text-xs text-slate-400">
                {JSON.stringify(result.explainability, null, 2)}
              </pre>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
