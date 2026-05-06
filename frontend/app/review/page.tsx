"use client";

import { FormEvent, useState } from "react";
import { SimulateReviewResponse, simulateReview } from "@/lib/api";

export default function ReviewPage() {
  const [userId, setUserId] = useState("user_demo_1");
  const [location, setLocation] = useState("Lagos");
  const [interests, setInterests] = useState("food, lifestyle");
  const [sentimentBias, setSentimentBias] = useState("positive");
  const [personaStyle, setPersonaStyle] = useState("nigerian_twitter");
  const [itemName, setItemName] = useState("Amala Spot");
  const [itemContext, setItemContext] = useState("Service was quick and food was hot.");

  const [result, setResult] = useState<SimulateReviewResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await simulateReview({
        user_profile: {
          user_id: userId.trim(),
          location,
          interests: interests
            .split(",")
            .map((i) => i.trim())
            .filter(Boolean),
          sentiment_bias: sentimentBias
        },
        item_data: {
          item_name: itemName.trim(),
          item_context: itemContext
        },
        persona_style: personaStyle
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="grid gap-6 lg:grid-cols-2">
      <form onSubmit={onSubmit} className="glass space-y-4 rounded-2xl p-6">
        <h2 className="text-xl font-semibold">Review Simulation</h2>
        <p className="text-sm text-slate-400">Input user profile and item context.</p>

        <input className="field" value={userId} onChange={(e) => setUserId(e.target.value)} placeholder="User ID" required />
        <input className="field" value={location} onChange={(e) => setLocation(e.target.value)} placeholder="Location" />
        <input className="field" value={interests} onChange={(e) => setInterests(e.target.value)} placeholder="Interests (comma-separated)" />
        <select className="field" value={sentimentBias} onChange={(e) => setSentimentBias(e.target.value)}>
          <option value="positive">Positive</option>
          <option value="balanced">Balanced</option>
          <option value="critical">Critical</option>
        </select>
        <select className="field" value={personaStyle} onChange={(e) => setPersonaStyle(e.target.value)}>
          <option value="nigerian_twitter">Nigerian Twitter</option>
          <option value="formal">Formal</option>
        </select>

        <input className="field" value={itemName} onChange={(e) => setItemName(e.target.value)} placeholder="Item name" required />
        <textarea className="field min-h-24" value={itemContext} onChange={(e) => setItemContext(e.target.value)} placeholder="Item context" />

        <button className="btn w-full" disabled={loading}>
          {loading ? "Generating..." : "Generate Review"}
        </button>
        {error && <p className="rounded-lg bg-red-500/15 p-2 text-sm text-red-300">{error}</p>}
      </form>

      <div className="glass rounded-2xl p-6">
        <h3 className="text-lg font-semibold">Generated Output</h3>
        {!result && <p className="mt-3 text-sm text-slate-400">Submit the form to see generated review and persona breakdown.</p>}
        {result && (
          <div className="mt-4 space-y-4">
            <div className="rounded-xl bg-slate-900 p-4">
              <p className="text-sm text-slate-300">{result.review_text}</p>
              <p className="mt-2 text-sm font-semibold text-brand-500">Rating: {result.rating.toFixed(1)}/5</p>
            </div>
            <div className="rounded-xl bg-slate-900 p-4">
              <p className="text-sm font-semibold text-slate-200">Persona Breakdown</p>
              <pre className="mt-2 overflow-auto text-xs text-slate-400">
                {JSON.stringify(result.persona_breakdown, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
