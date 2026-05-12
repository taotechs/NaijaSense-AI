"use client";

import { FormEvent, useEffect, useState } from "react";
import {
  AgentGatewayResponse,
  postAgentGateway,
  UserPersonaPayload,
} from "@/lib/agent-api";

type BehavioralPreset = {
  id: string;
  label: string;
  description: string;
  location: string;
  interests: string;
  sentiment: string;
  tone_notes: string;
};

const PRESETS: BehavioralPreset[] = [
  {
    id: "lagos_foodie",
    label: "Lagos foodie (Naija tone)",
    description: "Street-food explorer, Twitter voice, balanced sentiment.",
    location: "Lagos",
    interests: "amala, suya, jollof, lounges",
    sentiment: "balanced",
    tone_notes: "Use Nigerian twitter tone with light pidgin.",
  },
  {
    id: "vi_lifestyle",
    label: "VI lifestyle critic",
    description: "Lekki / VI lounge regular with a critical eye.",
    location: "Victoria Island, Lagos",
    interests: "cocktails, lounges, fashion, afrobeats",
    sentiment: "critical",
    tone_notes: "Clear, natural English. Keep slang minimal.",
  },
  {
    id: "abuja_professional",
    label: "Abuja professional",
    description: "Formal, neutral, professional product evaluator.",
    location: "Abuja",
    interests: "tech, productivity, books",
    sentiment: "balanced",
    tone_notes: "Use plain English. No slang. Professional tone.",
  },
  {
    id: "campus_student",
    label: "Campus student",
    description: "Budget-conscious student, positive bias.",
    location: "Ibadan",
    interests: "buka food, budget snacks, study cafes",
    sentiment: "positive",
    tone_notes: "Casual but clear. Mild Nigerian colour OK.",
  },
];

const EXAMPLE_PROMPTS: string[] = [
  "Review the new Suya spot in Ikeja — went on Friday, queue was long but yaji on point.",
  "It’s 11 PM on a Saturday; where is the best place to get freshly made Akara or noodles near Yaba?",
  "Review of jollof rice from Iya Eba kitchen — soft amala, rich egusi, 20 min wait.",
  "Suggest things to do in Abuja this weekend on a 10k budget.",
];

const AGENT_STEPS = [
  "Routing intent",
  "Inferring persona",
  "Generating response",
  "Critique pass",
];

function StarRow({ rating }: { rating: number }) {
  const full = Math.round(rating);
  return (
    <div
      className="flex items-center gap-1 text-amber-400"
      aria-label={`Rating ${rating} of 5`}
    >
      {Array.from({ length: 5 }).map((_, i) => (
        <span key={i}>{i < full ? "\u2605" : "\u2606"}</span>
      ))}
      <span className="ml-2 text-sm text-slate-400">
        {rating.toFixed(1)} / 5
      </span>
    </div>
  );
}

function AgenticWorkflowIndicator({ active }: { active: boolean }) {
  const [step, setStep] = useState(0);
  useEffect(() => {
    if (!active) {
      setStep(0);
      return;
    }
    const id = setInterval(() => {
      setStep((s) => (s + 1) % AGENT_STEPS.length);
    }, 900);
    return () => clearInterval(id);
  }, [active]);

  if (!active) return null;
  return (
    <div
      className="flex items-center gap-3 rounded-xl border border-brand-500/40 bg-brand-500/10 px-4 py-3"
      role="status"
      aria-live="polite"
    >
      <span className="relative inline-flex h-2.5 w-2.5">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-brand-500 opacity-75"></span>
        <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-brand-500"></span>
      </span>
      <div className="flex flex-1 flex-wrap items-center gap-x-3 gap-y-1 text-sm">
        <span className="font-medium text-brand-500">Agent reasoning</span>
        {AGENT_STEPS.map((label, idx) => (
          <span
            key={label}
            className={`transition-colors ${
              idx === step
                ? "text-slate-100"
                : idx < step
                ? "text-slate-400"
                : "text-slate-600"
            }`}
          >
            {idx === step ? "\u25cf" : "\u25cb"} {label}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function UnifiedAgentPage() {
  const [userId, setUserId] = useState("naija_user_1");
  const [location, setLocation] = useState(PRESETS[0].location);
  const [interests, setInterests] = useState(PRESETS[0].interests);
  const [sentiment, setSentiment] = useState(PRESETS[0].sentiment);
  const [toneNotes, setToneNotes] = useState(PRESETS[0].tone_notes);
  const [history, setHistory] = useState("");
  const [presetId, setPresetId] = useState<string>(PRESETS[0].id);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [showIndicator, setShowIndicator] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AgentGatewayResponse | null>(null);

  function applyPreset(id: string) {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return;
    setPresetId(p.id);
    setLocation(p.location);
    setInterests(p.interests);
    setSentiment(p.sentiment);
    setToneNotes(p.tone_notes);
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setShowIndicator(true);
    setError(null);
    setResult(null);
    const startedAt = Date.now();
    const persona: UserPersonaPayload = {
      user_id: userId.trim() || "naija_user_1",
      location,
      interests: interests
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      sentiment_bias: sentiment,
      tone_notes: toneNotes.trim() || undefined,
      history: history.trim() || undefined,
    };
    try {
      const res = await postAgentGateway({
        user_persona: persona,
        query: q,
        top_k: 5,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed.");
    } finally {
      setLoading(false);
      // Keep the agentic workflow indicator visible long enough for the user
      // to actually observe the four-stage pipeline, even on fast responses.
      const elapsed = Date.now() - startedAt;
      const minVisibleMs = 1800;
      if (elapsed >= minVisibleMs) {
        setShowIndicator(false);
      } else {
        setTimeout(() => setShowIndicator(false), minVisibleMs - elapsed);
      }
    }
  }

  const critiqueStep = result?.reasoning_steps.find((s) =>
    s.toLowerCase().includes("critique")
  );

  return (
    <div className="space-y-8">
      <section>
        <p className="text-sm leading-relaxed text-slate-300">
          Behavioral user modeling and contextual recommendation in one agentic
          gateway. Type any question or product experience &mdash; the system
          routes between <span className="text-brand-500">Task A</span> (review
          simulation) and <span className="text-brand-500">Task B</span>{" "}
          (recommendation), grounded in retrieval over real Yelp, Amazon, and
          Goodreads reviews.
        </p>
      </section>

      <form onSubmit={onSubmit} className="glass space-y-4 rounded-2xl p-6">
        <textarea
          className="field min-h-28"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Simulate a review for a Nigerian spot or ask for personalized recommendations..."
          required
        />

        <div className="flex flex-wrap gap-2">
          <span className="text-xs uppercase tracking-wider text-slate-500">
            Try
          </span>
          {EXAMPLE_PROMPTS.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => setQuery(ex)}
              className="rounded-full border border-slate-700 bg-slate-900/70 px-3 py-1 text-xs text-slate-300 transition hover:border-brand-500 hover:text-brand-500"
            >
              {ex.length > 56 ? ex.slice(0, 53) + "\u2026" : ex}
            </button>
          ))}
        </div>

        <details className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 text-sm">
          <summary className="cursor-pointer text-slate-300">
            <span className="font-medium text-brand-500">
              Behavioral profile
            </span>{" "}
            <span className="text-slate-500">
              &mdash; tone, sentiment bias, interests (Task A user modeling)
            </span>
          </summary>
          <div className="mt-4 space-y-4">
            <div>
              <label
                htmlFor="preset"
                className="mb-1 block text-xs uppercase tracking-wider text-slate-500"
              >
                Quick preset
              </label>
              <select
                id="preset"
                className="field"
                value={presetId}
                onChange={(e) => applyPreset(e.target.value)}
              >
                {PRESETS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label} &mdash; {p.description}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <label className="mb-1 block text-xs uppercase tracking-wider text-slate-500">
                  User ID
                </label>
                <input
                  className="field"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                />
              </div>
              <div>
                <label className="mb-1 block text-xs uppercase tracking-wider text-slate-500">
                  Location
                </label>
                <input
                  className="field"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                />
              </div>
            </div>

            <div>
              <label className="mb-1 block text-xs uppercase tracking-wider text-slate-500">
                Interests (comma-separated)
              </label>
              <input
                className="field"
                value={interests}
                onChange={(e) => setInterests(e.target.value)}
              />
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <label className="mb-1 block text-xs uppercase tracking-wider text-slate-500">
                  Sentiment bias
                </label>
                <select
                  className="field"
                  value={sentiment}
                  onChange={(e) => setSentiment(e.target.value)}
                >
                  <option value="positive">Positive</option>
                  <option value="balanced">Balanced</option>
                  <option value="critical">Critical</option>
                </select>
              </div>
              <div>
                <label className="mb-1 block text-xs uppercase tracking-wider text-slate-500">
                  Tone / style notes
                </label>
                <input
                  className="field"
                  value={toneNotes}
                  onChange={(e) => setToneNotes(e.target.value)}
                />
              </div>
            </div>

            <div>
              <label className="mb-1 block text-xs uppercase tracking-wider text-slate-500">
                Background &amp; history (optional)
              </label>
              <textarea
                className="field min-h-20"
                value={history}
                onChange={(e) => setHistory(e.target.value)}
                placeholder="Paste prior reviews, preferences, or a short profile narrative."
              />
            </div>
          </div>
        </details>

        <button className="btn w-full" disabled={loading || !query.trim()}>
          {loading ? "Routing\u2026" : "Send to agent"}
        </button>
        {error && <p className="text-sm text-red-300">{error}</p>}
      </form>

      <AgenticWorkflowIndicator active={showIndicator} />

      {result && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2 rounded-xl border border-slate-800 bg-slate-950/80 px-4 py-3 text-sm">
            <span className="rounded-full bg-brand-500/15 px-2 py-0.5 text-xs font-medium uppercase tracking-wider text-brand-500">
              Task {result.task === "review" ? "A" : "B"} &middot; {result.task}
            </span>
            <span className="rounded-full border border-slate-700 px-2 py-0.5 text-xs text-slate-400">
              Router: {result.routing_source}
            </span>
            {critiqueStep && (
              <span
                className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-xs text-amber-300"
                title={critiqueStep}
              >
                Critique applied
              </span>
            )}
            <p className="basis-full text-slate-400">
              {result.orchestrator_rationale}
            </p>
          </div>

          {result.task === "review" && result.review && (
            <article className="glass rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-slate-100">
                Simulated review
              </h3>
              <div className="mt-3">
                <StarRow rating={result.review.rating} />
              </div>
              <p className="mt-4 whitespace-pre-wrap text-slate-200">
                {result.review.review_text}
              </p>
            </article>
          )}

          {result.task === "recommend" && result.recommendation && (
            <article className="glass rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-slate-100">
                Recommendations
              </h3>
              {result.recommendation.conversational_response && (
                <p className="mt-3 rounded-xl bg-brand-500/10 p-3 text-sm text-brand-500">
                  {result.recommendation.conversational_response}
                </p>
              )}
              <ul className="mt-4 space-y-3">
                {result.recommendation.recommendations.map((item) => (
                  <li
                    key={item.item_name}
                    className="rounded-xl bg-slate-900/80 p-4"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium text-slate-100">
                        {item.item_name}
                      </span>
                      <span className="text-sm text-brand-500">
                        {item.score.toFixed(2)}
                      </span>
                    </div>
                    <p className="mt-2 text-sm text-slate-400">
                      {item.explanation}
                    </p>
                  </li>
                ))}
              </ul>
            </article>
          )}

          <details className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 text-sm">
            <summary className="cursor-pointer text-slate-300">
              <span className="font-medium text-brand-500">
                Agentic reasoning trace
              </span>{" "}
              <span className="text-slate-500">
                &mdash; {result.reasoning_steps.length} step
                {result.reasoning_steps.length === 1 ? "" : "s"}
              </span>
            </summary>
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
