"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  AgentGatewayResponse,
  AgentLanguage,
  StreamEvent,
  UserPersonaPayload,
  postFeedback,
  streamAgentGateway,
} from "@/lib/agent-api";
import { BackendStatus } from "@/components/BackendStatus";
import {
  ReasoningTimeline,
  TimelineNode,
  buildNodesFromSteps,
} from "@/components/ReasoningTimeline";

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
  "Review the new Suya spot in Ikeja - went on Friday, queue was long but yaji on point.",
  "It's 11 PM on a Saturday; where is the best place to get freshly made Akara or noodles near Yaba?",
  "Review of jollof rice from Iya Eba kitchen - soft amala, rich egusi, 20 min wait.",
  "Suggest things to do in Abuja this weekend on a 10k budget.",
];

const LANGUAGE_OPTIONS: Array<{ value: AgentLanguage; label: string }> = [
  { value: "english", label: "English" },
  { value: "pidgin", label: "Nigerian Pidgin" },
  { value: "yoruba_mix", label: "English + Yoruba mix" },
];

const SAFETY_FLAG_COPY: Record<string, string> = {
  prompt_injection_suspected: "Possible prompt-injection language in input.",
  pii_email_in_input: "Email detected in input - consider redacting.",
  pii_phone_in_input: "Phone number detected in input - consider redacting.",
  pii_bvn_in_input: "BVN detected in input - REDACT before resubmitting.",
  pii_email_in_output: "Output contains an email-like token.",
  pii_phone_in_output: "Output contains a phone-like token.",
  pii_bvn_in_output: "Output contains a BVN-like token.",
  ungrounded_numeric_specifics:
    "Output contains numeric facts not present in the input - verify before sharing.",
  compare_variant_failed: "Comparison variant failed; main result is shown.",
};

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

function SafetyFlagsRow({ flags }: { flags: string[] }) {
  if (!flags || flags.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
      <span className="font-semibold uppercase tracking-wider text-amber-300">
        Safety advisories
      </span>
      {flags.map((flag) => (
        <span
          key={flag}
          title={SAFETY_FLAG_COPY[flag] || flag}
          className="rounded-full border border-amber-500/40 bg-amber-500/15 px-2 py-0.5"
        >
          {flag.replace(/_/g, " ")}
        </span>
      ))}
    </div>
  );
}

function ThumbsRow({
  taskKey,
  payload,
}: {
  taskKey: string;
  payload: {
    user_id: string;
    task: "review" | "recommend";
    query: string;
    output_preview: string;
    routing_source?: string;
    language?: AgentLanguage;
  };
}) {
  const [sent, setSent] = useState<null | 1 | -1>(null);
  const [error, setError] = useState<string | null>(null);

  async function vote(rating: 1 | -1) {
    if (sent === rating) return;
    setError(null);
    try {
      await postFeedback({
        user_id: payload.user_id,
        task: payload.task,
        rating,
        query: payload.query,
        output_preview: payload.output_preview.slice(0, 4000),
        routing_source: payload.routing_source,
        language: payload.language,
      });
      setSent(rating);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Feedback failed.");
    }
  }

  return (
    <div className="flex items-center gap-2 text-xs text-slate-500" key={taskKey}>
      <span>Helpful?</span>
      <button
        type="button"
        onClick={() => vote(1)}
        className={`rounded-full border px-2 py-0.5 transition ${
          sent === 1
            ? "border-emerald-500/50 bg-emerald-500/15 text-emerald-300"
            : "border-slate-700 hover:border-emerald-500/60 hover:text-emerald-300"
        }`}
        aria-pressed={sent === 1}
        aria-label="thumbs up"
      >
        {"\u{1F44D}"} Yes
      </button>
      <button
        type="button"
        onClick={() => vote(-1)}
        className={`rounded-full border px-2 py-0.5 transition ${
          sent === -1
            ? "border-rose-500/50 bg-rose-500/15 text-rose-300"
            : "border-slate-700 hover:border-rose-500/60 hover:text-rose-300"
        }`}
        aria-pressed={sent === -1}
        aria-label="thumbs down"
      >
        {"\u{1F44E}"} No
      </button>
      {sent && <span className="text-slate-600">thanks - logged</span>}
      {error && <span className="text-rose-400">{error}</span>}
    </div>
  );
}

function AgentResultCard({
  result,
  variantLabel,
  userId,
  query,
}: {
  result: AgentGatewayResponse;
  variantLabel?: string;
  userId: string;
  query: string;
}) {
  const critiqueStep = result.reasoning_steps.find((s) =>
    s.toLowerCase().includes("critique")
  );
  const outputPreview =
    result.task === "review" && result.review
      ? result.review.review_text
      : result.task === "recommend" && result.recommendation
      ? result.recommendation.conversational_response ||
        result.recommendation.recommendations
            .map((r) => `${r.item_name}: ${r.explanation}`)
            .join(" | ")
      : "";

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 rounded-xl border border-slate-800 bg-slate-950/80 px-4 py-3 text-sm">
        {variantLabel && (
          <span className="rounded-full border border-slate-600 bg-slate-800/70 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-slate-300">
            {variantLabel}
          </span>
        )}
        <span className="rounded-full bg-brand-500/15 px-2 py-0.5 text-xs font-medium uppercase tracking-wider text-brand-500">
          Task {result.task === "review" ? "A" : "B"} &middot; {result.task}
        </span>
        <span className="rounded-full border border-slate-700 px-2 py-0.5 text-xs text-slate-400">
          Router: {result.routing_source}
        </span>
        {result.language && result.language !== "english" && (
          <span className="rounded-full border border-purple-500/40 bg-purple-500/10 px-2 py-0.5 text-xs text-purple-300">
            Lang: {result.language}
          </span>
        )}
        {typeof result.timing_ms === "number" && (
          <span className="rounded-full border border-slate-700 px-2 py-0.5 text-xs text-slate-400">
            {result.timing_ms}ms
          </span>
        )}
        {critiqueStep && (
          <span
            className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-xs text-amber-300"
            title={critiqueStep}
          >
            Critique applied
          </span>
        )}
        <p className="basis-full text-slate-400">{result.orchestrator_rationale}</p>
      </div>

      <SafetyFlagsRow flags={result.safety_flags || []} />

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
          <div className="mt-4">
            <ThumbsRow
              taskKey={`review-${variantLabel ?? "main"}`}
              payload={{
                user_id: userId,
                task: "review",
                query,
                output_preview: result.review.review_text,
                routing_source: result.routing_source,
                language: result.language,
              }}
            />
          </div>
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
                <p className="mt-2 text-sm text-slate-400">{item.explanation}</p>
              </li>
            ))}
          </ul>
          <div className="mt-4">
            <ThumbsRow
              taskKey={`recommend-${variantLabel ?? "main"}`}
              payload={{
                user_id: userId,
                task: "recommend",
                query,
                output_preview: outputPreview,
                routing_source: result.routing_source,
                language: result.language,
              }}
            />
          </div>
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
        <div className="mt-3">
          <ReasoningTimeline nodes={buildNodesFromSteps(result.reasoning_steps)} />
          <ol className="mt-4 list-decimal space-y-1 pl-5 text-xs text-slate-500">
            {result.reasoning_steps.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ol>
        </div>
      </details>
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
  const [language, setLanguage] = useState<AgentLanguage>("english");
  const [includeHistory, setIncludeHistory] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AgentGatewayResponse | null>(null);
  const [liveNodes, setLiveNodes] = useState<TimelineNode[]>([]);

  const abortRef = useRef<AbortController | null>(null);

  // Cleanup any in-flight stream on unmount.
  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  function applyPreset(id: string) {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return;
    setPresetId(p.id);
    setLocation(p.location);
    setInterests(p.interests);
    setSentiment(p.sentiment);
    setToneNotes(p.tone_notes);
  }

  function applyStreamEvent(ev: StreamEvent) {
    setLiveNodes((prev) => {
      if (ev.type === "plan") {
        // Initialise a node per planned step (all pending).
        return ev.steps.map((step, idx) => ({
          id: `${step}-${idx}`,
          step,
          label: "pending\u2026",
          status: "pending" as const,
        }));
      }
      if (ev.type === "step_start") {
        return prev.map((n) =>
          n.step === ev.step && n.status !== "done"
            ? { ...n, status: "active" as const, label: "running\u2026" }
            : n.status === "active"
            ? { ...n, status: "done" as const, label: "complete" }
            : n
        );
      }
      if (ev.type === "step_end") {
        return prev.map((n) =>
          n.step === ev.step ? { ...n, status: "done" as const, label: "complete" } : n
        );
      }
      return prev;
    });
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setLiveNodes([]);
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

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
      language,
    };

    try {
      const res = await streamAgentGateway(
        {
          user_persona: persona,
          query: q,
          top_k: 5,
          include_history: includeHistory,
        },
        applyStreamEvent,
        { signal: controller.signal }
      );
      setResult(res);
      // Make sure all timeline nodes show "done" after a successful run,
      // even if step_end events were coalesced or skipped.
      setLiveNodes((prev) =>
        prev.map((n) => ({ ...n, status: "done" as const, label: "complete" }))
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed.");
    } finally {
      setLoading(false);
    }
  }

  const activeNodeIdx = useMemo(
    () => liveNodes.findIndex((n) => n.status === "active"),
    [liveNodes]
  );

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm leading-relaxed text-slate-300">
          Behavioral user modeling and contextual recommendation in one agentic
          gateway. Type any question or product experience &mdash; the system
          routes between <span className="text-brand-500">Task A</span> (review
          simulation) and <span className="text-brand-500">Task B</span>{" "}
          (recommendation), grounded in retrieval over real Yelp, Amazon, and
          Goodreads reviews.
        </p>
        <BackendStatus />
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

        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <label className="mb-1 block text-xs uppercase tracking-wider text-slate-500">
              Output language
            </label>
            <select
              className="field"
              value={language}
              onChange={(e) => setLanguage(e.target.value as AgentLanguage)}
            >
              {LANGUAGE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          <label className="flex cursor-pointer items-end gap-2 text-xs text-slate-300">
            <input
              type="checkbox"
              checked={includeHistory}
              onChange={(e) => setIncludeHistory(e.target.checked)}
              className="h-4 w-4 rounded border-slate-600 bg-slate-900 text-brand-500 focus:ring-brand-500"
            />
            Use silent history (Task A baseline)
          </label>
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
          {loading ? "Streaming\u2026" : "Send to agent"}
        </button>
        {error && <p className="text-sm text-red-300">{error}</p>}
      </form>

      {(loading || liveNodes.length > 0) && (
        <div className="glass rounded-2xl p-5">
          <div className="mb-3 flex items-center justify-between text-xs uppercase tracking-wider text-brand-500">
            <span>Live agent trace</span>
            <span className="text-slate-500">
              {activeNodeIdx >= 0
                ? `step ${activeNodeIdx + 1} of ${liveNodes.length}`
                : loading
                ? "starting\u2026"
                : "complete"}
            </span>
          </div>
          <ReasoningTimeline
            nodes={liveNodes}
            emptyHint="Waiting for the first reasoning step\u2026"
          />
        </div>
      )}

      {result && (
        <AgentResultCard result={result} userId={userId} query={query} />
      )}
    </div>
  );
}
