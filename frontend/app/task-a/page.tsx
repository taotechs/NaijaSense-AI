"use client";

import Link from "next/link";
import { FormEvent, useState } from "react";
import { BackendStatus } from "@/components/BackendStatus";
import { publicTaskUrl } from "@/lib/api-root";
import { TASK_A_PRESETS } from "@/lib/task-a-presets";
import { TaskAResponse, postTaskA } from "@/lib/task-api";

function StarRow({ rating }: { rating: number }) {
  const full = Math.round(rating);
  return (
    <div className="flex items-center gap-1 text-amber-400" aria-label={`Rating ${rating} of 5`}>
      {Array.from({ length: 5 }).map((_, i) => (
        <span key={i} className="text-xl">
          {i < full ? "\u2605" : "\u2606"}
        </span>
      ))}
      <span className="ml-2 text-base font-medium text-slate-200">{rating.toFixed(1)} / 5</span>
    </div>
  );
}

export default function TaskAPage() {
  const defaultPreset = TASK_A_PRESETS[0];
  const [presetId, setPresetId] = useState(defaultPreset.id);
  const [userPersona, setUserPersona] = useState(defaultPreset.user_persona);
  const [productDetails, setProductDetails] = useState(defaultPreset.product_details);

  const [result, setResult] = useState<TaskAResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const endpoint = publicTaskUrl("/task-a/user-modeling");

  function onPresetChange(id: string) {
    const preset = TASK_A_PRESETS.find((p) => p.id === id);
    if (!preset) return;
    setPresetId(id);
    setUserPersona(preset.user_persona);
    setProductDetails(preset.product_details);
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await postTaskA({
        user_persona: userPersona.trim(),
        product_details: productDetails.trim(),
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
            User persona + product details → star rating and full review text.
          </p>
          <p className="mt-2 break-all font-mono text-[10px] text-slate-500">
            POST {endpoint}
          </p>
        </div>
        <BackendStatus />
      </div>

      <section className="grid gap-6 lg:grid-cols-2">
        <form onSubmit={onSubmit} className="glass space-y-4 rounded-2xl p-6">
          <label className="block text-xs text-slate-400">
            Preset (optional)
            <select
              className="field mt-1"
              value={presetId}
              onChange={(e) => onPresetChange(e.target.value)}
            >
              {TASK_A_PRESETS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
              <option value="custom">Custom</option>
            </select>
          </label>

          <label className="block text-sm font-medium text-slate-200">
            User Persona
            <textarea
              className="field mt-2 min-h-36"
              value={userPersona}
              onChange={(e) => {
                setUserPersona(e.target.value);
                setPresetId("custom");
              }}
              placeholder="Who is this user? Location, budget, tone, preferences…"
              required
              minLength={20}
            />
          </label>

          <label className="block text-sm font-medium text-slate-200">
            Product Details
            <textarea
              className="field mt-2 min-h-32"
              value={productDetails}
              onChange={(e) => {
                setProductDetails(e.target.value);
                setPresetId("custom");
              }}
              placeholder="What product or experience? Price, wait, taste, service…"
              required
              minLength={10}
            />
          </label>

          <button type="submit" className="btn w-full" disabled={loading}>
            {loading ? "Generating…" : "Run Task A"}
          </button>
          {error && <p className="rounded-lg bg-red-500/15 p-2 text-sm text-red-300">{error}</p>}
        </form>

        <div className="glass space-y-4 rounded-2xl p-6">
          <h3 className="text-lg font-semibold text-slate-100">Output</h3>
          {!result && (
            <p className="text-sm text-slate-400">
              Submit persona and product details to see the star rating and review.
            </p>
          )}
          {result && (
            <div className="space-y-4">
              <div>
                <p className="mb-2 text-xs uppercase tracking-wide text-slate-500">Star rating</p>
                <StarRow rating={result.rating} />
              </div>
              <div>
                <p className="mb-2 text-xs uppercase tracking-wide text-slate-500">Review</p>
                <p className="rounded-xl bg-slate-900/80 p-4 text-sm leading-relaxed text-slate-200">
                  {result.review_text}
                </p>
              </div>
            </div>
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
      </p>
    </div>
  );
}
