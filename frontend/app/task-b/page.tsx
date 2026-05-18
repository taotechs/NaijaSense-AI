"use client";

import Link from "next/link";
import { FormEvent, useState } from "react";
import { BackendStatus } from "@/components/BackendStatus";
import { publicTaskUrl } from "@/lib/api-root";
import { TASK_B_PERSONA_PRESETS } from "@/lib/task-b-personas";
import { TaskBResponse, postTaskB } from "@/lib/task-api";

export default function TaskBPage() {
  const defaultPreset = TASK_B_PERSONA_PRESETS[0];
  const [presetId, setPresetId] = useState(defaultPreset.id);
  const [userId, setUserId] = useState(defaultPreset.user_id);
  const [persona, setPersona] = useState(defaultPreset.persona);

  const [result, setResult] = useState<TaskBResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const endpoint = publicTaskUrl("/task-b/recommendation");

  function onPresetChange(id: string) {
    const preset = TASK_B_PERSONA_PRESETS.find((p) => p.id === id);
    if (!preset) return;
    setPresetId(id);
    setUserId(preset.user_id);
    setPersona(preset.persona);
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await postTaskB({
        user_persona: {
          user_id: userId.trim(),
          persona: persona.trim(),
        },
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
            User persona only → personalized picks across Food, Movies, Drinks, and more.
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

          <label className="block text-xs text-slate-400">
            Preset profile
            <select
              className="field mt-1"
              value={presetId}
              onChange={(e) => onPresetChange(e.target.value)}
            >
              {TASK_B_PERSONA_PRESETS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
              <option value="custom">Custom (edit below)</option>
            </select>
          </label>

          <input
            className="field"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="User ID"
            required
          />

          <textarea
            className="field min-h-40"
            value={persona}
            onChange={(e) => {
              setPersona(e.target.value);
              setPresetId("custom");
            }}
            placeholder="Describe lifestyle, budget, location, and tastes (food, movies, drinks, tech…)"
            required
            minLength={20}
          />

          <button type="submit" className="btn w-full" disabled={loading}>
            {loading ? "Ranking…" : "Run Task B"}
          </button>
          {error && <p className="rounded-lg bg-red-500/15 p-2 text-sm text-red-300">{error}</p>}
        </form>

        <div className="glass space-y-4 rounded-2xl p-6">
          <h3 className="text-lg font-semibold">Output</h3>
          {!result && (
            <p className="text-sm text-slate-400">
              Submit a persona to see corpus-backed recommendations with LLM confidence scores.
            </p>
          )}
          {result && (
            <>
              <details className="rounded-xl border border-slate-800 bg-slate-900/50 p-3" open>
                <summary className="cursor-pointer text-sm font-medium text-slate-300">
                  agent_reasoning
                </summary>
                <p className="mt-2 whitespace-pre-wrap text-xs text-slate-400">
                  {result.agent_reasoning}
                </p>
              </details>
              <ul className="space-y-3">
                {result.recommendations.map((item, idx) => (
                  <li
                    key={item.item_id}
                    className="rounded-xl border border-slate-800 bg-slate-900/80 p-3"
                  >
                    <p className="text-sm font-medium text-slate-100">
                      #{idx + 1} {item.title}
                    </p>
                    <p className="mt-1 text-xs text-slate-400">
                      <span className="font-medium text-brand-400">Category: {item.domain}</span>
                      {" · "}
                      <span>Confidence: {(item.confidence_score * 100).toFixed(1)}%</span>
                    </p>
                  </li>
                ))}
              </ul>
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
