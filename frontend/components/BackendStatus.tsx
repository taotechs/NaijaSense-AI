"use client";

/**
 * BackendStatus
 * -------------
 * Fires a fire-and-forget GET to the backend's /api/v1/health endpoint
 * on mount (which doubles as a cold-start pre-warm on Koyeb's free tier)
 * and exposes the result as a tiny pill the user can see at a glance.
 *
 * States:
 *   - "checking"   - initial; we haven't heard back yet.
 *   - "ready"      - backend returned 200 within the timeout.
 *   - "waking"     - the first request took >2s; we're showing this so the
 *                    user knows why "Send to agent" might lag on the very
 *                    first interaction after the service has been idle.
 *   - "down"       - the request errored out / timed out.
 *
 * Re-checks every 60s while the page is visible, so a recovered backend
 * surfaces without a page reload.
 */

import { useEffect, useRef, useState } from "react";
import { getBackendHealth } from "@/lib/agent-api";

type Status = "checking" | "waking" | "ready" | "down";

const POLL_MS = 60_000;
const WAKING_THRESHOLD_MS = 2_000;

export function BackendStatus() {
  const [status, setStatus] = useState<Status>("checking");
  const [latency, setLatency] = useState<number | null>(null);
  const wakingTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function check() {
      const started = performance.now();
      // After ~2s without a response we tentatively flip to "waking" so the
      // user gets a hint that this is normal cold-start behaviour.
      wakingTimer.current = setTimeout(() => {
        if (!cancelled) setStatus((s) => (s === "checking" ? "waking" : s));
      }, WAKING_THRESHOLD_MS);
      try {
        await getBackendHealth();
        const elapsed = Math.round(performance.now() - started);
        if (cancelled) return;
        setStatus("ready");
        setLatency(elapsed);
      } catch {
        if (cancelled) return;
        setStatus("down");
        setLatency(null);
      } finally {
        if (wakingTimer.current) clearTimeout(wakingTimer.current);
      }
    }

    check();
    const interval = setInterval(check, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
      if (wakingTimer.current) clearTimeout(wakingTimer.current);
    };
  }, []);

  const meta = MAP[status];
  const label =
    status === "ready" && latency !== null
      ? `${meta.label} · ${latency}ms`
      : meta.label;

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium ${meta.className}`}
      title={meta.title}
      aria-live="polite"
    >
      <span
        className={`relative inline-flex h-1.5 w-1.5 ${meta.dotPulse ? "" : ""}`}
      >
        {meta.dotPulse && (
          <span
            className={`absolute inline-flex h-full w-full animate-ping rounded-full ${meta.dotColor} opacity-75`}
          ></span>
        )}
        <span
          className={`relative inline-flex h-1.5 w-1.5 rounded-full ${meta.dotColor}`}
        ></span>
      </span>
      {label}
    </span>
  );
}

const MAP: Record<
  Status,
  {
    label: string;
    title: string;
    className: string;
    dotColor: string;
    dotPulse: boolean;
  }
> = {
  checking: {
    label: "checking backend",
    title: "Pinging the backend health endpoint.",
    className: "border-slate-700 bg-slate-900/60 text-slate-400",
    dotColor: "bg-slate-400",
    dotPulse: true,
  },
  waking: {
    label: "backend waking up…",
    title:
      "Free-tier backend is cold-starting. First request can take 20-60 seconds.",
    className: "border-amber-500/40 bg-amber-500/10 text-amber-300",
    dotColor: "bg-amber-400",
    dotPulse: true,
  },
  ready: {
    label: "backend ready",
    title: "Backend is reachable. Round-trip latency shown.",
    className: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
    dotColor: "bg-emerald-400",
    dotPulse: false,
  },
  down: {
    label: "backend unreachable",
    title: "Health check failed. Requests will likely error.",
    className: "border-rose-500/40 bg-rose-500/10 text-rose-300",
    dotColor: "bg-rose-400",
    dotPulse: true,
  },
};
