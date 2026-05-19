"use client";

/**
 * ReasoningTimeline
 * -----------------
 * Renders the agent's reasoning as a visual timeline rather than a list of
 * bullet points. Two modes:
 *
 *   - "live"    - fed by the NDJSON stream from /api/agent/v1/stream;
 *                 each ``step_start`` / ``step_end`` mutates the visible
 *                 list of nodes so the user watches the agent think.
 *   - "final"   - fed by ``result.reasoning_steps`` once the request has
 *                 settled. Used to render the trace after the fact.
 *
 * The component is intentionally presentation-only - the parent owns
 * the state. This keeps it usable on the unified page (live) AND on
 * the existing "Agentic reasoning trace" disclosure (final).
 */

import { useMemo } from "react";

export type TimelineNode = {
  id: string;
  step: string;
  label: string;
  status: "pending" | "active" | "done";
  detail?: string;
};

const STEP_LABELS: Record<string, { label: string; icon: string }> = {
  silent_context_retrieval: {
    label: "Silent context retrieval",
    icon: "search",
  },
  reason_about_persona_strategy: {
    label: "Reason about persona strategy",
    icon: "brain",
  },
  build_persona_from_profile_and_history: {
    label: "Build persona from profile + history",
    icon: "user",
  },
  generate_review_with_persona_tone: {
    label: "Generate review with persona tone",
    icon: "pen",
  },
  persist_review_to_memory: {
    label: "Persist review to memory",
    icon: "save",
  },
  reason_about_retrieval_strategy: {
    label: "Reason about retrieval strategy",
    icon: "brain",
  },
  retrieve_relevant_user_memory: {
    label: "Retrieve relevant user memory",
    icon: "search",
  },
  run_recommendation_ranking: {
    label: "Run recommendation ranking",
    icon: "rank",
  },
  return_ranked_output: {
    label: "Return ranked output",
    icon: "check",
  },
};

function describe(step: string): { label: string; icon: string } {
  return STEP_LABELS[step] ?? { label: step.replace(/_/g, " "), icon: "dot" };
}

function Icon({ name }: { name: string }) {
  // Inline SVG so we don't pull a whole icon library for five glyphs.
  const common = {
    width: 14,
    height: 14,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
  };
  switch (name) {
    case "search":
      return (
        <svg {...common}>
          <circle cx="11" cy="11" r="7" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
      );
    case "brain":
      return (
        <svg {...common}>
          <path d="M9 3a3 3 0 0 0-3 3 3 3 0 0 0-3 3v4a3 3 0 0 0 3 3 3 3 0 0 0 3 3h0a3 3 0 0 0 3-3v-12a3 3 0 0 0-3-3z" />
          <path d="M15 3a3 3 0 0 1 3 3 3 3 0 0 1 3 3v4a3 3 0 0 1-3 3 3 3 0 0 1-3 3 3 3 0 0 1-3-3v-12a3 3 0 0 1 3-3z" />
        </svg>
      );
    case "user":
      return (
        <svg {...common}>
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
          <circle cx="12" cy="7" r="4" />
        </svg>
      );
    case "pen":
      return (
        <svg {...common}>
          <path d="M12 20h9" />
          <path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4 12.5-12.5z" />
        </svg>
      );
    case "save":
      return (
        <svg {...common}>
          <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
          <polyline points="17 21 17 13 7 13 7 21" />
          <polyline points="7 3 7 8 15 8" />
        </svg>
      );
    case "rank":
      return (
        <svg {...common}>
          <line x1="4" y1="6" x2="20" y2="6" />
          <line x1="4" y1="12" x2="14" y2="12" />
          <line x1="4" y1="18" x2="9" y2="18" />
        </svg>
      );
    case "check":
      return (
        <svg {...common}>
          <polyline points="20 6 9 17 4 12" />
        </svg>
      );
    default:
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="3" />
        </svg>
      );
  }
}

export function buildNodesFromSteps(steps: string[]): TimelineNode[] {
  // Map free-text reasoning_steps into timeline nodes. Used for the
  // "final" (non-streaming) render path.
  if (!steps.length) return [];
  // Take the first few that look like step transitions for the headline
  // timeline; show the rest collapsed in the parent.
  return steps.slice(0, 6).map((line, idx) => ({
    id: `final-${idx}`,
    step: `final_${idx}`,
    label: line.replace(/\s+/g, " ").trim().slice(0, 120),
    status: "done" as const,
  }));
}

export function ReasoningTimeline({
  nodes,
  emptyHint,
}: {
  nodes: TimelineNode[];
  emptyHint?: string;
}) {
  const resolved = useMemo(() => nodes.filter((n) => n.label.trim()), [nodes]);
  if (resolved.length === 0) {
    return emptyHint ? (
      <p className="text-xs text-slate-500">{emptyHint}</p>
    ) : null;
  }
  return (
    <ol className="space-y-2">
      {resolved.map((node, idx) => {
        const meta = describe(node.step);
        const isLast = idx === resolved.length - 1;
        const palette = paletteFor(node.status);
        return (
          <li key={node.id} className="relative pl-8">
            {!isLast && (
              <span className="absolute left-3 top-7 h-full w-px bg-slate-800" />
            )}
            <span
              className={`absolute left-0 top-0.5 flex h-6 w-6 items-center justify-center rounded-full border ${palette.ring}`}
            >
              <span className={palette.icon}>
                <Icon name={meta.icon} />
              </span>
              {node.status === "active" && (
                <span className="absolute -inset-1 animate-ping rounded-full bg-brand-500/40" />
              )}
            </span>
            <div className="flex flex-col">
              <span className={`text-sm ${palette.text}`}>{meta.label}</span>
              <span className="text-xs text-slate-500">{node.label}</span>
              {node.detail && (
                <span className="text-xs text-slate-600">{node.detail}</span>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function paletteFor(status: TimelineNode["status"]) {
  switch (status) {
    case "active":
      return {
        ring: "border-brand-500 bg-brand-500/15",
        icon: "text-brand-500",
        text: "text-slate-100",
      };
    case "done":
      return {
        ring: "border-emerald-500/50 bg-emerald-500/10",
        icon: "text-emerald-300",
        text: "text-slate-300",
      };
    default:
      return {
        ring: "border-slate-700 bg-slate-900/60",
        icon: "text-slate-500",
        text: "text-slate-500",
      };
  }
}
