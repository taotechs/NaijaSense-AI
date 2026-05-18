/**
 * NaijaSense AI - Frontend API Client
 * This handles the bridge to the Behavioral Intelligence Hub (FastAPI backend)
 */

import { getApiRoot } from "@/lib/api-root";

// 1. Base Configuration
const AGENT_URL =
  process.env.NEXT_PUBLIC_AGENT_API_URL || "http://127.0.0.1:8000/api/agent/v1";

// 2. Type Definitions
export type AgentLanguage = "english" | "pidgin" | "yoruba_mix";

export type UserPersonaPayload = {
  user_id: string;
  location?: string;
  interests: string[];
  sentiment_bias?: string;
  tone_notes?: string;
  history?: string;
  language?: AgentLanguage;
};

export type AgentGatewayRequest = {
  user_persona: UserPersonaPayload;
  query: string;
  top_k?: number;
  include_history?: boolean;
  compare_with_no_history?: boolean;
};

export type AgentReview = {
  review_text: string;
  rating: number;
  persona_breakdown: Record<string, unknown>;
};

export type AgentRecommendation = {
  recommendations: Array<{
    item_name: string;
    score: number;
    explanation: string;
  }>;
  conversational_response?: string;
  explainability?: Record<string, unknown>;
  memory_retrieved?: string[];
};

export type AgentGatewayResponse = {
  task: "review" | "recommend";
  orchestrator_rationale: string;
  routing_source: string;
  review?: AgentReview;
  recommendation?: AgentRecommendation;
  reasoning_steps: string[];
  safety_flags?: string[];
  timing_ms?: number;
  language?: AgentLanguage;
  no_history_variant?: AgentGatewayResponse | null;
};

export type StreamEvent =
  | { type: "start"; ts: number }
  | { type: "plan"; flow: string; steps: string[] }
  | { type: "route"; task: string; source: string; rationale: string }
  | { type: "step_start"; flow: string; step: string }
  | { type: "step_end"; flow: string; step: string }
  | { type: "final"; result: AgentGatewayResponse }
  | { type: "error"; status: number; detail: unknown };

// 3. The Unified Gateway Function (non-streaming)
export async function postAgentGateway(
  body: AgentGatewayRequest
): Promise<AgentGatewayResponse> {
  // The fetch call precisely targets the "/api/agent/v1" endpoint defined in agent_routes.py
  // Note: We use a template literal to combine the base URL and the fixed path.
  const response = await fetch(`${AGENT_URL}/api/agent/v1`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    let message = `Agent request failed (${response.status})`;
    try {
      const err = (await response.json()) as { detail?: unknown };

      // Handle standard FastAPI error strings or detail objects
      if (typeof err.detail === "string") {
        message = err.detail;
      } else if (
        err.detail &&
        typeof err.detail === "object" &&
        "detail" in err.detail
      ) {
        message = String(
          (err.detail as { detail?: string }).detail ?? message
        );
      }
    } catch {
      /* ignore parsing errors and use fallback message */
    }
    throw new Error(message);
  }

  return (await response.json()) as AgentGatewayResponse;
}

// 4. Streaming variant — parses newline-delimited JSON from /api/agent/v1/stream.
// Each event is delivered to ``onEvent`` as it arrives, including the
// final ``{ type: 'final', result }``. The promise resolves with the
// final response once the stream closes.
export async function streamAgentGateway(
  body: AgentGatewayRequest,
  onEvent: (event: StreamEvent) => void,
  options: { signal?: AbortSignal } = {}
): Promise<AgentGatewayResponse> {
  const response = await fetch(`${AGENT_URL}/api/agent/v1/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/x-ndjson",
    },
    body: JSON.stringify(body),
    signal: options.signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Stream failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult: AgentGatewayResponse | null = null;
  let lastErrorEvent: Extract<StreamEvent, { type: "error" }> | null = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // Process line-by-line — server flushes one JSON object per newline.
    let idx: number;
    while ((idx = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line) continue;
      try {
        const event = JSON.parse(line) as StreamEvent;
        onEvent(event);
        if (event.type === "final") finalResult = event.result;
        if (event.type === "error") lastErrorEvent = event;
      } catch {
        // Defensive: malformed line, skip it but keep the stream alive.
      }
    }
  }

  if (lastErrorEvent) {
    const detail = lastErrorEvent.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail && typeof detail === "object" && "detail" in detail
        ? String((detail as { detail?: string }).detail)
        : `Agent error (${lastErrorEvent.status})`;
    throw new Error(message);
  }
  if (!finalResult) {
    throw new Error("Agent stream ended without a final result.");
  }
  return finalResult;
}

// 5. Health check — used by the BackendStatus pill + page-load pre-warm.
export type BackendHealth = { status: string; service?: string };

export async function getBackendHealth(): Promise<BackendHealth> {
  const root = getApiRoot() || "http://127.0.0.1:8000";
  const res = await fetch(`${root}/api/v1/health`, {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return (await res.json()) as BackendHealth;
}

// 6. Feedback (thumbs up/down). Fire-and-forget from the UI is fine —
// failures should not break the page, so callers usually ignore errors.
export type FeedbackInput = {
  user_id: string;
  task: "review" | "recommend";
  rating: 1 | -1;
  query: string;
  output_preview?: string;
  note?: string;
  routing_source?: string;
  language?: AgentLanguage;
};

export type FeedbackAck = { received: boolean; id: string };

export async function postFeedback(payload: FeedbackInput): Promise<FeedbackAck> {
  const res = await fetch(`${AGENT_URL}/api/agent/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Feedback failed (${res.status})`);
  return (await res.json()) as FeedbackAck;
}
