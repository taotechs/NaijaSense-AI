const AGENT_URL =
  process.env.NEXT_PUBLIC_AGENT_API_URL ?? "http://127.0.0.1:8000/api/agent/v1";

export type UserPersonaPayload = {
  user_id: string;
  location?: string;
  interests: string[];
  sentiment_bias?: string;
  tone_notes?: string;
  history?: string;
};

export type AgentGatewayRequest = {
  user_persona: UserPersonaPayload;
  query: string;
  top_k?: number;
};

export type AgentGatewayResponse = {
  task: "review" | "recommend";
  orchestrator_rationale: string;
  routing_source: string;
  review?: {
    review_text: string;
    rating: number;
    persona_breakdown: Record<string, unknown>;
  };
  recommendation?: {
    recommendations: Array<{ item_name: string; score: number; explanation: string }>;
    conversational_response?: string;
    explainability?: Record<string, unknown>;
    memory_retrieved?: string[];
  };
  reasoning_steps: string[];
};

export async function postAgentGateway(body: AgentGatewayRequest): Promise<AgentGatewayResponse> {
  const response = await fetch(AGENT_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    let message = `Agent request failed (${response.status})`;
    try {
      const err = (await response.json()) as { detail?: unknown };
      if (typeof err.detail === "string") message = err.detail;
      else if (err.detail && typeof err.detail === "object" && "detail" in err.detail) {
        message = String((err.detail as { detail?: string }).detail ?? message);
      }
    } catch {
      /* ignore */
    }
    throw new Error(message);
  }
  return (await response.json()) as AgentGatewayResponse;
}
