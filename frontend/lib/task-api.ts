import { getApiRoot } from "@/lib/api-root";

type ApiError = {
  detail?: string | { detail?: string };
  error?: string;
};

async function taskRequest<T>(path: string, body: unknown): Promise<T> {
  const root = getApiRoot();
  const url = `${root}${path}`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const parsed = (await response.json()) as ApiError;
      if (typeof parsed.detail === "string") message = parsed.detail;
      else if (parsed.error) message = parsed.error;
    } catch {
      /* keep default */
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

export type TaskAPayload = {
  user_persona: {
    user_id: string;
    location?: string;
    interests?: string[];
    sentiment_bias?: string;
    tone_notes?: string;
    history?: string;
    language?: string;
  };
  product_details: {
    item_name: string;
    item_context?: string;
    category?: string;
  };
  persona_style?: string;
};

export type TaskAResponse = {
  rating: number;
  review: string;
  reasoning_steps: string[];
  persona_breakdown: Record<string, unknown>;
};

export function postTaskA(payload: TaskAPayload) {
  return taskRequest<TaskAResponse>("/task-a/user-modeling", payload);
}

export type TaskBPayload = {
  user_persona: {
    user_id: string;
    location?: string;
    interests?: string[];
    sentiment_bias?: string;
    tone_notes?: string;
    history?: string;
    language?: string;
  };
  context?: string;
  top_k?: number;
  candidate_items?: string[];
};

export type TaskBRecommendation = {
  rank: number;
  item_name: string;
  score: number;
  explanation: string;
};

export type TaskBResponse = {
  recommendations: TaskBRecommendation[];
  chain_of_thought: string[];
  reasoning_steps: string[];
  scenario_flags: Record<string, boolean>;
};

export function postTaskB(payload: TaskBPayload) {
  return taskRequest<TaskBResponse>("/task-b/recommendation", payload);
}
