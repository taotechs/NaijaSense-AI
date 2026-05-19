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
  user_persona: string;
  product_details: string;
};

export type TaskAResponse = {
  rating: number;
  review_text: string;
  review_reasoning?: string;
};

export function postTaskA(payload: TaskAPayload) {
  return taskRequest<TaskAResponse>("/task-a/user-modeling", payload);
}

export type TaskBPayload = {
  user_persona: {
    user_id: string;
    persona: string;
  };
};

export type TaskBResponse = {
  /** Single flowing paragraph of recommendation sentences. */
  recommendations: string;
  agent_reasoning: string;
};

export function postTaskB(payload: TaskBPayload) {
  return taskRequest<TaskBResponse>("/task-b/recommendation", payload);
}
