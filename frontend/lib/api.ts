export type UserProfile = {
  user_id: string;
  age_range?: string;
  location?: string;
  interests: string[];
  tone_preference?: string;
  sentiment_bias?: string;
};

type ApiError = {
  detail?: string | { detail?: string };
  error?: string;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

async function request<T>(path: string, init: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init.headers ?? {})
    }
  });

  if (!response.ok) {
    let message = `Request failed with ${response.status}`;
    try {
      const parsed = (await response.json()) as ApiError;
      if (typeof parsed.detail === "string") {
        message = parsed.detail;
      } else if (parsed.detail && typeof parsed.detail.detail === "string") {
        message = parsed.detail.detail;
      } else if (parsed.error) {
        message = parsed.error;
      }
    } catch {
      // ignore parse errors and keep fallback message
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

export type SimulateReviewPayload = {
  user_profile: UserProfile;
  item_data: {
    item_name: string;
    item_context?: string;
  };
  persona_style?: string;
};

export type SimulateReviewResponse = {
  review_text: string;
  rating: number;
  persona_breakdown: Record<string, unknown>;
  reasoning_steps: string[];
};

export function simulateReview(payload: SimulateReviewPayload) {
  return request<SimulateReviewResponse>("/simulate-review", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export type RecommendationPayload = {
  user_profile: UserProfile;
  candidate_items: string[];
  context?: string;
  conversation_history?: string[];
  top_k: number;
  recommender_personality?: "analyst" | "coach" | "friend" | "nigerian_twitter";
  conversational_mode?: boolean;
};

export type RecommendationResponse = {
  recommendations: Array<{
    item_name: string;
    score: number;
    explanation: string;
  }>;
  memory_retrieved: string[];
  reasoning_steps: string[];
  conversational_response?: string;
  explainability?: Record<string, unknown>;
};

export function recommendItems(payload: RecommendationPayload) {
  return request<RecommendationResponse>("/recommend", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}
