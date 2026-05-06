"use client";

import { FormEvent, useMemo, useState } from "react";
import { recommendItems, simulateReview } from "@/lib/api";
import {
  extractCandidateItems,
  extractInterests,
  inferChatMode,
  inferPersonaStyle,
  inferSentimentBias
} from "@/lib/prompt-parser";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  meta?: string;
};

export default function ChatPage() {
  const [prompt, setPrompt] = useState("");
  const [userId, setUserId] = useState("chat_user_1");
  const [location, setLocation] = useState("Lagos");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Tell me what you like (e.g. 'I like spicy food and cheap restaurants') and I will respond with recommendations or review simulation + reasoning.",
      meta: "NaijaSense Copilot"
    }
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const placeholder = useMemo(
    () => "Type naturally: I like spicy food and cheap restaurants",
    []
  );

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    const userText = prompt.trim();
    if (!userText) return;
    setError(null);
    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", content: userText }]);
    setPrompt("");

    try {
      const mode = inferChatMode(userText);
      const interests = extractInterests(userText);
      const sentiment = inferSentimentBias(userText);
      const personaStyle = inferPersonaStyle(userText);

      if (mode === "recommend") {
        const candidateItems = extractCandidateItems(userText);
        const res = await recommendItems({
          user_profile: {
            user_id: userId,
            location,
            interests,
            sentiment_bias: sentiment
          },
          candidate_items: candidateItems,
          context: userText,
          top_k: 3,
          recommender_personality: "nigerian_twitter",
          conversational_mode: true
        });

        const list = res.recommendations
          .map((item, idx) => `${idx + 1}. ${item.item_name} (${item.score.toFixed(2)}) - ${item.explanation}`)
          .join("\n");
        const reasoning = res.reasoning_steps.map((s, idx) => `${idx + 1}. ${s}`).join("\n");
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `${res.conversational_response ?? "Here are recommendations."}\n\nRecommendations:\n${list}\n\nReasoning:\n${reasoning}`,
            meta: `Mode: recommendation | Personality: nigerian_twitter | Interests: ${interests.join(", ")}`
          }
        ]);
      } else {
        const itemGuess = interests[0] === "general lifestyle" ? "General Product" : `${interests[0]} option`;
        const res = await simulateReview({
          user_profile: {
            user_id: userId,
            location,
            interests,
            sentiment_bias: sentiment
          },
          item_data: {
            item_name: itemGuess,
            item_context: userText
          },
          persona_style: personaStyle
        });
        const reasoning = res.reasoning_steps.map((s, idx) => `${idx + 1}. ${s}`).join("\n");
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Generated Review:\n${res.review_text}\n\nRating: ${res.rating.toFixed(
              1
            )}/5\n\nReasoning:\n${reasoning}`,
            meta: `Mode: review | Persona: ${personaStyle} | Interests: ${interests.join(", ")}`
          }
        ]);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Request failed.";
      setError(message);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "I hit an error while processing that request.", meta: message }
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="grid gap-6 lg:grid-cols-[300px_1fr]">
      <div className="glass rounded-2xl p-5">
        <h2 className="text-lg font-semibold">Chat Settings</h2>
        <p className="mt-1 text-sm text-slate-400">
          Keep this minimal and focus on natural conversation storytelling.
        </p>
        <div className="mt-4 space-y-3">
          <input className="field" value={userId} onChange={(e) => setUserId(e.target.value)} placeholder="User ID" />
          <input className="field" value={location} onChange={(e) => setLocation(e.target.value)} placeholder="Location" />
          <div className="rounded-xl bg-slate-900 p-3 text-xs text-slate-400">
            The assistant auto-detects whether your prompt is for recommendation or review simulation.
          </div>
        </div>
      </div>

      <div className="glass flex min-h-[70vh] flex-col rounded-2xl p-4">
        <h3 className="px-2 text-lg font-semibold">NaijaSense Copilot</h3>
        <div className="mt-3 flex-1 space-y-3 overflow-y-auto rounded-xl bg-slate-950/70 p-3">
          {messages.map((message, idx) => (
            <div
              key={`${message.role}-${idx}`}
              className={`max-w-3xl rounded-2xl px-4 py-3 text-sm ${
                message.role === "user"
                  ? "ml-auto bg-brand-500 text-slate-950"
                  : "bg-slate-900 text-slate-200"
              }`}
            >
              <p className="whitespace-pre-wrap">{message.content}</p>
              {message.meta && (
                <p className="mt-2 text-[11px] opacity-70">{message.meta}</p>
              )}
            </div>
          ))}
          {loading && (
            <div className="max-w-3xl rounded-2xl bg-slate-900 px-4 py-3 text-sm text-slate-400">
              Thinking...
            </div>
          )}
        </div>
        <form onSubmit={onSubmit} className="mt-3 space-y-2">
          <textarea
            className="field min-h-24"
            placeholder={placeholder}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
          <div className="flex items-center justify-between">
            {error ? <p className="text-xs text-red-300">{error}</p> : <span />}
            <button className="btn" disabled={loading || !prompt.trim()}>
              {loading ? "Sending..." : "Send"}
            </button>
          </div>
        </form>
      </div>
    </section>
  );
}
