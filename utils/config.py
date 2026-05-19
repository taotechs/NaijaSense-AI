"""Application configuration."""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    app_env: str = "dev"
    api_prefix: str = "/api/v1"
    default_persona_style: str = "formal"
    max_history_items: int = 30
    model_name: str = "gpt-4o-mini"
    # Single committed corpus for Task A few-shots, Task B stage-1, and unified history.
    review_corpus_path: str = "data/processed/review_corpus.jsonl"
    corpus_index_path: str = "data/processed/corpus_index.json"
    # Alias kept for health checks / legacy env; always points at review_corpus_path.
    large_corpus_path: str = "data/processed/review_corpus.jsonl"
    corpus_query_timeout_sec: float = 2.5
    # When true, uvicorn startup builds large_corpus if missing (Docker uses entrypoint instead).
    corpus_build_on_startup: bool = True

    # Unified agent / LangChain orchestrator (none | openai | groq)
    orchestrator_provider: str = "none"
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.1-8b-instant"
    # Split models per role: cheap+fast for routing/persona, strong for generation.
    groq_router_model: Optional[str] = None
    groq_generator_model: Optional[str] = None
    orchestrator_model: str = "gpt-4o-mini"

    # Task B stage-2 rerank: groq (free tier, default) | gemini | auto
    task_b_rerank_provider: str = "groq"
    task_b_top_k: int = 6
    gemini_api_key: Optional[str] = None
    task_b_gemini_model: str = "gemini-3-flash-preview"

    # Generator sampling controls — surfaced so they can be tuned via .env
    # without code changes. Larger penalties / top_p reduce repetition.
    gen_temperature: float = 0.85
    gen_top_p: float = 0.9
    gen_presence_penalty: float = 0.6
    gen_frequency_penalty: float = 0.5
    gen_max_tokens: int = 320

    # Review critique → regenerate pass. When enabled, a cheap second call
    # (uses the router model) scores the generated review for specificity and
    # rewrites it once if the score is below ``review_critique_threshold``.
    review_critique_enabled: bool = True
    review_critique_threshold: int = 4

    # Chroma (optional; compose service chroma:8000)
    chroma_host: Optional[str] = None
    chroma_port: int = 8000

    # Comma-separated list of origins allowed to hit the API. Local dev
    # defaults match the Next.js dev server; in production this should be
    # set to the deployed frontend URL (e.g. ``CORS_ORIGINS=https://your-app.vercel.app``)
    # via the host's secret manager.
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

