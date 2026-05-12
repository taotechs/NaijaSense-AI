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
    review_corpus_path: str = "data/processed/review_corpus.jsonl"

    # Unified agent / LangChain orchestrator (none | openai | groq)
    orchestrator_provider: str = "none"
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.1-8b-instant"
    # Split models per role: cheap+fast for routing/persona, strong for generation.
    groq_router_model: Optional[str] = None
    groq_generator_model: Optional[str] = None
    orchestrator_model: str = "gpt-4o-mini"

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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

