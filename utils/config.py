"""Application configuration."""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    app_env: str = "dev"
    api_prefix: str = "/api/v1"
    default_persona_style: str = "nigerian_twitter"
    max_history_items: int = 30
    model_name: str = "gpt-4o-mini"
    review_corpus_path: str = "data/processed/review_corpus.jsonl"

    # Unified agent / LangChain orchestrator (none | openai | groq)
    orchestrator_provider: str = "none"
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.1-8b-instant"
    orchestrator_model: str = "gpt-4o-mini"

    # Chroma (optional; compose service chroma:8000)
    chroma_host: Optional[str] = None
    chroma_port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

