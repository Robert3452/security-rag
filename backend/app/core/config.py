from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    llm_provider: Literal["anthropic", "openai", "groq"] = "openai"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.4-mini"
    groq_api_model: str = "openai/gpt-oss-120b"
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-haiku-5-5"
    llm_max_tokens: int = 1024
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
