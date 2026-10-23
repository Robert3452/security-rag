from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file = ".env", extra = "ignore")
    
    database_url:str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    llm_provider: Literal["openai","ollama"]="openai" 
    openai_api_key: SecretStr | None = None
    ollama_base_url:str = "http://localhost:11434"
    embedding_model:str = "intfloat/multilingual-e5-small"
    embedding_dim:int = 384
    top_k:int = 5
    
@lru_cache
def get_settings() -> Settings:
    return Settings()
