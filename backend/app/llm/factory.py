from typing import assert_never

from app.core.config import Settings
from app.llm.anthropic_llm import AnthropicLLM
from app.llm.base import LLM
from app.llm.openai_llm import OpenAILLM


def build_llm(settings: Settings) -> LLM:
    match settings.llm_provider:
        case "openai":
            if settings.openai_api_key is None:
                raise RuntimeError("LLM_PROVIDER=openai requiere OPENAI_API_KEY")
            return OpenAILLM(settings.openai_api_key.get_secret_value(), settings.openai_model)
        case "anthropic":
            if settings.anthropic_api_key is None:
                raise RuntimeError("LLM_PROVIDER=anthropic requiere ANTHROPIC_API_KEY")
            return AnthropicLLM(
                settings.anthropic_api_key.get_secret_value(),
                settings.anthropic_model,
                settings.llm_max_tokens,
            )
        case "groq":
            if settings.groq_api_key is None:
                raise RuntimeError("LLM_PROVIDER=groq requiere GROQ_VARIABLES")
            return OpenAILLM(
                settings.groq_api_key.get_secret_value(),
                settings.groq_api_model,
                settings.groq_base_url,
            )

        case _:
            assert_never(settings.llm_provider)
