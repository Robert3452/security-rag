import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.llm.anthropic_llm import AnthropicLLM
from app.llm.factory import build_llm
from app.llm.openai_llm import OpenAILLM


def test_selects_openai() -> None:
    settings = Settings(llm_provider="openai", openai_api_key=SecretStr("test"))
    assert isinstance(build_llm(settings), OpenAILLM)


def test_selects_anthropic() -> None:
    settings = Settings(llm_provider="anthropic", anthropic_api_key=SecretStr("test"))
    assert isinstance(build_llm(settings), AnthropicLLM)


def test_missing_key_fails_fast() -> None:
    settings = Settings(llm_provider="openai", openai_api_key=None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_llm(settings)
