import openai
from openai import AsyncOpenAI

from app.llm.base import LLMError


class OpenAILLM:
    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def generate(self, system: str, user: str) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except openai.APIError as e:
            raise LLMError(str(e)) from e
        return response.choices[0].message.content or ""
