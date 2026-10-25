class FakeEmbedder:
    dim = 384

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 384

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]


class FakeLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    async def generate(self, system: str, user: str) -> str:
        self.calls += 1
        return self.answer
