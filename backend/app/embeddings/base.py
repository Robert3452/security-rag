from typing import Protocol


class Embedder(Protocol):
    dim: int

    # Consulta:
    def embed_query(self, text: str) -> list[float]: ...
    # Fragmento de documento
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...
