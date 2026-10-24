from sentence_transformers import SentenceTransformer


class E5Embedder:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self._model = SentenceTransformer(model_name, device="cpu")
        self._batch_size = batch_size
        dim = self._model.get_embedding_dimension()

        if dim is None:
            raise ValueError(f"No se pudo determinar la dimensión de {model_name}")
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"query: {text}"])[0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode([f"passge: {t}" for t in texts])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [[float(x) for x in v] for v in vectors]
