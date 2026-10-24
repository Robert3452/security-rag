from httpx import ASGITransport, AsyncClient

from app.api.deps import get_embedder
from app.main import app


class FakeEmbedder:
    dim = 384

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 384

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]


async def post_ingest(files: dict[str, tuple[str, bytes, str]], data: dict[str, str]) -> int:
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/ingest", files=files, data=data)
    finally:
        app.dependency_overrides.clear()
    return r.status_code


VALID_META = {"source_id": "test-doc", "title": "Prueba", "language": "es"}


async def test_rejects_non_pdf() -> None:
    status = await post_ingest({"file": ("x.pdf", b"hola", "application/pdf")}, VALID_META)
    assert status == 415


async def test_rejects_invalid_source_id() -> None:
    meta = {**VALID_META, "source_id": "Con Espacios"}
    status = await post_ingest({"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}, meta)
    assert status == 422
