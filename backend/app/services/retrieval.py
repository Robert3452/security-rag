import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.base import Embedder
from app.repositories import chunks as repo


async def retrieve(
    session: AsyncSession,
    embedder: Embedder,
    question: str,
    k: int = 5,
    language: str | None = None,
) -> list[repo.RetrievedChunk]:
    vector = await asyncio.to_thread(embedder.embed_query, question)
    return await repo.search_similar(session, vector, k=k, language=language)
