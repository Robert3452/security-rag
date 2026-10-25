from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.schemas.documents import Kind


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    source_id: str
    title: str
    url: str | None
    page: int
    content: str
    score: float
    kind: Kind


async def search_similar(
    session: AsyncSession,
    query_embedding: list[float],
    k: int = 5,
    language: str | None = None,
) -> list[RetrievedChunk]:
    distance = Chunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(
            Document.source_id,
            Document.title,
            Document.url,
            Document.kind,
            Chunk.page,
            Chunk.content,
            (1 - distance).label("score"),
        )
        .join(Document, Document.id == Chunk.document_id)
        .order_by(distance)
        .limit(k)
    )
    if language is not None:
        stmt = stmt.where(Document.language == language)

    rows = (await session.execute(stmt)).all()

    return [
        RetrievedChunk(
            source_id=r.source_id,
            title=r.title,
            url=r.url,
            page=r.page,
            content=r.content,
            score=float(r.score),
            kind=r.kind,
        )
        for r in rows
    ]
