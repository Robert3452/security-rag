from collections.abc import Sequence

from sqlalchemy import delete, insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.ingest.chunk import TextChunk
from app.schemas.documents import DocumentMeta


async def replace_document(
    session: AsyncSession, meta: DocumentMeta, chunks: Sequence[tuple[TextChunk, list[float]]]
) -> Document:
    await session.execute(delete(Document).where(Document.source_id == meta.source_id))

    document = Document(
        source_id=meta.source_id, title=meta.title, language=meta.language, url=meta.url
    )
    meta = meta.model_copy(update={"kind": "pdf"})
    session.add(document)
    await session.flush()

    await session.execute(
        insert(Chunk),
        [
            {
                "document_id": document.id,
                "page": chunk.page,
                "chunk_index": chunk.index,
                "content": chunk.text,
                "embedding": vector,
            }
            for chunk, vector in chunks
        ],
    )
    return document
