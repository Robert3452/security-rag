import asyncio
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.base import Embedder
from app.ingest.chunk import TextChunk, chunk_pages
from app.ingest.extract import extract_pages
from app.repositories import documents as repo
from app.schemas.documents import DocumentMeta, IngestResult


class EmptyDocumentError(Exception):
    """El PDF no produjo texto utilizable"""


@dataclass(frozen=True, slots=True)
class _Prepared:
    chunks: list[tuple[TextChunk, list[float]]]
    pages: int
    ocr_pages: int


def _prepare(source: Path | bytes, language: str, embedder: Embedder) -> _Prepared:
    pages = list(extract_pages(source, language))
    chunks = list(chunk_pages(pages))
    if not chunks:
        raise EmptyDocumentError("El Pdf no contiene texto extraible")

    vectors = embedder.embed_passages([c.text for c in chunks])

    return _Prepared(
        chunks=list(zip(chunks, vectors, strict=True)),
        pages=len(pages),
        ocr_pages=sum(p.ocr for p in pages),
    )


async def ingest_pdf(
    session: AsyncSession, embedder: Embedder, source: Path | bytes, meta: DocumentMeta
) -> IngestResult:
    prepared = await asyncio.to_thread(_prepare, source, meta.language, embedder)
    await repo.replace_document(session, meta, prepared.chunks)
    await session.commit()

    return IngestResult(
        source_id=meta.source_id,
        title=meta.title,
        pages=prepared.pages,
        chunks=len(prepared.chunks),
        ocr_pages=prepared.ocr_pages,
    )
