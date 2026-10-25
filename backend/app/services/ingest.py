import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.base import Embedder
from app.ingest.chunk import TextChunk, chunk_pages
from app.ingest.extract import PageText, extract_pages
from app.ingest.html import extract_html
from app.repositories import documents as repo
from app.schemas.documents import DocumentMeta, IngestResult


class EmptyDocumentError(Exception):
    """El documento no produjo texto utilizable."""


@dataclass(frozen=True, slots=True)
class _Prepared:
    chunks: list[tuple[TextChunk, list[float]]]
    pages: int
    ocr_pages: int


def _extract(source: Path | bytes, meta: DocumentMeta) -> list[PageText]:
    match meta.kind:
        case "pdf":
            return list(extract_pages(source, meta.language))
        case "html":
            raw = source.read_bytes() if isinstance(source, Path) else source
            return extract_html(raw)
        case _:
            assert_never(meta.kind)


def _prepare(source: Path | bytes, meta: DocumentMeta, embedder: Embedder) -> _Prepared:
    pages = _extract(source, meta)
    chunks = list(chunk_pages(pages))
    if not chunks:
        raise EmptyDocumentError(f"El {meta.kind.upper()} no contiene texto extraíble")

    vectors = embedder.embed_passages([c.text for c in chunks])
    return _Prepared(
        chunks=list(zip(chunks, vectors, strict=True)),
        pages=len(pages),
        ocr_pages=sum(p.ocr for p in pages),
    )


async def ingest_document(
    session: AsyncSession, embedder: Embedder, source: Path | bytes, meta: DocumentMeta
) -> IngestResult:
    prepared = await asyncio.to_thread(_prepare, source, meta, embedder)
    await repo.replace_document(session, meta, prepared.chunks)
    await session.commit()

    return IngestResult(
        source_id=meta.source_id,
        title=meta.title,
        pages=prepared.pages,
        chunks=len(prepared.chunks),
        ocr_pages=prepared.ocr_pages,
    )
