"""Indexa los PDFs de data/raw/ definidos en data/sources.csv."""

import asyncio
import csv
import sys
from pathlib import Path

from app.core.config import get_settings
from app.db.session import SessionLocal, engine
from app.embeddings.e5 import E5Embedder
from app.schemas.documents import DocumentMeta
from app.services.ingest import EmptyDocumentError, ingest_pdf

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "data" / "sources.csv"
RAW_DIR = ROOT / "data" / "raw"


def load_jobs() -> tuple[list[tuple[DocumentMeta, Path]], list[str]]:
    jobs: list[tuple[DocumentMeta, Path]] = []
    missing: list[str] = []
    with SOURCES.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = RAW_DIR / f"{row['id']}.pdf"
            if not path.exists():
                missing.append(row["id"])
                continue
            meta = DocumentMeta.model_validate(
                {
                    "source_id": row["id"],
                    "title": row["title"],
                    "language": row["language"],
                    "url": row["url"] or None,
                }
            )
            jobs.append((meta, path))
    return jobs, missing


async def main() -> int:
    jobs, missing = load_jobs()
    for source_id in missing:
        print(f"x {source_id}: falta el PDF en data/raw/", file=sys.stderr)

    embedder = E5Embedder(get_settings().embedding_model)
    failures = len(missing)

    for meta, path in jobs:
        async with SessionLocal() as session:
            try:
                r = await ingest_pdf(session, embedder, path, meta)
            except EmptyDocumentError as e:
                failures += 1
                print(f"x {meta.source_id}: {e}", file=sys.stderr)
                continue
        print(f"+ {r.source_id}: {r.pages} págs, {r.chunks} chunks, {r.ocr_pages} con OCR")

    await engine.dispose()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
