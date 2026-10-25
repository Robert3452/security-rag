"""Indexa los documentos de data/raw/ definidos en data/sources.csv."""

import asyncio
import csv
import sys
from pathlib import Path

from pydantic import ValidationError

from app.core.config import get_settings
from app.db.session import SessionLocal, engine
from app.embeddings.e5 import E5Embedder
from app.schemas.documents import DocumentMeta
from app.services.ingest import EmptyDocumentError, ingest_document

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "data" / "sources.csv"
RAW_DIR = ROOT / "data" / "raw"


def load_jobs() -> tuple[list[tuple[DocumentMeta, Path]], list[str]]:
    jobs: list[tuple[DocumentMeta, Path]] = []
    errors: list[str] = []
    with SOURCES.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source_id, kind = row["id"], row["kind"]
            path = RAW_DIR / f"{source_id}.{kind}"
            if not path.exists():
                errors.append(f"{source_id}: falta {path.name} en data/raw/")
                continue
            try:
                meta = DocumentMeta.model_validate(
                    {
                        "source_id": source_id,
                        "title": row["title"],
                        "language": row["language"],
                        "kind": kind,
                        "url": row["url"] or None,
                    }
                )
            except ValidationError as e:
                errors.append(
                    f"{source_id}: fila inválida en sources.csv ({e.error_count()} errores)"
                )
                continue
            jobs.append((meta, path))
    return jobs, errors


async def main() -> int:
    jobs, errors = load_jobs()
    for error in errors:
        print(f"x {error}", file=sys.stderr)

    embedder = E5Embedder(get_settings().embedding_model)
    failures = len(errors)

    for meta, path in jobs:
        async with SessionLocal() as session:
            try:
                r = await ingest_document(session, embedder, path, meta)
            except EmptyDocumentError as e:
                failures += 1
                print(f"x {meta.source_id}: {e}", file=sys.stderr)
                continue
        print(
            f"+ {r.source_id} [{meta.kind}]: {r.pages} págs,"
            f"{r.chunks} chunks, {r.ocr_pages} con OCR"
        )

    await engine.dispose()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
