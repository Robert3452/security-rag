from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.api.deps import EmbedderDep, SessionDep
from app.schemas.documents import DocumentMeta, IngestResult
from app.services.ingest import EmptyDocumentError, ingest_document

router = APIRouter(tags=["ingest"])

MAX_BYTES = 50 * 1024 * 1024


@router.post("/ingest", status_code=status.HTTP_201_CREATED)
async def ingest(
    session: SessionDep,
    embedder: EmbedderDep,
    file: Annotated[UploadFile, File()],
    meta: Annotated[DocumentMeta, Form()],
) -> IngestResult:
    content = await file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "El pdf supera 50 MB")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "El archivo no es un PDF")
    try:
        return await ingest_document(session, embedder, content, meta)
    except EmptyDocumentError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
