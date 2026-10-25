from fastapi import APIRouter, HTTPException, status

from app.api.deps import EmbedderDep, LLMDep, SessionDep
from app.llm.base import LLMError
from app.schemas.documents import AskRequest, AskResponse, AskSource
from app.services.ask import ask

router = APIRouter(tags=["ask"])


@router.post("/ask")
async def ask_question(
    body: AskRequest, session: SessionDep, embedder: EmbedderDep, llm: LLMDep
) -> AskResponse:
    try:
        result = await ask(session, embedder, llm, body.question, k=body.k, language=body.language)
    except LLMError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "El proveedor del LLM no respondió") from e

    return AskResponse(
        answer=result.answer,
        sources=[
            AskSource(
                ref=c.ref,
                source_id=c.chunk.source_id,
                title=c.chunk.title,
                page=c.chunk.page,
                url=c.chunk.url,
                score=c.chunk.score,
                kind=c.chunk.kind,
            )
            for c in result.citations
        ],
    )
