from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.embeddings.base import Embedder
from app.llm.base import LLM


# Que hace esta parte?
def get_embedder(request: Request) -> Embedder:
    embedder: Embedder = request.app.state.embedder  # viene del lifespan
    return embedder


def get_llm(request: Request) -> LLM:
    llm: LLM | None = getattr(request.app.state, "llm", None)
    if llm is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "El llm no está configurado")
    return llm


LLMDep = Annotated[LLM, Depends(get_llm)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
