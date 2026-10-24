from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.embeddings.base import Embedder


# Que hace esta parte?
def get_embedder(request: Request) -> Embedder:
    embedder: Embedder = request.app.state.embedder  # viene del lifespan
    return embedder


SessionDep = Annotated[AsyncSession, Depends(get_session)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
