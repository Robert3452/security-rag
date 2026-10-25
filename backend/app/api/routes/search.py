from fastapi import APIRouter

from app.api.deps import EmbedderDep, SessionDep
from app.schemas.documents import SearchHit, SearchRequest
from app.services.retrieval import retrieve

router = APIRouter(tags=["search"])


@router.post("/search")
async def search(
    body: SearchRequest, session: SessionDep, embedder: EmbedderDep
) -> list[SearchHit]:
    chunks = await retrieve(session, embedder, body.question, k=body.k, language=body.language)
    return [SearchHit.model_validate(c) for c in chunks]
