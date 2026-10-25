from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import ask, health, ingest, search
from app.core.config import get_settings
from app.embeddings.e5 import E5Embedder
from app.llm.factory import build_llm


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    embedder = E5Embedder(settings.embedding_model)
    if embedder.dim != settings.embedding_dim:
        raise RuntimeError(
            f"El modelo genera vectores de {embedder.dim} dimensiones."
            f"pero la DB espera {settings.embedding_dim}"
        )
    app.state.llm = build_llm(settings)
    app.state.embedder = embedder

    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Rag Normativa de seguridad", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(ask.router)
    app.include_router(search.router)
    return app


app = create_app()
