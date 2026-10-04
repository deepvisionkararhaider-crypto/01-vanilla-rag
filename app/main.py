"""FastAPI application entrypoint for the Vanilla RAG service."""

from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.rag_pipeline import get_pipeline

settings = get_settings()
setup_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the pipeline at startup so configuration/model errors surface early
    # rather than on the first request.
    get_pipeline()
    logger.info("Vanilla RAG service started.")
    yield
    logger.info("Vanilla RAG service shutting down.")


app = FastAPI(
    title="Vanilla RAG",
    version=__version__,
    description=(
        "A from-scratch Retrieval-Augmented Generation service: document ingestion, "
        "recursive chunking, embeddings, vector similarity search, and grounded answers "
        "with citations. Runs fully offline with a deterministic embedder and extractive "
        "generator, and upgrades to real transformer embeddings + FAISS + an LLM via env config."
    ),
    lifespan=lifespan,
)

_origins = settings.cors_origin_list or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials="*" not in _origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["system"])
def root() -> dict:
    return {
        "service": "vanilla-rag",
        "version": __version__,
        "health": "/health",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
