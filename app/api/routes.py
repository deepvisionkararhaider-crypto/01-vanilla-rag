"""FastAPI routes for ingestion, querying, and introspection.

The pipeline is injected via ``Depends(get_pipeline)`` so tests can override it
with an isolated, deterministic instance (hashing embedder + numpy store).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.schemas.rag import (
    DocumentInfo,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
)
from app.services.rag_pipeline import RAGPipeline, get_pipeline

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(pipeline: RAGPipeline = Depends(get_pipeline)) -> HealthResponse:
    """Liveness + a snapshot of the active pipeline configuration and index size."""
    s = pipeline.stats()
    return HealthResponse(status="ok", **s)


@router.post("/ingest/text", response_model=IngestResponse, tags=["ingest"])
def ingest_text(
    req: IngestTextRequest, pipeline: RAGPipeline = Depends(get_pipeline)
) -> IngestResponse:
    """Ingest raw text (TXT/Markdown content) into the index."""
    try:
        return pipeline.ingest_text(req.text, source_id=req.source_id, metadata=req.metadata)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ingest/file", response_model=IngestResponse, tags=["ingest"])
async def ingest_file(
    file: UploadFile = File(...), pipeline: RAGPipeline = Depends(get_pipeline)
) -> IngestResponse:
    """Ingest an uploaded ``.txt``/``.md``/``.pdf`` file into the index."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        return pipeline.ingest_file(file.filename or "upload", data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/query", response_model=QueryResponse, tags=["query"])
def query(req: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> QueryResponse:
    """Retrieve the most relevant passages and return a grounded, cited answer.

    An empty result set is *not* an error: the response returns ``retrieved: 0``
    with a graceful answer so callers can distinguish "no evidence" from a failure.
    """
    try:
        return pipeline.query(req.question, top_k=req.top_k, source_filter=req.source_filter)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/documents", response_model=list[DocumentInfo], tags=["system"])
def documents(pipeline: RAGPipeline = Depends(get_pipeline)) -> list[DocumentInfo]:
    """List all ingested documents with their chunk counts."""
    return pipeline.list_documents()
