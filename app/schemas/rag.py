"""API data models.

These define the contract for ingestion, querying, and introspection. Every
query response carries machine-readable ``citations`` so the UI (and any
downstream consumer) can attribute each grounded statement to a specific
source chunk with its similarity score.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw document text to ingest.")
    source_id: str | None = Field(
        None, description="Optional stable identifier for the document."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    source_id: str
    num_chunks: int
    embedding_provider: str
    vector_store: str


class Citation(BaseModel):
    chunk_id: str
    source_id: str
    chunk_index: int
    score: float = Field(..., description="Cosine similarity between query and chunk.")
    text: str = Field(..., description="The retrieved passage backing the answer.")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=20)
    source_filter: str | None = Field(
        None, description="Restrict retrieval to a single source_id."
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    llm_provider: str
    retrieved: int
    timings_ms: dict[str, float] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    source_id: str
    document_id: str
    num_chunks: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    vector_store: str
    llm_provider: str
    documents: int
    chunks: int
