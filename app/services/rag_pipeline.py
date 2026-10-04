"""The RAG pipeline orchestrator.

Wires together: loaders -> chunking -> embeddings -> vector store -> retrieval ->
generation, and adds citations, timings, and lightweight JSON persistence.

Persistence stores *document text + chunk metadata* (not vectors) to
``DATA_DIR/documents.json``. On startup the index is rebuilt by re-embedding the
stored chunks. This keeps the on-disk format independent of the embedding backend
(so you can switch from hashing to sentence-transformers without corrupting an
index) at the cost of a re-embed on boot, which is instant for hashing and cached
for transformer models.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.schemas.rag import Citation, DocumentInfo, IngestResponse, QueryResponse
from app.services.chunking import chunk_text
from app.services.embeddings import build_embedder
from app.services.generation import build_llm_provider
from app.services.loaders import extract_text
from app.services.retrieval import Retriever
from app.services.vector_store import build_vector_store

logger = get_logger(__name__)


class RAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = build_embedder(
            self.settings.embedding_provider, self.settings.embedding_model_name
        )
        self.store = build_vector_store(self.settings.vector_store, self.embedder.dim)
        self.llm = build_llm_provider(self.settings)
        self.retriever = Retriever(self.embedder, self.store)

        self._documents: dict[str, dict[str, Any]] = {}
        self._persist_path: Path = self.settings.data_path / "documents.json"
        self._load()

        logger.info(
            "RAGPipeline ready | embedder=%s(dim=%d) store=%s llm=%s | docs=%d chunks=%d",
            self.embedder.name,
            self.embedder.dim,
            self.store.name,
            self.llm.name,
            len(self._documents),
            self.store.count(),
        )

    # ------------------------------------------------------------------ ingest
    def ingest_text(
        self, text: str, source_id: str | None = None, metadata: dict | None = None
    ) -> IngestResponse:
        if not text or not text.strip():
            raise ValueError("Document text is empty; nothing to ingest.")

        metadata = dict(metadata or {})
        source_id = source_id or f"doc-{uuid.uuid4().hex[:8]}"
        document_id = uuid.uuid4().hex

        chunks = chunk_text(
            text,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            base_metadata={"source_id": source_id, **metadata},
        )
        if not chunks:
            raise ValueError("Document produced no chunks after splitting.")

        records: list[dict[str, Any]] = []
        for ch in chunks:
            records.append(
                {
                    "chunk_id": f"{document_id}::{ch.index}",
                    "document_id": document_id,
                    "source_id": source_id,
                    "index": ch.index,
                    "text": ch.text,
                    "metadata": {"char_start": ch.char_start, "char_end": ch.char_end, **metadata},
                }
            )

        self.retriever.add(records)
        self._documents[source_id] = {
            "document_id": document_id,
            "source_id": source_id,
            "num_chunks": len(records),
            "metadata": metadata,
            "chunks": [
                {
                    "chunk_id": r["chunk_id"],
                    "index": r["index"],
                    "text": r["text"],
                    "metadata": r["metadata"],
                }
                for r in records
            ],
        }
        self._persist()

        return IngestResponse(
            document_id=document_id,
            source_id=source_id,
            num_chunks=len(records),
            embedding_provider=self.embedder.name,
            vector_store=self.store.name,
        )

    def ingest_file(
        self, filename: str, data: bytes, source_id: str | None = None,
        metadata: dict | None = None,
    ) -> IngestResponse:
        text = extract_text(filename, data)
        if not text.strip():
            raise ValueError(f"No text could be extracted from '{filename}'.")
        meta = dict(metadata or {})
        meta.setdefault("filename", filename)
        return self.ingest_text(text, source_id=source_id or filename, metadata=meta)

    # ------------------------------------------------------------------- query
    def query(
        self, question: str, top_k: int | None = None, source_filter: str | None = None
    ) -> QueryResponse:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        k = top_k or self.settings.top_k
        t0 = time.perf_counter()
        results = self.retriever.search(question, k, source_filter=source_filter)
        t1 = time.perf_counter()

        citations = [
            Citation(
                chunk_id=r.payload.get("chunk_id", r.id),
                source_id=r.payload.get("source_id", ""),
                chunk_index=int(r.payload.get("index", 0)),
                score=round(r.score, 6),
                text=r.payload.get("text", ""),
            )
            for r in results
        ]
        contexts = [r.payload.get("text", "") for r in results]

        answer = self.llm.generate(question, contexts)
        t2 = time.perf_counter()

        return QueryResponse(
            question=question,
            answer=answer,
            citations=citations,
            llm_provider=self.llm.name,
            retrieved=len(results),
            timings_ms={
                "retrieval": round((t1 - t0) * 1000, 2),
                "generation": round((t2 - t1) * 1000, 2),
                "total": round((t2 - t0) * 1000, 2),
            },
        )

    # -------------------------------------------------------------- inspection
    def list_documents(self) -> list[DocumentInfo]:
        return [
            DocumentInfo(
                source_id=d["source_id"],
                document_id=d["document_id"],
                num_chunks=d["num_chunks"],
                metadata=d.get("metadata", {}),
            )
            for d in self._documents.values()
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "documents": len(self._documents),
            "chunks": self.store.count(),
            "embedding_provider": self.embedder.name,
            "embedding_model": self.embedder.model_name,
            "embedding_dim": self.embedder.dim,
            "vector_store": self.store.name,
            "llm_provider": self.llm.name,
        }

    # ------------------------------------------------------------- persistence
    def _persist(self) -> None:
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(
                json.dumps({"documents": self._documents}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - filesystem dependent
            logger.warning("Failed to persist documents: %s", exc)

    def _load(self) -> None:
        if not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            documents = raw.get("documents", {})
            for source_id, doc in documents.items():
                records = []
                for ch in doc.get("chunks", []):
                    records.append(
                        {
                            "chunk_id": ch["chunk_id"],
                            "document_id": doc["document_id"],
                            "source_id": source_id,
                            "index": ch["index"],
                            "text": ch["text"],
                            "metadata": ch.get("metadata", {}),
                        }
                    )
                if records:
                    self.retriever.add(records)
                self._documents[source_id] = doc
            logger.info("Restored %d document(s) from disk.", len(documents))
        except Exception as exc:  # pragma: no cover - corrupt/absent file
            logger.warning("Failed to load persisted documents: %s", exc)


_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Process-wide singleton used by the API layer."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(get_settings())
    return _pipeline
