"""Seed the vector index with the bundled sample documents.

This runs the *real* pipeline offline (no server required) and persists the
result to ``DATA_DIR/documents.json``. Start the API afterwards and it reloads
the persisted index on boot.

Usage:
    python scripts/seed.py
    EMBEDDING_PROVIDER=hashing VECTOR_STORE=numpy python scripts/seed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script (scripts/ is not a package).
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.services.rag_pipeline import RAGPipeline  # noqa: E402

logger = get_logger("seed")
SAMPLE_DIR = _REPO_ROOT / "app" / "data"


def main() -> None:
    setup_logging(get_settings().log_level)
    files = sorted(
        p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in {".txt", ".md"}
    ) if SAMPLE_DIR.exists() else []

    if not files:
        logger.warning("No sample files found in %s", SAMPLE_DIR)
        return

    pipeline = RAGPipeline(get_settings())
    for path in files:
        text = path.read_text(encoding="utf-8")
        result = pipeline.ingest_text(text, source_id=path.stem, metadata={"seeded": True})
        logger.info("Seeded '%s' -> %d chunks", result.source_id, result.num_chunks)

    stats = pipeline.stats()
    logger.info(
        "Index ready: %d documents / %d chunks (embedder=%s, store=%s, generator=%s)",
        stats["documents"],
        stats["chunks"],
        stats["embedding_provider"],
        stats["vector_store"],
        stats["llm_provider"],
    )


if __name__ == "__main__":
    main()
