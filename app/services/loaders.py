"""Document loaders: turn uploaded bytes into plain text.

Supported: ``.txt`` / ``.md`` / ``.rst`` / ``.csv`` / ``.json`` (any UTF-8 text)
and ``.pdf`` (via ``pypdf``). Unknown binary types are decoded leniently so the
service never hard-crashes on ingestion; callers surface a clear error when the
extracted text is empty.
"""

from __future__ import annotations

import io
import os

from app.core.logging import get_logger

logger = get_logger(__name__)

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".text", ".rst", ".csv", ".json", ".py", ".yaml", ".yml",
}


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - depends on pdf content
            logger.warning("Failed to extract a PDF page: %s", exc)
            pages.append("")
    return "\n\n".join(p for p in pages if p.strip())


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from raw file bytes based on the filename extension."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext == ".pdf":
        return _extract_pdf(data)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")
