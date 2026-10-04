"""Application configuration via environment variables (pydantic-settings).

Every tunable knob of the RAG pipeline is configurable without touching code:
embedding backend, vector index, chunking sizes, retrieval depth, and the
generation provider. Secrets (API keys) are read from the environment only and
are never committed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Retrieval / indexing ---
    embedding_provider: str = "auto"  # auto | hashing | sentence-transformers
    embedding_model_name: str = "all-MiniLM-L6-v2"
    vector_store: str = "auto"  # auto | numpy | faiss
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    # --- Generation ---
    llm_provider: str = "extractive"  # extractive | openai | huggingface
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4o-mini"
    huggingface_api_key: str = ""
    huggingface_model: str = "meta-llama/Llama-3.2-3B-Instruct"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"
    data_dir: str = "./data"
    index_dir: str = "./index"
    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def index_path(self) -> Path:
        return Path(self.index_dir)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
