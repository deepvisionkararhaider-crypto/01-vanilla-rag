"""Shared pytest fixtures.

Tests run fully offline and deterministically by forcing the hashing embedder,
numpy vector store, and extractive generator, with an isolated temp data dir so
no state leaks between tests or touches the repo.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app
from app.services.rag_pipeline import RAGPipeline, get_pipeline

SAMPLE_POLICY = (
    "Acme Corp Refund and Support Policy\n\n"
    "Refunds are available within 30 days of purchase for all standard plans. "
    "To request a refund, contact support with your order number. Refunds are "
    "processed to the original payment method within 5 to 7 business days.\n\n"
    "Enterprise contracts have custom refund terms defined in the master services "
    "agreement. Annual enterprise plans may be prorated at Acme's discretion.\n\n"
    "Technical support is available 24/7 for enterprise customers via phone and "
    "email. Standard plan customers receive email support with a response target "
    "of one business day.\n\n"
    "Warranty covers manufacturing defects for twelve months from the date of "
    "delivery. Damage from misuse or unauthorized repair is not covered."
)


@pytest.fixture
def sample_policy() -> str:
    return SAMPLE_POLICY


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        embedding_provider="hashing",
        vector_store="numpy",
        llm_provider="extractive",
        chunk_size=200,
        chunk_overlap=40,
        top_k=3,
        data_dir=str(tmp_path / "data"),
        index_dir=str(tmp_path / "index"),
        cors_origins="*",
    )


@pytest.fixture
def pipeline(settings) -> RAGPipeline:
    return RAGPipeline(settings)


@pytest.fixture
def client(pipeline):
    # Override the singleton so route handlers use the isolated test pipeline.
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    # NOTE: TestClient is used without the context manager on purpose, so the
    # app's lifespan (which would warm a separate default pipeline) is not run.
    yield TestClient(app)
    app.dependency_overrides.clear()
