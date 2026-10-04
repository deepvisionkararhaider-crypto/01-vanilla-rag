"""Tests for the generation providers."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.services.generation import (
    ExtractiveProvider,
    build_llm_provider,
)


def test_extractive_answer_cites_context():
    prov = ExtractiveProvider()
    contexts = [
        "Refunds are available within 30 days of purchase. Contact support to start one.",
        "Enterprise contracts have custom refund terms in the master services agreement.",
    ]
    answer = prov.generate("What is the refund policy?", contexts)
    assert "[1]" in answer or "[2]" in answer
    assert "refund" in answer.lower()


def test_extractive_handles_empty_contexts():
    prov = ExtractiveProvider()
    answer = prov.generate("anything?", [])
    assert "couldn't find" in answer.lower()


def test_extractive_prefers_relevant_sentence():
    prov = ExtractiveProvider()
    contexts = [
        "The sky is blue. Warranty covers manufacturing defects for twelve months.",
    ]
    answer = prov.generate("How long is the warranty?", contexts)
    assert "warranty" in answer.lower()
    assert "twelve months" in answer.lower()


def test_factory_returns_extractive_by_default():
    prov = build_llm_provider(Settings(llm_provider="extractive"))
    assert isinstance(prov, ExtractiveProvider)


def test_factory_openai_requires_key():
    with pytest.raises(ValueError):
        build_llm_provider(Settings(llm_provider="openai", openai_api_key=""))


def test_factory_huggingface_requires_key():
    with pytest.raises(ValueError):
        build_llm_provider(Settings(llm_provider="huggingface", huggingface_api_key=""))
