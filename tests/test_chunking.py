"""Tests for the recursive chunker."""

from __future__ import annotations

import pytest

from app.services.chunking import chunk_text


def test_short_text_is_single_chunk():
    chunks = chunk_text("Hello world.", chunk_size=200, chunk_overlap=40)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].index == 0


def test_empty_and_whitespace_text_yield_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_long_text_produces_multiple_ordered_chunks():
    text = " ".join(f"Sentence number {i} about topic {i % 7}." for i in range(120))
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_size_is_respected_on_the_emitted_span():
    text = " ".join(f"word{i}" for i in range(500))
    size, overlap = 100, 20
    chunks = chunk_text(text, chunk_size=size, chunk_overlap=overlap)
    for c in chunks:
        # A single atomic can push one chunk over by at most ~overlap + one token.
        assert len(c.text) <= size + overlap + 20


def test_adjacent_chunks_overlap():
    text = " ".join(f"tok{i}" for i in range(400))
    chunks = chunk_text(text, chunk_size=60, chunk_overlap=20)
    assert len(chunks) >= 2
    # The end of one chunk should overlap the start of the next.
    a, b = chunks[0], chunks[1]
    assert b.char_start < a.char_end


def test_char_spans_are_valid_and_within_document():
    text = "Alpha. Beta. Gamma. Delta. Epsilon. Zeta."
    chunks = chunk_text(text, chunk_size=12, chunk_overlap=3)
    for c in chunks:
        assert 0 <= c.char_start < c.char_end <= len(text)
        assert text[c.char_start : c.char_end].strip() == c.text


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=100, chunk_overlap=-1)


def test_metadata_is_attached():
    chunks = chunk_text("Some text here.", chunk_size=200, chunk_overlap=10,
                        base_metadata={"source_id": "doc1"})
    assert chunks[0].metadata["source_id"] == "doc1"
    assert "char_start" in chunks[0].metadata
