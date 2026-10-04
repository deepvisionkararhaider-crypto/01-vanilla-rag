"""Tests for the deterministic hashing embedder and the embedder factory."""

from __future__ import annotations

import numpy as np

from app.services.embeddings import HashingEmbedder, build_embedder


def test_embedding_shape_and_dim():
    emb = HashingEmbedder(dim=64)
    vecs = emb.embed(["hello world", "another document"])
    assert vecs.shape == (2, 64)
    assert vecs.dtype == np.float32


def test_embeddings_are_l2_normalized():
    emb = HashingEmbedder(dim=128)
    vecs = emb.embed(["the quick brown fox", "lazy dogs sleep"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embedding_is_deterministic():
    a = HashingEmbedder(dim=128).embed(["reproducibility matters"])
    b = HashingEmbedder(dim=128).embed(["reproducibility matters"])
    assert np.allclose(a, b)


def test_similar_texts_score_higher_than_unsimilar():
    emb = HashingEmbedder(dim=256)
    q, pos, neg = emb.embed([
        "refund policy for standard plans",
        "the refund policy applies to standard plan purchases",
        "quantum chromodynamics describes the strong nuclear force",
    ])
    sim_pos = float(q @ pos)
    sim_neg = float(q @ neg)
    assert sim_pos > sim_neg


def test_empty_input_returns_empty_matrix():
    emb = HashingEmbedder(dim=32)
    assert emb.embed([]).shape == (0, 32)


def test_factory_explicit_hashing():
    emb = build_embedder("hashing")
    assert emb.name == "hashing"
    assert emb.dim > 0


def test_factory_auto_runs_without_ml_deps():
    # sentence-transformers is not installed in the test env, so auto must
    # transparently fall back to the hashing embedder rather than crash.
    emb = build_embedder("auto")
    assert emb.name in ("hashing", "sentence-transformers")
    assert emb.embed(["smoke test"]).shape[0] == 1
