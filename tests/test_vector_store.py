"""Tests for the numpy vector store and the store factory."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.vector_store import NumpyStore, build_vector_store


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) or 1.0)


def test_add_and_count():
    store = NumpyStore(dim=3)
    assert store.count() == 0
    store.add(["a", "b"], np.array([_unit([1, 0, 0]), _unit([0, 1, 0])]), [{"i": 0}, {"i": 1}])
    assert store.count() == 2


def test_search_ranks_by_cosine_similarity():
    store = NumpyStore(dim=3)
    store.add(
        ["x", "y", "z"],
        np.array([_unit([1, 0, 0]), _unit([0.9, 0.1, 0]), _unit([0, 0, 1])]),
        [{"id": "x"}, {"id": "y"}, {"id": "z"}],
    )
    results = store.search(_unit([1, 0, 0]), k=2)
    assert [r.id for r in results] == ["x", "y"]
    assert results[0].score >= results[1].score


def test_search_respects_k_and_empty_store():
    store = NumpyStore(dim=2)
    assert store.search(_unit([1, 0]), k=3) == []
    store.add(["a"], np.array([_unit([1, 0])]), [{}])
    assert len(store.search(_unit([1, 0]), k=5)) == 1
    assert store.search(_unit([1, 0]), k=0) == []


def test_search_with_predicate_filters():
    store = NumpyStore(dim=2)
    store.add(
        ["a", "b"],
        np.array([_unit([1, 0]), _unit([1, 0])]),
        [{"source_id": "doc1"}, {"source_id": "doc2"}],
    )
    results = store.search(_unit([1, 0]), k=5, predicate=lambda p: p["source_id"] == "doc2")
    assert len(results) == 1
    assert results[0].id == "b"


def test_clear_resets_store():
    store = NumpyStore(dim=2)
    store.add(["a"], np.array([_unit([1, 0])]), [{}])
    store.clear()
    assert store.count() == 0


def test_add_rejects_wrong_dim():
    store = NumpyStore(dim=3)
    with pytest.raises(ValueError):
        store.add(["a"], np.array([[1.0, 0.0]]), [{}])


def test_factory_prefers_numpy_when_requested():
    store = build_vector_store("numpy", dim=16)
    assert isinstance(store, NumpyStore)
    assert store.dim == 16
