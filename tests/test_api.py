"""End-to-end API tests over HTTP (FastAPI TestClient)."""

from __future__ import annotations


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["embedding_provider"] == "hashing"
    assert body["vector_store"] == "numpy"
    assert body["llm_provider"] == "extractive"
    assert body["documents"] == 0


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "vanilla-rag"


def test_ingest_text(client, sample_policy):
    r = client.post("/ingest/text", json={"text": sample_policy, "source_id": "policy"})
    assert r.status_code == 200
    body = r.json()
    assert body["source_id"] == "policy"
    assert body["num_chunks"] > 1
    assert body["embedding_provider"] == "hashing"


def test_ingest_then_query_returns_grounded_cited_answer(client, sample_policy):
    client.post("/ingest/text", json={"text": sample_policy, "source_id": "policy"})
    r = client.post("/query", json={"question": "What is the refund window?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["retrieved"] >= 1
    assert len(body["citations"]) == body["retrieved"]
    # The lexical retriever should surface refund content among the top passages.
    assert any("refund" in c["text"].lower() for c in body["citations"])
    # Citations are ordered by descending similarity score.
    scores = [c["score"] for c in body["citations"]]
    assert scores == sorted(scores, reverse=True)
    # Answer is grounded and carries an inline citation marker.
    assert "[" in body["answer"]
    assert body["timings_ms"]["total"] >= 0


def test_query_on_empty_index_is_graceful(client):
    r = client.post("/query", json={"question": "anything at all?"})
    assert r.status_code == 200
    body = r.json()
    assert body["retrieved"] == 0
    assert body["citations"] == []
    assert "couldn't find" in body["answer"].lower()


def test_query_rejects_malformed_input(client):
    # Missing required field -> 422 from Pydantic.
    assert client.post("/query", json={}).status_code == 422
    # Empty question violates min_length -> 422.
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_ingest_rejects_blank_text(client):
    # Whitespace passes Pydantic min_length but the pipeline refuses it -> 400.
    r = client.post("/ingest/text", json={"text": "    \n  "})
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_documents_listing(client, sample_policy):
    client.post("/ingest/text", json={"text": sample_policy, "source_id": "policy"})
    r = client.get("/documents")
    assert r.status_code == 200
    docs = r.json()
    assert any(d["source_id"] == "policy" for d in docs)
    assert docs[0]["num_chunks"] > 0


def test_source_filter_restricts_retrieval(client, sample_policy):
    client.post("/ingest/text", json={"text": sample_policy, "source_id": "policy"})
    client.post(
        "/ingest/text",
        json={"text": "Unrelated cooking recipes and tips.", "source_id": "recipes"},
    )
    r = client.post("/query", json={"question": "refund", "top_k": 5, "source_filter": "policy"})
    assert r.status_code == 200
    for c in r.json()["citations"]:
        assert c["source_id"] == "policy"


def test_file_upload_ingest(client, sample_policy):
    files = {"file": ("note.txt", sample_policy.encode("utf-8"), "text/plain")}
    r = client.post("/ingest/file", files=files)
    assert r.status_code == 200
    assert r.json()["num_chunks"] > 0


def test_file_upload_rejects_empty(client):
    files = {"file": ("empty.txt", b"", "text/plain")}
    r = client.post("/ingest/file", files=files)
    assert r.status_code == 400
