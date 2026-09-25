"""Tests for the support assistant (run: `pytest -q` inside /support_assistant).

Most tests use a tiny deterministic bag-of-words `FakeEmbedder` so they run
offline in about a second. `test_real_minilm_retrieval` uses the real
all-MiniLM-L6-v2 model and is skipped automatically if it cannot be loaded.
"""
import hashlib
import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import graph as G  # noqa: E402
import llm  # noqa: E402
import prompts  # noqa: E402
import rag  # noqa: E402
from schemas import AskResponse  # noqa: E402

STOP = set("a an the is are to of for in on my i do can what how if it be with by and or you your".split())


class FakeEmbedder:
    """Hashed bag-of-words vectors: offline, deterministic, good enough for keyword-level retrieval."""
    dim = 256

    def encode(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if w not in STOP:
                    v[int(hashlib.md5(w.rstrip("s").encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


@pytest.fixture
def index():
    idx = rag.PolicyIndex(embedder=FakeEmbedder(), persist_dir=None)
    idx.build()
    return idx


@pytest.fixture(autouse=True)
def mock_mode_no_network(monkeypatch):
    """Default mode + fail loudly if anything tries to reach an LLM provider."""
    monkeypatch.delenv("MOCK_LLM", raising=False)

    def boom(*a, **k):
        raise AssertionError("network call to an LLM provider attempted in mock mode")
    monkeypatch.setattr(llm.requests, "post", boom)


def test_corpus_chunked_and_indexed(index):
    docs = rag.load_documents()
    assert sorted(docs) == [f"doc_0{i}" for i in range(1, 9)]
    chunks = rag.chunk_corpus(docs)
    assert index.count() == len(chunks)
    assert {c.doc_id for c in chunks} == set(docs)                     # all 8 docs are queryable
    stored = index.collection.get(include=["metadatas"])
    assert {m["doc_id"] for m in stored["metadatas"]} == set(docs)
    assert index.collection.metadata["hnsw:space"] == "cosine"


@pytest.mark.parametrize("query,intent", [
    ("What is the delivery fee?", "policy_question"),
    ("How long does a REFUND take?", "policy_question"),
    ("Can I cancel my order?", "policy_question"),
    ("Do gift cards expire?", "policy_question"),                   # substring "gift card" matches
    ("What are your support hours?", "policy_question"),
    ("What is the capital of France?", "general_question"),
    ("Tell me a joke", "general_question"),
])
def test_keyword_heuristic(query, intent):
    assert G.keyword_intent(query) == intent


def test_policy_route_mock(index):
    graph = G.build_graph(index)
    resp, state = G.answer(graph, "What is the delivery fee if my order is below INR 149?")
    assert state["intent"] == "policy_question"
    assert state["route_log"] == ["classify_intent", "retrieve_and_answer"]
    assert len(state["retrieved"]) == 3 and resp.sources == [c.chunk_id for c in state["retrieved"]]
    assert resp.sources[0].startswith("doc_01")                         # correct source document
    assert resp.answer.startswith("Based on the retrieved context: ")
    assert resp.answer.removeprefix("Based on the retrieved context: ").rstrip(".")[:40] in state["retrieved"][0].text
    assert resp.confidence == 1.0


def test_general_route_mock(index):
    graph = G.build_graph(index)
    resp, state = G.answer(graph, "What is the capital of France?")
    assert state["route_log"] == ["classify_intent", "direct_answer"]
    assert resp == AskResponse(answer=G.MOCK_GENERAL_ANSWER, sources=[], confidence=1.0)


def test_mock_is_deterministic(index):
    graph = G.build_graph(index)
    a = G.answer(graph, "How do refunds work for spoiled items?")[0]
    b = G.answer(graph, "How do refunds work for spoiled items?")[0]
    assert a == b


def test_prompt_template_structure(index):
    text = prompts.render_answer_prompt("Q?", index.retrieve("delivery fee"))
    for section in ["### ROLE", "### CONTEXT", "### TASK", "### FORMAT", "### LENGTH", "### EXAMPLE"]:
        assert section in text
    assert "Do NOT answer using any information that is not present in the provided context" in text
    assert "doc_01#c1" in text


def test_schema_rejects_bad_output():
    with pytest.raises(Exception):
        AskResponse(answer="x", sources=[], confidence=1.5)
    with pytest.raises(Exception):
        AskResponse.model_validate({"answer": "x", "sources": [], "confidence": 0.5, "extra": 1})


def test_real_llm_retry_then_success(index, monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "0")
    replies = iter(["not json at all", '{"answer": "Flat INR 25 fee.", "sources": ["doc_01#c0", "made_up#c9"], '
                                       '"confidence": 0.9}'])
    calls = []
    monkeypatch.setattr(llm, "chat", lambda messages, **k: calls.append(messages) or next(replies))
    graph = G.build_graph(index)
    resp, state = G.answer(graph, "What is the delivery fee below INR 149?")
    # classify call returns "not json at all" -> falls back to heuristic; answer call gets the JSON
    assert resp.answer == "Flat INR 25 fee." and "made_up#c9" not in resp.sources


def test_real_llm_gives_up_after_two_retries(index, monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "0")
    calls = []
    monkeypatch.setattr(llm, "chat", lambda messages, **k: calls.append(len(messages)) or "garbage")
    resp = G.generate_validated("prompt", {"doc_01#c0"})
    assert len(calls) == 1 + G.MAX_EXTRA_RETRIES          # 1 attempt + 2 corrective retries
    assert calls == [1, 3, 5]                              # each retry appends the corrective instruction
    assert resp.answer.startswith("[ERROR]") and resp.confidence == 0.0 and resp.sources == []


def test_fastapi_endpoint(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import main
    monkeypatch.setattr(rag, "default_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(rag, "CHROMA_DIR", tmp_path)
    monkeypatch.setattr(main, "PolicyIndex", lambda: rag.PolicyIndex(embedder=FakeEmbedder(), persist_dir=tmp_path))
    with TestClient(main.app) as client:
        assert client.get("/health").json()["mock_llm"] is True
        r1 = client.post("/ask", json={"query": "How long does a refund take to reach my card?"})
        r2 = client.post("/ask", json={"query": "Who won the 2011 cricket world cup?"})
        assert r1.status_code == r2.status_code == 200
        assert r1.json()["answer"].startswith("Based on the retrieved context:") and len(r1.json()["sources"]) == 3
        assert r2.json() == {"answer": G.MOCK_GENERAL_ANSWER, "sources": [], "confidence": 1.0}
        assert client.post("/ask", json={"query": ""}).status_code == 422


def _load_minilm():
    try:
        return rag.MiniLMEmbedder()
    except Exception as exc:   # no model cache and no network
        pytest.skip(f"all-MiniLM-L6-v2 not available: {exc}")


@pytest.mark.parametrize("query,doc", [
    ("What is the delivery fee if my order is below INR 149?", "doc_01"),
    ("How many days do I have to return an unopened packaged item?", "doc_02"),
    ("What do I get with Zepto Pass+ membership?", "doc_03"),
    ("Can I cancel my order after it has been packed?", "doc_05"),
    ("My order arrived with a missing item, can I get a refund?", "doc_06"),
    ("Is phone support available? What are the support hours?", "doc_08"),
])
def test_real_minilm_retrieval(query, doc, tmp_path):
    idx = rag.PolicyIndex(embedder=_load_minilm(), persist_dir=tmp_path)
    idx.build()
    assert idx.retrieve(query)[0].doc_id == doc
