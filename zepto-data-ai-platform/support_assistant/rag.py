"""Ingestion -> chunking -> embedding -> ChromaDB storage -> retrieval.

Embeddings: sentence-transformers `all-MiniLM-L6-v2`, computed locally (no API key).
Vector store: a persistent ChromaDB collection `zepto_policies` using cosine distance.
Retrieval always runs for real, in both MOCK_LLM modes.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import chromadb

HERE = Path(__file__).resolve().parent
DOCS_DIR = Path(os.getenv("DOCS_DIR", HERE / "docs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", HERE / "chroma_db"))
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCES_PER_CHUNK = 1
TOP_K = 3

DOC_TITLES = {
    "doc_01": "Delivery Policy", "doc_02": "Returns & Refunds", "doc_03": "Membership Tiers",
    "doc_04": "Order Tracking", "doc_05": "Order Cancellation Policy", "doc_06": "Damaged or Missing Items",
    "doc_07": "Gift Cards", "doc_08": "Customer Support Hours",
}

log = logging.getLogger(__name__)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class MiniLMEmbedder:
    """Thin wrapper around sentence-transformers so the rest of the code sees plain lists."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        from sentence_transformers import SentenceTransformer   # heavy import, done lazily
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).tolist()


@lru_cache(maxsize=1)
def default_embedder() -> Embedder:
    return MiniLMEmbedder()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class Retrieved:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    similarity: float          # cosine similarity = 1 - cosine distance


# ---------------------------------------------------------------- ingestion
def load_documents(docs_dir: Path = DOCS_DIR) -> dict[str, str]:
    docs = {p.stem: p.read_text(encoding="utf-8").strip() for p in sorted(docs_dir.glob("doc_*.txt"))}
    if len(docs) != 8:
        raise RuntimeError(f"Expected 8 corpus documents in {docs_dir}, found {len(docs)}")
    return docs


def chunk_document(doc_id: str, text: str, sentences_per_chunk: int = SENTENCES_PER_CHUNK) -> list[Chunk]:
    """Fixed-size chunking by sentence (1 sentence per chunk, no overlap).

    The documents are short (55-89 words), but each packs 3-4 separate rules,
    one per sentence. Sentence chunks keep every rule intact and make the *top*
    chunk specific enough that its first ~200 characters actually contain the
    answer, which matters for the mock "Based on the retrieved context" template.
    (With whole-document or 2-sentence chunks, e.g. the INR 25 fee rule sat past
    character 190 of its chunk and was cut off by the snippet.)
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return [Chunk(chunk_id=f"{doc_id}#c{i // sentences_per_chunk}", doc_id=doc_id,
                  text=" ".join(sentences[i:i + sentences_per_chunk]))
            for i in range(0, len(sentences), sentences_per_chunk)]


def chunk_corpus(docs: dict[str, str]) -> list[Chunk]:
    return [c for doc_id, text in docs.items() for c in chunk_document(doc_id, text)]


# ---------------------------------------------------------------- vector store
class PolicyIndex:
    """Owns the ChromaDB collection: builds it from the corpus and serves top-k queries."""

    def __init__(self, embedder: Embedder | None = None, persist_dir: Path | None = CHROMA_DIR,
                 docs_dir: Path = DOCS_DIR):
        self.embedder = embedder or default_embedder()
        self.client = (chromadb.PersistentClient(path=str(persist_dir)) if persist_dir
                       else chromadb.EphemeralClient())
        self.docs_dir = docs_dir
        self.collection = None

    def build(self) -> int:
        """(Re)build the collection from scratch so it always matches the files on disk."""
        chunks = chunk_corpus(load_documents(self.docs_dir))
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:        # collection did not exist yet
            pass
        self.collection = self.client.create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}, embedding_function=None)
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=self.embedder.encode([c.text for c in chunks]),
            metadatas=[{"doc_id": c.doc_id, "title": DOC_TITLES.get(c.doc_id, c.doc_id)} for c in chunks],
        )
        log.info("Indexed %d chunks from %d documents into '%s'", len(chunks),
                 len({c.doc_id for c in chunks}), COLLECTION_NAME)
        return len(chunks)

    def count(self) -> int:
        return self.collection.count() if self.collection else 0

    def retrieve(self, query: str, k: int = TOP_K) -> list[Retrieved]:
        if self.collection is None:
            raise RuntimeError("Index not built - call build() first")
        res = self.collection.query(query_embeddings=self.embedder.encode([query]), n_results=k,
                                    include=["documents", "metadatas", "distances"])
        return [Retrieved(chunk_id=cid, doc_id=meta["doc_id"], title=meta["title"], text=doc,
                          similarity=round(1.0 - dist, 4))
                for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                                res["metadatas"][0], res["distances"][0])]
