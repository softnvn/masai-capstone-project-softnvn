"""FastAPI wrapper around the LangGraph support assistant.

    uvicorn main:app --host 0.0.0.0 --port 7860
    curl -X POST localhost:7860/ask -H 'Content-Type: application/json' -d '{"query": "What is the delivery fee?"}'
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

import llm
from graph import answer, build_graph
from rag import COLLECTION_NAME, EMBED_MODEL_NAME, PolicyIndex
from schemas import AskRequest, AskResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("support_assistant")
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    index = PolicyIndex()
    n = index.build()                      # ingest + chunk + embed + store, once at startup
    STATE.update(index=index, graph=build_graph(index), n_chunks=n)
    log.info("Ready: %d chunks in '%s' | MOCK_LLM mode = %s", n, COLLECTION_NAME, llm.is_mock())
    yield
    STATE.clear()


app = FastAPI(title="Zepto Support Assistant", version="1.0.0",
              description="RAG over Zepto policy documents, orchestrated with LangGraph.", lifespan=lifespan)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    resp, state = answer(STATE["graph"], req.query)
    log.info("query=%r intent=%s route=%s sources=%s", req.query, state["intent"], state["route_log"], resp.sources)
    return resp


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mock_llm": llm.is_mock(), "collection": COLLECTION_NAME,
            "chunks_indexed": STATE.get("n_chunks", 0), "embedding_model": EMBED_MODEL_NAME}
