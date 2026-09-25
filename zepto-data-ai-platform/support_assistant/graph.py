"""LangGraph StateGraph: classify_intent -> (retrieve_and_answer | direct_answer) -> END.

Only the *generation* step inside each node branches on MOCK_LLM. The routing
(conditional edge) and the retrieval (embedding + ChromaDB) behave identically
in both modes.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

import llm
from prompts import CLASSIFY_PROMPT_TEMPLATE, CORRECTIVE_INSTRUCTION, GENERAL_PROMPT_TEMPLATE, render_answer_prompt
from rag import PolicyIndex, Retrieved
from schemas import AskResponse

log = logging.getLogger(__name__)

POLICY_KEYWORDS = ("delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours")
MOCK_GENERAL_ANSWER = "I can only answer questions about Zepto policies right now."
MOCK_CONFIDENCE = 1.0
SNIPPET_CHARS = 200
MAX_EXTRA_RETRIES = 2          # real-LLM path: 1 attempt + up to 2 corrective retries

Intent = Literal["policy_question", "general_question"]


class AssistantState(TypedDict, total=False):
    query: str
    intent: Intent
    retrieved: list[Retrieved]
    response: dict            # validated AskResponse, as a dict
    route_log: list[str]      # node names visited, for transparency in tests / logs


# ------------------------------------------------------------------ helpers
def keyword_intent(query: str) -> Intent:
    q = query.lower()
    return "policy_question" if any(k in q for k in POLICY_KEYWORDS) else "general_question"


def make_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]            # do not cut a word in half
    return cut.rstrip(",;:") + "..."


def _extract_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)  # tolerate stray text/code fences
    if not match:
        raise ValueError("no JSON object found")
    return json.loads(match.group(0))


def generate_validated(prompt: str, allowed_sources: set[str]) -> AskResponse:
    """Real-LLM path: validate against AskResponse, retrying up to 2 extra times."""
    messages = [{"role": "user", "content": prompt}]
    last_error = "unknown"
    for attempt in range(1 + MAX_EXTRA_RETRIES):
        try:
            raw = llm.chat(messages)
        except llm.LLMError as exc:
            last_error = str(exc)
            log.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
            continue
        try:
            resp = AskResponse.model_validate(_extract_json(raw))
            resp.sources = [s for s in resp.sources if s in allowed_sources]   # no invented citations
            return resp
        except (ValueError, ValidationError) as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:150]}"
            log.warning("LLM output failed validation (attempt %d): %s", attempt + 1, last_error)
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": CORRECTIVE_INSTRUCTION.format(error=last_error)}]
    return AskResponse(answer=f"[ERROR] The language model did not return a valid response after "
                              f"{1 + MAX_EXTRA_RETRIES} attempts ({last_error}).", sources=[], confidence=0.0)


# ------------------------------------------------------------------ graph
def build_graph(index: PolicyIndex):
    def classify_intent(state: AssistantState) -> AssistantState:
        query = state["query"]
        if llm.is_mock():                                   # graded baseline: keyword heuristic, no LLM
            intent = keyword_intent(query)
        else:                                               # optional extension
            try:
                raw = llm.chat([{"role": "user", "content": CLASSIFY_PROMPT_TEMPLATE.format(question=query)}],
                               max_tokens=5).strip().lower()
                intent = raw if raw in ("policy_question", "general_question") else keyword_intent(query)
            except llm.LLMError:
                intent = keyword_intent(query)
        return {"intent": intent, "route_log": state.get("route_log", []) + ["classify_intent"]}

    def retrieve_and_answer(state: AssistantState) -> AssistantState:
        chunks = index.retrieve(state["query"])            # always real: MiniLM embedding + ChromaDB top-3
        sources = [c.chunk_id for c in chunks]
        if llm.is_mock():
            top = chunks[0]
            resp = AskResponse(answer=f"Based on the retrieved context: {make_snippet(top.text)}",
                               sources=sources, confidence=MOCK_CONFIDENCE)
        else:
            resp = generate_validated(render_answer_prompt(state["query"], chunks), set(sources))
        return {"retrieved": chunks, "response": resp.model_dump(),
                "route_log": state.get("route_log", []) + ["retrieve_and_answer"]}

    def direct_answer(state: AssistantState) -> AssistantState:
        if llm.is_mock():
            resp = AskResponse(answer=MOCK_GENERAL_ANSWER, sources=[], confidence=MOCK_CONFIDENCE)
        else:
            resp = generate_validated(GENERAL_PROMPT_TEMPLATE.format(question=state["query"]), set())
            resp.sources = []
        return {"retrieved": [], "response": resp.model_dump(),
                "route_log": state.get("route_log", []) + ["direct_answer"]}

    def route(state: AssistantState) -> str:
        return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

    g = StateGraph(AssistantState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("retrieve_and_answer", retrieve_and_answer)
    g.add_node("direct_answer", direct_answer)
    g.add_edge(START, "classify_intent")
    g.add_conditional_edges("classify_intent", route,
                            {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"})
    g.add_edge("retrieve_and_answer", END)
    g.add_edge("direct_answer", END)
    return g.compile()


def answer(graph, query: str) -> tuple[AskResponse, AssistantState]:
    state = graph.invoke({"query": query, "route_log": []})
    return AskResponse.model_validate(state["response"]), state
