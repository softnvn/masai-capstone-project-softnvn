# /support_assistant — Grounded GenAI support assistant for Zepto policies

Part of the **Zepto Data & AI Platform** capstone. This is a small but complete RAG service:
8 Zepto policy documents → sentence chunks → `all-MiniLM-L6-v2` embeddings → ChromaDB → a LangGraph intent router
→ a Pydantic-validated JSON answer → FastAPI `POST /ask` → Docker.

**Graded baseline = mock mode.** `MOCK_LLM` is left unset (or set to `1`). In that mode no LLM provider is contacted,
no API key is needed, and no signup is required. Embedding and retrieval still run for real, locally.

## Files

| File | Responsibility |
|---|---|
| `docs/doc_01.txt … doc_08.txt` | The 8 corpus documents, copied verbatim from the brief |
| `rag.py` | **Ingestion, chunking, embedding, storage, retrieval**: `load_documents`, `chunk_document`, `MiniLMEmbedder`, `PolicyIndex.build()` / `.retrieve()` |
| `prompts.py` | Structured prompt templates (role–context–task–format–length, negative constraints, few-shot example) and the corrective retry instruction |
| `llm.py` | The single `MOCK_LLM` toggle (`is_mock()`) and the optional Groq client (`chat()`) |
| `graph.py` | LangGraph `StateGraph` with a `TypedDict` state and nodes `classify_intent`, `retrieve_and_answer` and `direct_answer`, plus the conditional edge and the validate-and-retry logic |
| `schemas.py` | Pydantic `AskRequest {query}` and `AskResponse {answer, sources, confidence}` |
| `main.py` | FastAPI app: builds the index at startup, serves `POST /ask` and `GET /health` |
| `demo_calls.py` | Sends example calls to the running server and records the raw JSON into this README |
| `Dockerfile`, `.dockerignore` | Container that serves `/ask` on port 7860 |
| `tests/test_assistant.py` | 22 tests: routing, mock templates, no-network guard, schema, retry logic, API, real-MiniLM retrieval |

## How to run locally

```bash
cd support_assistant
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # pulls torch via sentence-transformers

# terminal 1: start the API (MOCK_LLM unset = graded mock mode)
uvicorn main:app --host 0.0.0.0 --port 7860
# first start downloads all-MiniLM-L6-v2 (~90 MB) from Hugging Face once, then it is cached

# terminal 2: record the example calls into this README
python demo_calls.py --url http://127.0.0.1:7860

# or call it by hand
curl -s -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "What is the delivery fee if my order is below INR 149?"}'

pytest -q                                                # tests (the real-MiniLM retrieval tests need the model cached)
```
Interactive API docs are served at `http://127.0.0.1:7860/docs`.

## Docker (graded baseline: local build + run)

```bash
cd support_assistant
docker build -t zepto-support-assistant .
docker run --rm -p 7860:7860 zepto-support-assistant          # MOCK_LLM=1 is baked in as the default
curl -s -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d '{"query": "Can I cancel my order after it has been packed?"}'
```
The image installs CPU-only PyTorch (a much smaller image) and **pre-downloads the embedding model at build time**, so the
running container needs no network. It runs as uid 1000, which is also what Hugging Face Spaces expects. The command is
`uvicorn main:app --host 0.0.0.0 --port 7860`. There is also a `HEALTHCHECK` on `/health`.
Optional real-LLM mode in Docker: `docker run --rm -p 7860:7860 -e MOCK_LLM=0 -e GROQ_API_KEY=... zepto-support-assistant`.

## Architecture: the RAG pipeline, stage by stage

```
 docs/doc_01..08.txt
        │  (1) INGESTION         rag.load_documents()  -> rag.chunk_document()   1 sentence = 1 chunk, id "doc_0X#cN"
        ▼
 26 chunks  ─(2) EMBEDDING──►  rag.MiniLMEmbedder.encode()   all-MiniLM-L6-v2, 384-d, L2-normalised
        │
        ▼
 ChromaDB collection "zepto_policies"  (hnsw:space = cosine, persisted in ./chroma_db, rebuilt at startup)
        ▲
        │ (3) RETRIEVAL   top-3 by cosine similarity   ◄── runs for real in BOTH modes
 POST /ask {query} ─► main.ask() ─► LangGraph StateGraph
                                   ┌───────────────────┐
                                   │  classify_intent   │  mock: keyword heuristic │ real: LLM label
                                   └─────────┬─────────┘
                       policy_question       │ conditional edge │ general_question
                   ┌─────────────────────────┴──────────────────┐
                   ▼                                            ▼
      ┌─────────────────────────┐                  ┌──────────────────────┐
      │  retrieve_and_answer     │ (3)+(4)          │  direct_answer        │ (4)
      │  PolicyIndex.retrieve()  │                  │  no retrieval         │
      │  mock: canned template   │                  │  mock: fixed string   │
      │  real: ANSWER_PROMPT     │                  │  real: GENERAL_PROMPT │
      └────────────┬────────────┘                  └──────────┬───────────┘
                   └────────────► AskResponse (Pydantic) ◄────┘ ─► JSON {answer, sources, confidence}
```

1. **Ingestion.** `rag.load_documents()` reads the 8 files. `rag.chunk_document()` splits each document into sentence
   chunks (1 sentence per chunk, 26 chunks in total, ids like `doc_01#c1`). The documents are short, but each sentence is a
   separate rule (fee, time limit, exception). Sentence chunks make the *top* hit specific enough that its first 200
   characters contain the answer, which the mock template depends on. Whole-document chunks would cut off the answer. For
   example, the "INR 25 fee" rule sits past character 190 of the delivery document.
2. **Embedding.** `rag.MiniLMEmbedder` wraps `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` on CPU with
   normalised embeddings. `PolicyIndex.build()` embeds every chunk and adds it to the ChromaDB collection **`zepto_policies`**,
   along with metadata (`doc_id`, `title`). The collection is created with `hnsw:space = cosine` and rebuilt at every startup,
   so it always matches the files on disk.
3. **Retrieval.** The LangGraph node **`retrieve_and_answer`** calls `PolicyIndex.retrieve(query, k=3)`. That embeds the
   query with the same model and runs `collection.query(query_embeddings=..., n_results=3)`. Similarity is reported as
   1 − cosine distance. This stage **never branches on `MOCK_LLM`**.
4. **Generation.** Also inside **`retrieve_and_answer`** (for policy questions) or **`direct_answer`** (for general ones).
   The result is always validated into the Pydantic `AskResponse` before FastAPI returns it.

**Routing.** `classify_intent` writes `intent` into the `TypedDict` state. `add_conditional_edges("classify_intent", route, …)`
sends `policy_question` to `retrieve_and_answer` and `general_question` to `direct_answer`. Both then go to `END`. The routing
function only reads the state, so it is identical in both modes.

**What `MOCK_LLM` changes.** Only the generation step inside each node branches on it:

| Stage / node | Default (`MOCK_LLM` unset or `1`, graded) | Optional `MOCK_LLM=0` |
|---|---|---|
| `classify_intent` | Keyword heuristic: `policy_question` if the lowercased query contains any of `delivery, return, refund, membership, tracking, cancel, gift card, support hours`, otherwise `general_question`. No LLM call | `CLASSIFY_PROMPT_TEMPLATE` sent to the LLM. Falls back to the heuristic if the reply is not a valid label |
| Retrieval (in `retrieve_and_answer`) | Real MiniLM + ChromaDB top-3 | Same, unchanged |
| Answer (in `retrieve_and_answer`) | `f"Based on the retrieved context: {top_chunk_snippet}"` (first ~200 chars of the most similar chunk, cut on a word boundary); `sources` = the 3 retrieved chunk ids; `confidence` = 1.0 | `ANSWER_PROMPT_TEMPLATE` with the 3 chunks, then JSON parsed and validated against `AskResponse`, **retried up to 2 more times** with a corrective instruction, then a clearly marked `[ERROR]` response (`confidence` 0.0). Sources the model invents are filtered out |
| `direct_answer` | Fixed string `"I can only answer questions about Zepto policies right now."`, `sources = []`, `confidence` = 1.0 | `GENERAL_PROMPT_TEMPLATE` sent to the LLM, same validation and retry; `sources` forced to `[]` |

In mock mode `llm.chat()` is never reached. The test suite enforces this by replacing `requests.post` with a function
that fails the test if it is ever called.

## Structured prompt template (`prompts.py`, used by the optional `MOCK_LLM=0` path)

```text
### ROLE
You are ZippyHelp, Zepto's customer-support assistant. You answer questions about Zepto's
delivery, returns, membership, tracking, cancellation, damaged-item, gift-card and support
policies for customers in India. You are precise, friendly and never speculate.

### CONTEXT
The ONLY facts you may use are the policy excerpts below. Each excerpt starts with its chunk id.
{context}

### TASK
Answer the customer's question using only the policy excerpts above.
- Do NOT answer using any information that is not present in the provided context, even if you
  believe you know Zepto's policy from elsewhere.
- Do NOT invent prices, time limits, fees or policy names.
- If the context does not contain the answer, say exactly: "I don't have that information in
  Zepto's policy documents." and return an empty sources list with confidence 0.0.
- Cite, in "sources", only the chunk ids you actually used.

### EXAMPLE
Customer question: "Is there a fee if my order is below INR 149?"
Policy excerpts:
[doc_01#c1] (Delivery Policy) Standard delivery is free on orders over INR 149; orders below this
threshold incur a flat INR 25 delivery fee.
[doc_01#c2] (Delivery Policy) Priority delivery, which reserves the next available rider slot, is
available at checkout for an additional INR 15.
Correct output:
{"answer": "Yes. Orders below INR 149 have a flat INR 25 delivery fee; standard delivery is free on orders over INR 149.", "sources": ["doc_01#c1"], "confidence": 0.95}
(doc_01#c2 is not cited because it was not needed for the answer.)

### FORMAT
Return ONLY one JSON object, no markdown fences and no text before or after it, with exactly these keys:
{"answer": string, "sources": list of chunk-id strings, "confidence": number between 0 and 1}

### LENGTH
Keep "answer" to at most 3 sentences (about 60 words).

### CUSTOMER QUESTION
{question}
```
Skeleton: **Role** (ZippyHelp persona), **Context** (retrieved chunks with ids), **Task** (with **negative constraints**:
"Do NOT answer using any information that is not present in the provided context", "Do NOT invent prices…"), a **few-shot
example** that also shows *not* citing an unused chunk, **Format** (strict JSON matching `AskResponse`) and **Length** (≤ 3 sentences).

## Output schema

```python
class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(..., min_length=1)
    sources: list[str]                              # chunk ids used; [] for general questions
    confidence: float = Field(..., ge=0.0, le=1.0)
```
FastAPI's `response_model=AskResponse` validates every response again on the way out. Requests are validated by
`AskRequest` (an empty query returns HTTP 422).

## Example calls (MOCK_LLM left at its default)

The block below is written by `python demo_calls.py` while `uvicorn main:app` is running. It records the exact raw JSON
the server returned for 4 policy questions (routed to `retrieve_and_answer`) and 2 unrelated questions (routed to
`direct_answer`). For each call it also checks that the top retrieved chunk comes from the expected source document.

<!-- DEMO:START -->
_Not recorded yet. Start the server and run `python demo_calls.py` to fill this section._
<!-- DEMO:END -->

## Optional extensions (ungraded)

- **Real LLM (`MOCK_LLM=0`).** Implemented against Groq's OpenAI-compatible endpoint (free tier, no card required).
  Set `MOCK_LLM=0` and `GROQ_API_KEY` in the environment (never commit the key); the model can be changed with `GROQ_MODEL`
  (default `llama-3.1-8b-instant`). The code path and its retry logic are covered by tests that stub the LLM. This path was
  not used to produce any graded output above.
- **Hugging Face Spaces.** Not deployed. The Dockerfile is Spaces-compatible (port 7860, uid 1000) if you want to try it
  on the free CPU tier; store `GROQ_API_KEY` as a Space secret.
