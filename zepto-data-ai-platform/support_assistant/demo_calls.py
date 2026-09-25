"""Record example POST /ask calls against a running server and write them into README.md.

    # terminal 1  (MOCK_LLM left at its default)
    uvicorn main:app --host 0.0.0.0 --port 7860
    # terminal 2
    python demo_calls.py --url http://127.0.0.1:7860

Uses only the standard library so it can run from any Python.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
README = HERE / "README.md"
START, END = "<!-- DEMO:START -->", "<!-- DEMO:END -->"

EXAMPLES = [
    ("policy_question -> retrieve_and_answer", "What is the delivery fee if my order is below INR 149?", "doc_01"),
    ("policy_question -> retrieve_and_answer", "How many days do I have to return an unopened packaged item?", "doc_02"),
    ("policy_question -> retrieve_and_answer", "What do I get with Zepto Pass+ membership?", "doc_03"),
    ("policy_question -> retrieve_and_answer", "Can I cancel my order after it has been packed?", "doc_05"),
    ("general_question -> direct_answer", "What is the capital of France?", None),
    ("general_question -> direct_answer", "Can you recommend a good pasta recipe?", None),
]


def call(url: str, path: str, payload: dict | None = None) -> tuple[int, str]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url.rstrip("/") + path, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, resp.read().decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args()

    status, health = call(args.url, "/health")
    health_json = json.loads(health)
    if not health_json.get("mock_llm", False):
        print("WARNING: server is NOT in mock mode; graded transcripts must use the default MOCK_LLM.")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = [f"_Recorded by `python demo_calls.py` against `uvicorn main:app` on {ts}, "
           f"MOCK_LLM left unset (server reported `mock_llm: {str(health_json.get('mock_llm')).lower()}`)._\n",
           f"`GET /health` -> {status}\n", "```json", json.dumps(health_json, indent=2), "```\n"]
    ok = True
    for i, (route, query, expected_doc) in enumerate(EXAMPLES, 1):
        status, body = call(args.url, "/ask", {"query": query})
        parsed = json.loads(body)
        top_doc = parsed["sources"][0].split("#")[0] if parsed["sources"] else None
        verdict = ("top source matches expected document" if expected_doc and top_doc == expected_doc else
                   "no retrieval, empty sources" if not expected_doc and not parsed["sources"] else "UNEXPECTED")
        ok &= verdict != "UNEXPECTED"
        print(f"[{i}] {query}\n    -> {body}\n    {verdict}")
        curl = f"curl -s -X POST {args.url}/ask -H 'Content-Type: application/json' -d '{json.dumps({'query': query})}'"
        out += [f"**Call {i}: expected route `{route}`**" + (f" (expected document `{expected_doc}`)" if expected_doc else ""),
                "```bash", curl, "```", f"Raw response (HTTP {status}):", "```json", body, "```",
                f"Check: {verdict}.\n"]

    markdown = "\n".join(out)
    if not args.no_readme and README.exists():
        text = README.read_text(encoding="utf-8")
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        README.write_text(f"{head}{START}\n{markdown}\n{END}{tail}", encoding="utf-8")
        print(f"\nREADME.md updated with {len(EXAMPLES)} recorded calls.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
