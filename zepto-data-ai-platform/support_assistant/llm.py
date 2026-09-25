"""The single MOCK_LLM toggle and the optional real-LLM client.

MOCK_LLM unset or "1"  -> mock mode (graded baseline): no LLM client is ever
                          created and no request is ever sent to an LLM provider.
MOCK_LLM="0"           -> call Groq's OpenAI-compatible chat API (free tier),
                          key read from the GROQ_API_KEY environment variable.
"""
from __future__ import annotations

import os

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def is_mock() -> bool:
    """Read on every call so the toggle can be flipped without code changes."""
    return os.getenv("MOCK_LLM", "1").strip() != "0"


class LLMError(RuntimeError):
    pass


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 300) -> str:
    """Send a chat completion to the real LLM. Never called in mock mode."""
    if is_mock():
        raise LLMError("chat() called while MOCK_LLM is enabled - this is a bug")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise LLMError("MOCK_LLM=0 but GROQ_API_KEY is not set")
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": GROQ_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
        timeout=30,
    )
    if resp.status_code != 200:
        raise LLMError(f"LLM HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()["choices"][0]["message"]["content"]
