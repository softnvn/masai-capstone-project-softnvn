"""Pydantic request/response models - the enforced JSON output schema."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, examples=["What is the delivery fee below INR 149?"])


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., min_length=1)
    sources: list[str] = Field(default_factory=list, description="chunk ids used; empty for general questions")
    confidence: float = Field(..., ge=0.0, le=1.0)
