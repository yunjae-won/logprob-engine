"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LogprobItem(BaseModel):
    """One scoring request: log p(output_ids | prompt_ids)."""

    prompt_ids: list[int] = Field(..., min_length=1)
    output_ids: list[int] = Field(..., min_length=1)


class LogprobRequest(BaseModel):
    items: list[LogprobItem] = Field(..., min_length=1)


class LogprobResponse(BaseModel):
    """``logprobs[i]`` is parallel to ``request.items[i].output_ids``."""

    logprobs: list[Any]


class InfoResponse(BaseModel):
    model: str
    dtype: str
    device: str
    vocab_size: int
    logprob_level: str
    top_k: int | None = None
    top_k_include_outputs: bool = True


class TokenizeResponse(BaseModel):
    input_ids: list[int]
