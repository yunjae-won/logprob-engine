"""FastAPI application exposing :class:`LogprobEngine` over HTTP.

This server assumes a *single client* — handlers are plain synchronous
functions and there is no locking or request queueing.
"""

from __future__ import annotations

import io

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response

from .engine import LogprobEngine
from .schemas import (
    InfoResponse,
    LogprobRequest,
    LogprobResponse,
    TokenizeResponse,
)


def create_app(engine: LogprobEngine) -> FastAPI:
    app = FastAPI(
        title="logprob-engine",
        version="0.1.0",
        description="Token-level log-probability server for a Hugging Face causal LM.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/info", response_model=InfoResponse)
    def info() -> InfoResponse:
        return InfoResponse(
            model=engine.model_name_or_path,
            dtype=str(engine.torch_dtype).replace("torch.", ""),
            device=str(engine.device),
            vocab_size=engine.vocab_size,
        )

    @app.get("/v1/tokenize", response_model=TokenizeResponse)
    def tokenize(
        text: str = Query(..., description="Text to tokenize."),
        add_special_tokens: bool = Query(False),
    ) -> TokenizeResponse:
        return TokenizeResponse(
            input_ids=engine.tokenize(text, add_special_tokens=add_special_tokens)
        )

    @app.post("/v1/logprobs")
    def logprobs(
        request: LogprobRequest,
        format: str = Query("json", pattern="^(json|npz)$"),
    ):
        items = [it.model_dump() for it in request.items]
        try:
            result = engine.process(items)
        except RuntimeError as e:
            # The engine raises RuntimeError when even a single item OOMs.
            raise HTTPException(status_code=413, detail=str(e)) from e

        if format == "json":
            return LogprobResponse(logprobs=result)

        # ``savez_compressed`` with one named array per item preserves the
        # ragged shape without forcing a Python-side concatenation.
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            **{f"item_{i}": np.asarray(lp, dtype=np.float32) for i, lp in enumerate(result)},
        )
        return Response(content=buf.getvalue(), media_type="application/octet-stream")

    return app
