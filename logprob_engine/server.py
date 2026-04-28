"""FastAPI application exposing :class:`LogprobEngine` over HTTP.

The server serializes model forward calls with a process-local lock. This
keeps one reward GPU from receiving overlapping full-vocab requests from
multiple training ranks.
"""

from __future__ import annotations

import io
import threading

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
    engine_lock = threading.Lock()
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
        format: str = Query("json", pattern="^(json|npz|npz_compressed)$"),
    ):
        items = [it.model_dump() for it in request.items]
        try:
            with engine_lock:
                result = engine.process(items) if format == "json" else engine.process_arrays(items)
        except RuntimeError as e:
            # The engine raises RuntimeError when even a single item OOMs.
            raise HTTPException(status_code=413, detail=str(e)) from e

        if format == "json":
            return LogprobResponse(logprobs=result)

        # One named array per item preserves ragged sequence lengths without
        # forcing a Python-side concatenation. Uncompressed NPZ is the default
        # fast path; compression is CPU-heavy for dense full-vocab logprobs.
        buf = io.BytesIO()
        arrays = {f"item_{i}": np.asarray(lp) for i, lp in enumerate(result)}
        if format == "npz_compressed":
            np.savez_compressed(buf, **arrays)
        else:
            np.savez(buf, **arrays)
        return Response(content=buf.getvalue(), media_type="application/octet-stream")

    return app
