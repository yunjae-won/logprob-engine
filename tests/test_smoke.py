"""Smoke tests that run on CPU without any HF Hub downloads.

We construct a tiny causal LM from scratch and a stub tokenizer, then exercise
the engine and the HTTP layer end-to-end.
"""

from __future__ import annotations

import io
import math
from types import SimpleNamespace

import numpy as np
import torch
from fastapi.testclient import TestClient

from logprob_engine import LogprobEngine, create_app


VOCAB = 64
PAD_ID = 0
EOS_ID = 1


class StubTokenizer:
    """Minimum surface area used by :class:`LogprobEngine`."""

    pad_token = "<pad>"
    pad_token_id = PAD_ID
    eos_token_id = EOS_ID

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [(ord(c) % (VOCAB - 2)) + 2 for c in text]
        return ids


class TinyBody(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = VOCAB, hidden_size: int = 16) -> None:
        super().__init__()
        self.model = TinyBody(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(vocab_size=vocab_size)


def _build_engine(*, compile: bool = False) -> LogprobEngine:
    torch.manual_seed(0)
    model = TinyCausalLM()
    return LogprobEngine.from_components(
        model,
        StubTokenizer(),
        model_name="tiny-llama-test",
        device="cpu",
        compile=compile,
    )


def test_engine_returns_per_token_logprobs():
    engine = _build_engine()
    items = [
        {"prompt_ids": [2, 3, 4], "output_ids": [5, 6, 7]},
        {"prompt_ids": [10, 11], "output_ids": [12]},
    ]
    out = engine.process(items)

    assert len(out) == 2
    assert len(out[0]) == 3
    assert len(out[1]) == 1
    for row in out:
        for lp in row:
            assert math.isfinite(lp)
            assert lp <= 0  # log-probabilities must be non-positive


def test_engine_preserves_input_order_after_internal_sort():
    """Items are sorted by length desc internally; results must come back in
    the original order."""
    engine = _build_engine()
    items = [
        {"prompt_ids": [2], "output_ids": [3]},                       # short
        {"prompt_ids": [2, 3, 4, 5, 6, 7], "output_ids": [8, 9, 10]}, # long
        {"prompt_ids": [2, 3], "output_ids": [4, 5]},                 # medium
    ]
    out = engine.process(items)
    assert [len(x) for x in out] == [1, 3, 2]


def test_oom_halving_recovers_and_keeps_order(monkeypatch):
    engine = _build_engine()

    real_process = engine._process_batch
    state = {"raised": False}

    def flaky(indexed_batch):
        if not state["raised"] and len(indexed_batch) > 1:
            state["raised"] = True
            raise torch.cuda.OutOfMemoryError("simulated")
        return real_process(indexed_batch)

    monkeypatch.setattr(engine, "_process_batch", flaky)

    items = [
        {"prompt_ids": [2, 3], "output_ids": [4, 5]},
        {"prompt_ids": [6, 7, 8, 9], "output_ids": [10, 11]},
        {"prompt_ids": [12], "output_ids": [13, 14, 15]},
    ]
    out = engine.process(items)

    assert state["raised"] is True
    assert [len(x) for x in out] == [2, 2, 3]


def _http_client(engine: LogprobEngine) -> TestClient:
    return TestClient(create_app(engine))


def test_http_logprobs_json():
    c = _http_client(_build_engine())
    assert c.get("/health").json() == {"status": "ok"}

    info = c.get("/v1/info").json()
    assert info["model"] == "tiny-llama-test"
    assert info["device"] == "cpu"
    assert info["vocab_size"] == VOCAB

    body = {"items": [{"prompt_ids": [2, 3, 4], "output_ids": [5, 6]}]}
    resp = c.post("/v1/logprobs", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["logprobs"]) == 1
    assert len(data["logprobs"][0]) == 2


def test_http_logprobs_npz():
    c = _http_client(_build_engine())
    body = {
        "items": [
            {"prompt_ids": [2, 3], "output_ids": [4, 5, 6]},
            {"prompt_ids": [7, 8, 9], "output_ids": [10]},
        ]
    }
    resp = c.post("/v1/logprobs", params={"format": "npz"}, json=body)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"

    with np.load(io.BytesIO(resp.content)) as npz:
        assert npz["item_0"].shape == (3,)
        assert npz["item_1"].shape == (1,)
        assert (npz["item_0"] <= 0).all()


def test_http_tokenize_endpoint():
    c = _http_client(_build_engine())
    resp = c.get("/v1/tokenize", params={"text": "abc"})
    assert resp.status_code == 200
    ids = resp.json()["input_ids"]
    assert ids == StubTokenizer().encode("abc")


def test_http_rejects_empty_request():
    c = _http_client(_build_engine())
    resp = c.post("/v1/logprobs", json={"items": []})
    assert resp.status_code == 422  # Pydantic validation error
