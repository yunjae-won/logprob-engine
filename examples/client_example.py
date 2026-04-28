"""Minimal client example.

Run the server first (e.g. ``logprob-engine serve --model gpt2``), then::

    python examples/client_example.py
"""

from __future__ import annotations

import math

from transformers import AutoTokenizer

from logprob_engine import LogprobClient


def main() -> None:
    base_url = "http://127.0.0.1:8000"
    client = LogprobClient(base_url)

    info = client.info()
    print("server info:", info)

    # The client is responsible for tokenization. Use the same tokenizer the
    # server loaded — its name is reported by /v1/info.
    tok = AutoTokenizer.from_pretrained(info["model"])

    prompt = "The capital of France is"
    output = " Paris."

    items = [
        {
            "prompt_ids": tok.encode(prompt, add_special_tokens=False),
            "output_ids": tok.encode(output, add_special_tokens=False),
        },
    ]

    logprobs = client.logprobs(items)
    for tok_id, lp in zip(items[0]["output_ids"], logprobs[0]):
        print(f"  token {tok_id!r:>6}  logp={lp:+.3f}  p={math.exp(lp):.3f}")

    total = sum(logprobs[0])
    print(f"sum logprob: {total:.3f}  (joint p = {math.exp(total):.3e})")

    arrays = client.logprob_arrays(items, format="npz")
    print("binary array shape:", arrays[0].shape)


if __name__ == "__main__":
    main()
