# logprob-engine

A small, single-GPU HTTP server that returns **token**, **sequence**,
**full-vocab**, or compact **top-k vocab** log-probabilities under a Hugging
Face causal language model. You send `(prompt_ids, output_ids)` pairs; you get
back either the sampled-token log-probs, the summed sequence log-prob, a
`[output_len, vocab]` tensor, or a compact `[output_len, k, 2]` tensor of
`(token_id, logprob)` pairs for each pair.

It is intentionally simple:

- one model, one GPU, serialized model forwards;
- the HTTP layer is a thin shell around a `LogprobEngine` class;
- `torch.compile` friendly dense forwards, with ragged splitting outside the
  compiled graph;
- batches that don't fit are recursively halved until they do.

## Install

```bash
pip install logprob-engine
# or, from a clone
pip install -e .
```

`torch` and `transformers` are runtime dependencies. If you want
`flash_attention_2` install it separately (`pip install flash-attn`) and pass
`--attn flash_attention_2` to the server.

## Run the server

```bash
logprob-engine serve --model gpt2 --host 127.0.0.1 --port 8000
```

Useful flags:

| flag | meaning |
| --- | --- |
| `--model`         | HF model name or local path (required) |
| `--dtype`         | `bfloat16` (default), `float16`, `float32` |
| `--attn`          | e.g. `flash_attention_2`, `sdpa` |
| `--device`        | `cuda`, `cuda:0`, `cpu`; auto-detected by default |
| `--logprob-level` | `token` (default), `seq`, `vocab`, or `topk` |
| `--logprob-dtype` | `float32` (default), `bfloat16`, or `float16` |
| `--top-k` | compact top-k vocab size; use with `vocab`/`topk` |
| `--no-top-k-include-output` | do not force sampled output tokens into top-k results |
| `--no-compile`    | disable `torch.compile` (useful while debugging) |

## API

### `POST /v1/logprobs`

Body:

```json
{
  "items": [
    {"prompt_ids": [464, 3139, 286, 4881, 318], "output_ids": [6342, 13]}
  ]
}
```

Response (default `?format=json`):

```json
{"logprobs": [[-1.42, -0.07]]}
```

`logprobs[i]` is the same length as `items[i].output_ids`, and each entry is
`log p(output_ids[t] | prompt_ids, output_ids[:t])`.

For long outputs or full-vocab mode, request `?format=npz` to receive an
uncompressed NumPy archive (`application/octet-stream`) with one `item_<i>`
array per item. `?format=npz_compressed` is also available, but compression is
usually CPU-bound and slow for dense full-vocab log-probs.

### Top-k vocab mode

Use `--logprob-level vocab --top-k K` (or `--logprob-level topk --top-k K`) to
return compact top-k vocab log-probs instead of the full `[tokens, vocab]`
matrix. By default the engine is **label-inclusive**: each sampled
`output_ids[t]` token is guaranteed to appear in that position's top-k result.
If it is not naturally in the model's top-k, it replaces the last top-k entry.
This makes `--top-k 1` exactly reduce to the sampled-token `token` logprob
setup.

For `K > 1`, each `item_<i>` NPZ array has shape `[output_len, K, 2]`:

- `array[..., 0]`: token ids, stored as float32 for the packed array format
- `array[..., 1]`: log-probabilities

Use `LogprobClient.logprob_topk_arrays(...)` or `unpack_topk_array(...)` to
split the packed array into integer token IDs and logprob arrays. If you want
the model's true top-k tokens without forcing the sampled token into the
result, pass `--no-top-k-include-output`.

### Other endpoints

- `GET /health` — liveness probe.
- `GET /v1/info` — `{model, dtype, device, vocab_size, logprob_level, top_k, top_k_include_outputs}`.
- `GET /v1/tokenize?text=...` — convenience helper that uses the server's
  tokenizer; clients normally tokenize on their own side.

Interactive docs are available at `/docs` (FastAPI's built-in Swagger UI).

## Python client

```python
from transformers import AutoTokenizer
from logprob_engine import LogprobClient, unpack_topk_array

client = LogprobClient("http://127.0.0.1:8000")
info = client.info()

tok = AutoTokenizer.from_pretrained(info["model"])
items = [{
    "prompt_ids": tok.encode("The capital of France is", add_special_tokens=False),
    "output_ids": tok.encode(" Paris.",                  add_special_tokens=False),
}]

logprobs = client.logprobs(items)  # JSON/list compatibility path

# Fast binary path; returns NumPy arrays directly.
arrays = client.logprob_arrays(items, format="npz")

# If the server was started with --top-k K, split packed arrays like this:
token_ids, topk_logprobs = unpack_topk_array(arrays[0])
# or:
token_ids, topk_logprobs = client.logprob_topk_arrays(items, format="npz")[0]
```

## curl

```bash
curl -s http://127.0.0.1:8000/v1/logprobs \
     -H 'content-type: application/json' \
     -d '{"items":[{"prompt_ids":[464,3139,286,4881,318],"output_ids":[6342,13]}]}'
```

## Notes

- **Single-GPU, serialized forwards.** The server uses a process-local lock
  around model execution so concurrent clients do not overlap full-vocab
  requests on one GPU. There is no cross-request batching or multi-worker
  uvicorn.
- **Tokenization.** Clients are responsible for tokenization. Use the same
  model name as the server (returned by `/v1/info`) so token IDs line up.
- **Vocab-level speed path.** Prefer `format=npz` plus
  `LogprobClient.logprob_arrays(...)`; this avoids Python-list inflation for
  large `[tokens, vocab]` payloads.
- **No scoring-result cache.** The engine does not cache request outputs. In
  online training workloads, repeated exact `(prompt_ids, output_ids)` requests
  are rare, so callers should rely on batching/top-k sizing rather than result
  caching.
- **OOM behaviour.** When a batch OOMs, the engine catches it, empties the
  CUDA cache, and retries the two halves recursively. If a single item OOMs
  the server returns `413 Payload Too Large`.

## Development

```bash
pip install -e '.[test]'
pytest -q
```

The smoke test runs against `sshleifer/tiny-gpt2` on CPU, so a GPU is not
required to validate the request/response path.

## License

MIT.
