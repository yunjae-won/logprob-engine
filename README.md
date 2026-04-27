# logprob-engine

A small, single-GPU HTTP server that returns **token-level log-probabilities**
under a Hugging Face causal language model. You send `(prompt_ids, output_ids)`
pairs; you get back, for each pair, the per-token log-probability of every
output token under the model.

It is intentionally simple:

- one model, one GPU, one client at a time;
- the HTTP layer is a thin shell around a `LogprobEngine` class;
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

For long outputs, request `?format=npz` to receive a compressed numpy archive
(`application/octet-stream`) with one `item_<i>` array per item.

### Other endpoints

- `GET /health` — liveness probe.
- `GET /v1/info` — `{model, dtype, device, vocab_size}`.
- `GET /v1/tokenize?text=...` — convenience helper that uses the server's
  tokenizer; clients normally tokenize on their own side.

Interactive docs are available at `/docs` (FastAPI's built-in Swagger UI).

## Python client

```python
from transformers import AutoTokenizer
from logprob_engine import LogprobClient

client = LogprobClient("http://127.0.0.1:8000")
info = client.info()

tok = AutoTokenizer.from_pretrained(info["model"])
items = [{
    "prompt_ids": tok.encode("The capital of France is", add_special_tokens=False),
    "output_ids": tok.encode(" Paris.",                  add_special_tokens=False),
}]

logprobs = client.logprobs(items)            # JSON
# logprobs = client.logprobs(items, format="npz")  # compressed numpy
```

## curl

```bash
curl -s http://127.0.0.1:8000/v1/logprobs \
     -H 'content-type: application/json' \
     -d '{"items":[{"prompt_ids":[464,3139,286,4881,318],"output_ids":[6342,13]}]}'
```

## Notes

- **Single-GPU, single-client.** No request queueing, no cross-request
  batching, no multi-worker uvicorn. If you need any of that, this isn't the
  right tool.
- **Tokenization.** Clients are responsible for tokenization. Use the same
  model name as the server (returned by `/v1/info`) so token IDs line up.
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
