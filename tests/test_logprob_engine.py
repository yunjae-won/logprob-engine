from __future__ import annotations

import io
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from fastapi.testclient import TestClient

from logprob_engine import LogprobEngine, create_app, unpack_topk_array


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 6
    pad_token = "<pad>"

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [int(x) for x in text.split()]


class TinyBody(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 7, hidden_size: int = 5) -> None:
        super().__init__()
        self.model = TinyBody(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(vocab_size=vocab_size)


def make_model() -> TinyCausalLM:
    torch.manual_seed(1234)
    model = TinyCausalLM()
    with torch.no_grad():
        model.model.embed.weight.copy_(
            torch.arange(model.config.vocab_size * 5, dtype=torch.float32).view(model.config.vocab_size, 5) / 17.0
        )
        model.lm_head.weight.copy_(
            torch.arange(model.config.vocab_size * 5, dtype=torch.float32).view(model.config.vocab_size, 5) / 23.0
        )
    return model


def expected_vocab(model: TinyCausalLM, prompt_ids: list[int], output_ids: list[int]) -> torch.Tensor:
    input_ids, labels = LogprobEngine._construct_input(prompt_ids, output_ids)
    shifted_labels = labels[1:]
    mask = shifted_labels != -100
    hidden = model.model.forward(input_ids.unsqueeze(0)).last_hidden_state[:, :-1][mask.unsqueeze(0)]
    return torch.log_softmax(model.lm_head(hidden), dim=-1)


def expected_label_inclusive_topk(
    vocab_logprobs: torch.Tensor,
    labels: list[int],
    top_k: int,
) -> torch.Tensor:
    values, indices = torch.topk(vocab_logprobs, k=top_k, dim=-1)
    labels_t = torch.as_tensor(labels, dtype=torch.long)[:, None]
    has_label = (indices == labels_t).any(dim=-1, keepdim=True)
    label_values = vocab_logprobs.gather(1, labels_t)
    values = values.clone()
    indices = indices.clone()
    values[:, -1:] = torch.where(has_label, values[:, -1:], label_values)
    indices[:, -1:] = torch.where(has_label, indices[:, -1:], labels_t)
    values, order = values.sort(dim=-1, descending=True)
    indices = indices.gather(1, order)
    return torch.stack([indices.float(), values], dim=-1)


class LogprobEngineFastPathTest(unittest.TestCase):
    def test_vocab_tensor_array_and_list_paths_match_expected_values(self) -> None:
        model = make_model()
        engine = LogprobEngine.from_components(
            model,
            TinyTokenizer(),
            device="cpu",
            compile=False,
            logprob_level="vocab",
            logprob_dtype="float32",
        )
        items = [
            {"prompt_ids": [1, 2, 3], "output_ids": [4, 5]},
            {"prompt_ids": [2], "output_ids": [3, 4, 5]},
        ]

        tensors = engine.process_tensors(items)
        arrays = engine.process_arrays(items)
        lists = engine.process(items)

        for item, tensor, array, list_value in zip(items, tensors, arrays, lists, strict=True):
            expected = expected_vocab(model, item["prompt_ids"], item["output_ids"])
            expected_np = expected.detach().numpy()
            self.assertEqual(tuple(tensor.shape), tuple(expected.shape))
            self.assertTrue(torch.allclose(tensor.cpu(), expected, atol=1e-6))
            self.assertIsInstance(array, np.ndarray)
            self.assertTrue(np.allclose(array, expected_np, atol=1e-6))
            self.assertTrue(np.allclose(np.asarray(list_value), expected_np, atol=1e-6))

    def test_token_and_seq_paths_match_vocab_gather_and_sum(self) -> None:
        items = [
            {"prompt_ids": [1, 2, 3], "output_ids": [4, 5]},
            {"prompt_ids": [2], "output_ids": [3, 4, 5]},
        ]
        token_engine = LogprobEngine.from_components(
            make_model(),
            TinyTokenizer(),
            device="cpu",
            compile=False,
            logprob_level="token",
            logprob_dtype="float32",
        )
        seq_engine = LogprobEngine.from_components(
            make_model(),
            TinyTokenizer(),
            device="cpu",
            compile=False,
            logprob_level="seq",
            logprob_dtype="float32",
        )

        token_tensors = token_engine.process_tensors(items)
        seq_tensors = seq_engine.process_tensors(items)
        expected_model = make_model()
        for item, token_tensor, seq_tensor in zip(items, token_tensors, seq_tensors, strict=True):
            vocab = expected_vocab(expected_model, item["prompt_ids"], item["output_ids"])
            labels = torch.as_tensor(item["output_ids"], dtype=torch.long)
            expected_tokens = vocab.gather(1, labels[:, None]).squeeze(1)
            self.assertTrue(torch.allclose(token_tensor.cpu(), expected_tokens, atol=1e-6))
            self.assertTrue(torch.allclose(seq_tensor.cpu(), expected_tokens.sum(), atol=1e-6))

    def test_top_k_one_aliases_token_logprobs(self) -> None:
        items = [{"prompt_ids": [1, 2, 3], "output_ids": [4, 5]}]
        top1_engine = LogprobEngine.from_components(
            make_model(),
            TinyTokenizer(),
            device="cpu",
            compile=False,
            logprob_level="vocab",
            logprob_dtype="float32",
            top_k=1,
        )
        token_engine = LogprobEngine.from_components(
            make_model(),
            TinyTokenizer(),
            device="cpu",
            compile=False,
            logprob_level="token",
            logprob_dtype="float32",
        )

        self.assertEqual(top1_engine.logprob_level, "token")
        self.assertTrue(torch.allclose(top1_engine.process_tensors(items)[0], token_engine.process_tensors(items)[0]))

    def test_label_inclusive_top_k_vocab_logprobs(self) -> None:
        model = make_model()
        engine = LogprobEngine.from_components(
            model,
            TinyTokenizer(),
            device="cpu",
            compile=False,
            logprob_level="vocab",
            logprob_dtype="float32",
            top_k=3,
        )
        item = {"prompt_ids": [1, 2, 3], "output_ids": [4, 5]}

        actual = engine.process_tensors([item])[0]
        expected = expected_label_inclusive_topk(
            expected_vocab(model, item["prompt_ids"], item["output_ids"]),
            item["output_ids"],
            top_k=3,
        )

        self.assertEqual(tuple(actual.shape), (2, 3, 2))
        self.assertTrue(torch.allclose(actual.cpu(), expected, atol=1e-6))
        token_ids = actual[..., 0].long()
        for row, label in zip(token_ids, item["output_ids"], strict=True):
            self.assertIn(label, row.tolist())

    def test_true_model_top_k_without_forced_output_tokens(self) -> None:
        model = make_model()
        engine = LogprobEngine.from_components(
            model,
            TinyTokenizer(),
            device="cpu",
            compile=False,
            logprob_level="topk",
            logprob_dtype="float32",
            top_k=2,
            top_k_include_outputs=False,
        )
        item = {"prompt_ids": [1, 2, 3], "output_ids": [4, 5]}
        vocab = expected_vocab(model, item["prompt_ids"], item["output_ids"])
        values, indices = torch.topk(vocab, k=2, dim=-1)
        expected = torch.stack([indices.float(), values], dim=-1)

        actual = engine.process_tensors([item])[0]
        self.assertTrue(torch.allclose(actual.cpu(), expected, atol=1e-6))

    def test_http_npz_path_returns_arrays_without_json_roundtrip(self) -> None:
        engine = LogprobEngine.from_components(
            make_model(),
            TinyTokenizer(),
            device="cpu",
            compile=False,
            logprob_level="vocab",
            logprob_dtype="float32",
        )
        items = [{"prompt_ids": [1, 2], "output_ids": [3, 4]}]
        client = TestClient(create_app(engine))

        resp = client.post("/v1/logprobs", params={"format": "npz"}, json={"items": items})
        self.assertEqual(resp.status_code, 200)
        with np.load(io.BytesIO(resp.content)) as npz:
            actual = npz["item_0"]

        expected = engine.process_arrays(items)[0]
        self.assertTrue(np.allclose(actual, expected, atol=1e-6))

    def test_http_npz_top_k_path_returns_packed_ids_and_logprobs(self) -> None:
        engine = LogprobEngine.from_components(
            make_model(),
            TinyTokenizer(),
            device="cpu",
            compile=False,
            logprob_level="vocab",
            logprob_dtype="float32",
            top_k=3,
        )
        items = [{"prompt_ids": [1, 2], "output_ids": [3, 4]}]
        client = TestClient(create_app(engine))

        resp = client.post("/v1/logprobs", params={"format": "npz"}, json={"items": items})
        self.assertEqual(resp.status_code, 200)
        with np.load(io.BytesIO(resp.content)) as npz:
            packed = npz["item_0"]

        token_ids, logprobs = unpack_topk_array(packed)
        self.assertEqual(packed.shape, (2, 3, 2))
        self.assertEqual(token_ids.shape, (2, 3))
        self.assertEqual(logprobs.shape, (2, 3))

    def test_http_json_top_k_path_accepts_nested_payload(self) -> None:
        engine = LogprobEngine.from_components(
            make_model(),
            TinyTokenizer(),
            device="cpu",
            compile=False,
            logprob_level="vocab",
            logprob_dtype="float32",
            top_k=2,
        )
        items = [{"prompt_ids": [1, 2], "output_ids": [3, 4]}]
        client = TestClient(create_app(engine))

        resp = client.post("/v1/logprobs", json={"items": items})
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()["logprobs"]
        self.assertEqual(len(payload), 1)
        self.assertEqual(np.asarray(payload[0]).shape, (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
