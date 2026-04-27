"""Single-GPU log-probability engine.

Wraps a Hugging Face causal LM in a small object that, given a list of
``(prompt_ids, output_ids)`` pairs, returns the per-token log-probability
of each output token under the model.

The compute path matches the structure of the original skeleton:

* a custom ``forward`` that runs the LM body once and applies the LM head
  only to the positions where a label is present,
* ``torch.compile`` over that forward (dynamic shapes),
* batch-level OOM recovery by recursive halving.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Sequence

import torch
import torch.nn.functional as F
import transformers
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass
class Item:
    prompt_ids: Sequence[int]
    output_ids: Sequence[int]


def _disable_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower().replace("torch.", "")
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype!r}. Choose one of {list(_DTYPE_MAP)}.")
    return _DTYPE_MAP[key]


class LogprobEngine:
    """Loads a causal LM and computes per-token log-probabilities of supplied outputs."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        dtype: str | torch.dtype = "bfloat16",
        attn_implementation: str | None = None,
        device: str | torch.device | None = None,
        compile: bool = True,
        low_cpu_mem_usage: bool = True,
    ) -> None:
        torch_dtype = _resolve_dtype(dtype)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

        self._configure(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            compile=compile,
        )

    @classmethod
    def from_components(
        cls,
        model: torch.nn.Module,
        tokenizer,
        *,
        model_name: str = "<custom>",
        device: str | torch.device | None = None,
        compile: bool = True,
    ) -> "LogprobEngine":
        """Build an engine from an already-loaded model and tokenizer.

        Useful for tests, custom architectures, or when you want full control
        over how the model is loaded (e.g. quantization, sharding).
        The model must expose ``model.model`` (the LM body returning a
        ``last_hidden_state``) and ``model.lm_head``, matching the structure
        used by LLaMA / Mistral / Qwen-family causal LMs in ``transformers``.
        """
        torch_dtype = next(model.parameters()).dtype
        self = cls.__new__(cls)
        self._configure(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            torch_dtype=torch_dtype,
            device=device,
            compile=compile,
        )
        return self

    def _configure(
        self,
        *,
        model: torch.nn.Module,
        tokenizer,
        model_name: str,
        torch_dtype: torch.dtype,
        device: str | torch.device | None,
        compile: bool,
    ) -> None:
        self.model_name_or_path = model_name
        self.torch_dtype = torch_dtype

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        torch.set_float32_matmul_precision("high")
        torch._dynamo.config.capture_scalar_outputs = True

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        _disable_dropout(model)
        model.to(self.device)
        model.eval()

        def custom_forward(model, input_ids, attention_mask, flat_labels, shifted_loss_mask):
            hidden_states = model.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).last_hidden_state[:, :-1][shifted_loss_mask]
            logits = -F.cross_entropy(
                model.lm_head(hidden_states).float(), flat_labels, reduction="none"
            )
            return logits.split(shifted_loss_mask.sum(-1).tolist(), dim=0)

        if compile:
            custom_forward = torch.compile(custom_forward, dynamic=True)
        model.forward = MethodType(custom_forward, model)
        self.model = model

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    def tokenize(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    # --------------------------- internals --------------------------- #

    @staticmethod
    def _construct_input(prompt_ids: Sequence[int], output_ids: Sequence[int]):
        prompt = torch.tensor(prompt_ids, dtype=torch.long)
        output = torch.tensor(output_ids, dtype=torch.long)
        input_ids = torch.cat([prompt, output], dim=0)
        labels = torch.cat([torch.full_like(prompt, -100), output], dim=0)
        return input_ids, labels

    def _collate(self, batch):
        pad_id = self.tokenizer.pad_token_id
        input_ids = [x["input_ids"] for x in batch]
        labels = [x["labels"] for x in batch]
        attention_mask = [torch.ones_like(x, dtype=torch.long) for x in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        shifted_labels = labels[:, 1:]
        shifted_loss_mask = (shifted_labels != -100).clone()
        return input_ids, attention_mask, shifted_labels[shifted_loss_mask], shifted_loss_mask

    def _process_batch(self, indexed_batch):
        indices = [i for i, _ in indexed_batch]
        records = [r for _, r in indexed_batch]
        input_ids, attention_mask, labels, loss_mask = self._collate(records)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        loss_mask = loss_mask.to(self.device)
        with torch.no_grad():
            logits = self.model(
                input_ids,
                attention_mask=attention_mask,
                flat_labels=labels,
                shifted_loss_mask=loss_mask,
            )
        return list(logits), indices

    def _retry_by_halving(self, batch):
        try:
            return self._process_batch(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise RuntimeError(
                    "OOM with batch size 1; the single item is too large for this GPU."
                )
            mid = len(batch) // 2
            left, left_idx = self._retry_by_halving(batch[:mid])
            right, right_idx = self._retry_by_halving(batch[mid:])
            return left + right, left_idx + right_idx

    # --------------------------- public API --------------------------- #

    def process(self, items: Sequence[Item | dict]) -> list[list[float]]:
        """Return per-token logprobs for each item's ``output_ids``.

        Output[i] has length ``len(items[i].output_ids)`` and each value is
        ``log p(output_ids[t] | prompt_ids, output_ids[:t])``.
        """
        records = []
        for it in items:
            prompt_ids = it["prompt_ids"] if isinstance(it, dict) else it.prompt_ids
            output_ids = it["output_ids"] if isinstance(it, dict) else it.output_ids
            input_ids, labels = self._construct_input(prompt_ids, output_ids)
            records.append({"input_ids": input_ids, "labels": labels})

        # Sort by length desc so the largest sample (most likely to OOM)
        # is tried first; remember original positions to unsort the result.
        indexed = list(enumerate(records))
        indexed.sort(key=lambda x: len(x[1]["input_ids"]), reverse=True)

        logits, indices = self._retry_by_halving(indexed)

        ordered = [None] * len(items)
        for idx, tensor in zip(indices, logits):
            ordered[idx] = tensor.detach().to("cpu", dtype=torch.float32).tolist()
        return ordered  # type: ignore[return-value]
