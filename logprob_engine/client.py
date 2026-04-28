"""Thin synchronous HTTP client for a logprob-engine server."""

from __future__ import annotations

import io
from typing import Iterable, Mapping

import numpy as np
import requests


def unpack_topk_array(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a packed ``[tokens, k, 2]`` top-k array into ids and logprobs."""

    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"Expected a packed top-k array with shape [tokens, k, 2], got {array.shape}.")
    return array[..., 0].astype(np.int64, copy=False), array[..., 1]


class LogprobClient:
    def __init__(self, base_url: str, *, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    # --------------------------- core endpoint --------------------------- #

    def logprobs(
        self,
        items: Iterable[Mapping[str, list[int]]],
        *,
        format: str = "json",
    ) -> list[list[float]]:
        """Score each ``{prompt_ids, output_ids}`` item.

        ``format="npz"`` requests a binary response. Use
        :meth:`logprob_arrays` when the caller can consume NumPy arrays
        directly; this compatibility wrapper converts back to Python lists.
        """
        if format == "json":
            items = [dict(it) for it in items]
            resp = self.session.post(
                f"{self.base_url}/v1/logprobs",
                json={"items": items},
                params={"format": format},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["logprobs"]

        return [arr.astype(np.float32, copy=False).tolist() for arr in self.logprob_arrays(items, format=format)]

    def logprob_arrays(
        self,
        items: Iterable[Mapping[str, list[int]]],
        *,
        format: str = "npz",
    ) -> list[np.ndarray]:
        """Score items and return NumPy arrays without Python-list inflation."""
        if format not in ("npz", "npz_compressed"):
            raise ValueError(f"format must be 'npz' or 'npz_compressed', got {format!r}")
        items = [dict(it) for it in items]
        resp = self.session.post(
            f"{self.base_url}/v1/logprobs",
            json={"items": items},
            params={"format": format},
            timeout=self.timeout,
        )
        resp.raise_for_status()

        with np.load(io.BytesIO(resp.content)) as npz:
            return [npz[f"item_{i}"] for i in range(len(items))]

    def logprob_topk_arrays(
        self,
        items: Iterable[Mapping[str, list[int]]],
        *,
        format: str = "npz",
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Score items from a top-k server and return ``(token_ids, logprobs)`` arrays."""

        return [unpack_topk_array(array) for array in self.logprob_arrays(items, format=format)]

    # --------------------------- helpers --------------------------- #

    def tokenize(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        resp = self.session.get(
            f"{self.base_url}/v1/tokenize",
            params={"text": text, "add_special_tokens": str(add_special_tokens).lower()},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["input_ids"]

    def info(self) -> dict:
        resp = self.session.get(f"{self.base_url}/v1/info", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("status") == "ok"
        except requests.RequestException:
            return False

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "LogprobClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
