"""Thin synchronous HTTP client for a logprob-engine server."""

from __future__ import annotations

import io
from typing import Iterable, Mapping

import numpy as np
import requests


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

        ``format="npz"`` requests a compressed binary response and decodes it
        back into a Python list — useful when output sequences are long.
        """
        if format not in ("json", "npz"):
            raise ValueError(f"format must be 'json' or 'npz', got {format!r}")
        items = [dict(it) for it in items]
        resp = self.session.post(
            f"{self.base_url}/v1/logprobs",
            json={"items": items},
            params={"format": format},
            timeout=self.timeout,
        )
        resp.raise_for_status()

        if format == "json":
            return resp.json()["logprobs"]

        with np.load(io.BytesIO(resp.content)) as npz:
            return [npz[f"item_{i}"].tolist() for i in range(len(items))]

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
