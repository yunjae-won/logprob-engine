"""Command-line entry point: ``logprob-engine serve --model ...``."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from .engine import LogprobEngine
from .server import create_app


def _build_serve_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser("serve", help="Run the HTTP server.")
    p.add_argument("--model", required=True, help="HF model name or local path.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model weight dtype.",
    )
    p.add_argument(
        "--attn",
        default=None,
        help="HF attn_implementation, e.g. 'flash_attention_2' or 'sdpa'.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Force device (e.g. 'cuda', 'cuda:0', 'cpu'). Defaults to cuda if available.",
    )
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    p.add_argument("--log-level", default="info")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="logprob-engine")
    sub = parser.add_subparsers(dest="command", required=True)
    _build_serve_parser(sub)
    args = parser.parse_args(argv)

    if args.command == "serve":
        engine = LogprobEngine(
            args.model,
            dtype=args.dtype,
            attn_implementation=args.attn,
            device=args.device,
            compile=not args.no_compile,
        )
        app = create_app(engine)
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
