"""C3 worker launcher that prepares the Gemini process environment safely."""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import dotenv_values

from .config import llm_provider


def load_gemini_api_key(dotenv_path: str | Path | None = None) -> bool:
    """Expose the existing local Gemini key to the C3 adapter process only.

    An explicitly exported environment variable always wins.  When it is
    absent, only ``GEMINI_API_KEY`` is read from the local dotenv file; other
    dotenv values are not copied into the process environment.
    """
    if os.environ.get("GEMINI_API_KEY", "").strip():
        return True

    path = Path(dotenv_path) if dotenv_path is not None else Path.cwd() / ".env"
    values = dotenv_values(path)
    value = values.get("GEMINI_API_KEY")
    if not isinstance(value, str) or not value.strip():
        return False
    os.environ["GEMINI_API_KEY"] = value
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare the explicitly selected C3 provider, then run the worker."""
    if llm_provider() == "gemini" and not load_gemini_api_key():
        raise RuntimeError(
            "C3 worker requires GEMINI_API_KEY in the process environment or .env."
        )

    from .claim_worker_runner import main as worker_main

    return worker_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["load_gemini_api_key", "main"]
