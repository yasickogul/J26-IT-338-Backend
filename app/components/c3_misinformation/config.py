"""Small, side-effect-free readers for C3 runtime feature flags."""

from __future__ import annotations

import os


def boolean_env(name: str, *, default: bool = False) -> bool:
    """Read a strict boolean environment value without changing its default."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("environment variable name must be nonblank")
    if not isinstance(default, bool):
        raise TypeError("default must be a boolean")

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    value = raw_value.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise RuntimeError(f"{name} must be a boolean value.")


def language_detection_enabled() -> bool:
    """Return the established opt-in statistical language-detection flag."""
    return boolean_env("COMPNET_LANGUAGE_DETECTION_ENABLED")


def claim_extraction_enabled() -> bool:
    """Return the established opt-in worker claim-extraction flag."""
    return boolean_env("COMPNET_CLAIM_EXTRACTION_ENABLED")


__all__ = [
    "boolean_env",
    "claim_extraction_enabled",
    "language_detection_enabled",
]
