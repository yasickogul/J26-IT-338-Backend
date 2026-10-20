"""Opaque browser session tokens; only their SHA-256 digests enter storage."""

import hashlib
import re
import secrets


COOKIE_NAME = "compnet_claim_session"
SECURE_COOKIE_NAME = "__Host-compnet_claim_session"
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}\Z")


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def valid_session_token(value: str | None) -> bool:
    return isinstance(value, str) and _TOKEN_PATTERN.fullmatch(value) is not None


def session_token_hash(token: str) -> str:
    if not valid_session_token(token):
        raise ValueError("invalid anonymous session token")
    return hashlib.sha256(b"compnet-claim-session-v1:\x00" + token.encode("ascii")).hexdigest()
