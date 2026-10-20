"""Offline LaBSE tokenizer-only admission rule shared by API and RAG."""

import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

MODEL_NAME = "sentence-transformers/LaBSE"


class EmbeddingInputTooLongError(ValueError):
    """Raised when tokenized input exceeds the loaded model's limit."""

    def __init__(self, *, index: int, token_count: int, limit: int) -> None:
        self.index = index
        self.token_count = token_count
        self.limit = limit
        super().__init__(
            f"Input[{index}] has {token_count} tokens (limit {limit}); "
            "truncation is not allowed."
        )


def _validate_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise TypeError("texts must be a sequence of strings, not a single string.")
    values = list(texts)
    if not values:
        raise ValueError("texts must contain at least one non-empty text.")
    for index, text in enumerate(values):
        if not isinstance(text, str):
            raise TypeError(f"Input[{index}] must be a string.")
        if not text.strip():
            raise ValueError(f"Input[{index}] must not be empty or whitespace-only.")
    return values


class TokenizerUnavailableError(RuntimeError):
    """The local tokenizer or its effective sequence limit is unavailable."""


@lru_cache(maxsize=1)
def _local_tokenizer_and_limit():
    """Load tokenizer assets only; never fetch or instantiate model weights."""
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        snapshot = Path(
            snapshot_download(
                MODEL_NAME,
                local_files_only=True,
                allow_patterns=[
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.txt",
                    "sentence_bert_config.json",
                ],
            )
        )
        if not (snapshot / "tokenizer.json").is_file():
            raise FileNotFoundError("cached fast tokenizer is unavailable")
        config = json.loads((snapshot / "sentence_bert_config.json").read_text())
        limit = config.get("max_seq_length")
        if type(limit) is not int or limit < 3:
            raise ValueError("cached sequence limit is invalid")
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, use_fast=True,
        )
        if not tokenizer.is_fast:
            raise ValueError("cached fast tokenizer is unavailable")
        return tokenizer, limit
    except Exception as exc:
        raise TokenizerUnavailableError("Local claim tokenizer is unavailable.") from exc


def effective_claim_token_limit() -> int:
    return _local_tokenizer_and_limit()[1]


def validate_claim_input_tokens(texts: list[str]) -> list[int]:
    """Count full input tokens, including specials, without truncation."""
    values = _validate_texts(texts)
    tokenizer, limit = _local_tokenizer_and_limit()
    try:
        token_ids = tokenizer(
            values, truncation=False, add_special_tokens=True, verbose=False,
        )["input_ids"]
    except Exception as exc:
        raise TokenizerUnavailableError("Local claim tokenizer is unavailable.") from exc
    counts = [len(tokens) for tokens in token_ids]
    for index, count in enumerate(counts):
        if count > limit:
            raise EmbeddingInputTooLongError(
                index=index, token_count=count, limit=limit,
            )
    return counts
