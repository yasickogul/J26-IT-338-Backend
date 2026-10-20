"""Independent CompNet RAG result and categorized error contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


class RagServiceError(RuntimeError):
    """Base class for failures with an explicit orchestration phase."""

    def __init__(self, message: str, *, category: str, attempts: int = 0) -> None:
        super().__init__(message)
        self.category = category
        self.attempts = attempts


class RagInputError(RagServiceError, ValueError):
    pass


class RagInputTooLongError(RagInputError):
    def __init__(self, *, token_count: int, limit: int) -> None:
        self.token_count = token_count
        self.limit = limit
        super().__init__(
            "The question or claim exceeds the LaBSE token limit "
            f"({token_count} tokens; limit {limit}). Please shorten the question or claim.",
            category="input_too_long",
        )


class RagRetrievalError(RagServiceError):
    pass


class RagPreparationError(RagServiceError):
    pass


class RagRequestTooLargeError(RagServiceError):
    def __init__(
        self,
        *,
        estimated_input_tokens: int,
        project_budget: int,
        model_input_limit: int,
        output_allowance: int,
    ) -> None:
        self.estimated_input_tokens = estimated_input_tokens
        self.project_budget = project_budget
        self.model_input_limit = model_input_limit
        self.output_allowance = output_allowance
        super().__init__(
            "The Gemini request is too large for the configured request budget "
            f"(estimated input {estimated_input_tokens} tokens; project budget "
            f"{project_budget}; model input limit {model_input_limit}; output "
            f"allowance {output_allowance}). Please "
            "shorten the question or reduce the evidence scope.",
            category="request_too_large",
        )


class RagProviderError(RagServiceError):
    pass


class RagResponseValidationError(RagServiceError):
    pass


@dataclass(frozen=True)
class RagResult:
    mode: Literal["question", "claim"]
    evidence_packet: dict[str, Any]
    response: dict[str, Any]
    response_origin: Literal["application", "model"]
    validation: dict[str, Any]
    model: str | None
    usage: dict[str, Any] | None
    attempts: int
    finish_reason: str | None
    limitations: list[str]
    transaction_status: dict[str, bool]


@dataclass(frozen=True)
class RagPrepared:
    mode: Literal["question", "claim"]
    evidence_packet: dict[str, Any]
    prompt: dict[str, Any]
    schema: dict[str, Any]
    transaction_status: dict[str, bool]
