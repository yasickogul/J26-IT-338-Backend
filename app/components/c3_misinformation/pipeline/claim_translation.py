"""Provider-independent translation boundary for legal claim text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


TranslationSourceLanguage = Literal[
    "en", "si", "ta", "mixed", "uncertain", "unsupported",
]
TranslationStatus = Literal["success", "skipped", "uncertain", "failed"]
TranslationErrorCode = Literal[
    "invalid_input",
    "provider_unavailable",
    "provider_error",
    "invalid_provider_response",
]
TranslationTargetLanguage = Literal["en"]


@dataclass(frozen=True)
class TranslationProviderResponse:
    """A translation actually returned by an injected provider."""

    translated_text: str
    method: str


class TranslationProvider(Protocol):
    def translate(
        self,
        text: str,
        *,
        source_language: Literal["si", "ta"],
        target_language: TranslationTargetLanguage,
    ) -> TranslationProviderResponse:
        """Translate supplied text without altering the caller's source value."""


class TranslationProviderError(RuntimeError):
    """A provider failure carrying only a safe, non-content error code."""

    def __init__(self, code: TranslationErrorCode) -> None:
        if code not in {
            "provider_unavailable", "provider_error", "invalid_provider_response",
        }:
            raise ValueError("provider error code is invalid")
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ClaimTranslationResult:
    """Translation outcome; original text always remains authoritative."""

    original_text: str
    source_language: TranslationSourceLanguage | None
    target_language: TranslationTargetLanguage
    translated_text: str | None
    method: str | None
    status: TranslationStatus
    error_code: TranslationErrorCode | None = None


def _result(
    original_text: str,
    source_language: TranslationSourceLanguage | None,
    *,
    status: TranslationStatus,
    translated_text: str | None = None,
    method: str | None = None,
    error_code: TranslationErrorCode | None = None,
) -> ClaimTranslationResult:
    return ClaimTranslationResult(
        original_text=original_text,
        source_language=source_language,
        target_language="en",
        translated_text=translated_text,
        method=method,
        status=status,
        error_code=error_code,
    )


def translate_claim(
    original_text: str,
    source_language: TranslationSourceLanguage | None,
    *,
    provider: TranslationProvider | None = None,
) -> ClaimTranslationResult:
    """Request Sinhala/Tamil-to-English translation without replacing source.

    English is a no-op. Uncertain, unsupported, or mixed-language input is not
    guessed or sent to a whole-text translator. A provider must be explicitly
    injected; this module has no default or fabricated translator.
    """
    if not isinstance(original_text, str):
        raise TypeError("original_text must be a string")
    if source_language is not None and (
        not isinstance(source_language, str)
        or source_language not in {
            "en", "si", "ta", "mixed", "uncertain", "unsupported",
        }
    ):
        raise ValueError("source_language is not supported by the translation contract")
    if not original_text.strip():
        return _result(
            original_text, source_language,
            status="failed", error_code="invalid_input",
        )
    if source_language == "en":
        return _result(original_text, source_language, status="skipped")
    if source_language in {None, "uncertain", "unsupported", "mixed"}:
        return _result(original_text, source_language, status="uncertain")
    if provider is None:
        return _result(
            original_text, source_language, status="failed",
            error_code="provider_unavailable",
        )

    try:
        response = provider.translate(
            original_text,
            source_language=source_language,
            target_language="en",
        )
    except TranslationProviderError as exc:
        return _result(
            original_text, source_language, status="failed", error_code=exc.code,
        )

    if (
        not isinstance(response, TranslationProviderResponse)
        or not isinstance(response.translated_text, str)
        or not response.translated_text.strip()
        or not isinstance(response.method, str)
        or not response.method.strip()
    ):
        return _result(
            original_text, source_language, status="failed",
            error_code="invalid_provider_response",
        )

    return _result(
        original_text,
        source_language,
        status="success",
        translated_text=response.translated_text,
        method=response.method,
    )
