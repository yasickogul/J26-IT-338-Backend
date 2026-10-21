"""Qwen claim extraction provider for the existing C3 extraction contract."""

from __future__ import annotations

from typing import Any

from ..agents.gemini_adapter import GeminiConfigurationError, GeminiInvalidResponseError
from ..agents.qwen_adapter import LocalQwenAdapter
from .claim_extraction import (
    ClaimExtractionProviderError,
    ClaimExtractionProviderResponse,
)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["extracted", "multiple_claims", "ambiguous", "no_legal_assertion"]},
        "extracted_claim": {"type": ["string", "null"]},
    },
    "required": ["status", "extracted_claim"],
    "additionalProperties": False,
}


class QwenClaimExtractionProvider:
    def __init__(self, *, adapter: LocalQwenAdapter | None = None) -> None:
        self._adapter = adapter or LocalQwenAdapter()

    def extract(self, cleaned_text: str) -> ClaimExtractionProviderResponse:
        if not isinstance(cleaned_text, str) or not cleaned_text.strip():
            raise ClaimExtractionProviderError("invalid_provider_response")
        prompt = {
            "trusted_instructions": (
                "Perform legal-claim extraction only. Identify one concise, self-contained "
                "legal assertion only when clearly present. Preserve negations, conditions, "
                "exceptions, statutory references, amounts, names, dates, and uncertainty. "
                "Do not verify, explain, classify truth, or invent facts. Multiple assertions "
                "must return multiple_claims with null extracted_claim. Ambiguous or non-legal "
                "input must return the corresponding status with null extracted_claim."
            ),
            "input": {"cleaned_text": cleaned_text},
        }
        try:
            generated = self._adapter.generate_json(prompt, _SCHEMA)
        except GeminiConfigurationError:
            raise ClaimExtractionProviderError("provider_unavailable") from None
        except GeminiInvalidResponseError:
            raise ClaimExtractionProviderError("invalid_provider_response") from None
        except Exception:  # noqa: BLE001 - provider boundary returns safe code
            raise ClaimExtractionProviderError("provider_error") from None
        response = generated.response
        status = response.get("status")
        claim = response.get("extracted_claim")
        if status not in {"extracted", "multiple_claims", "ambiguous", "no_legal_assertion"}:
            raise ClaimExtractionProviderError("invalid_provider_response")
        if status == "extracted":
            if not isinstance(claim, str) or not claim.strip():
                raise ClaimExtractionProviderError("invalid_provider_response")
            claim = claim.strip()
        elif claim is not None:
            raise ClaimExtractionProviderError("invalid_provider_response")
        return ClaimExtractionProviderResponse(status=status, extracted_claim=claim, method=f"qwen:{generated.model}")


__all__ = ["QwenClaimExtractionProvider"]
