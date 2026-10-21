"""Live C3-only Qwen smoke checks; never connects to the database."""

from __future__ import annotations

import asyncio

from app.components.c3_misinformation.agents.qwen_adapter import (
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL,
    LocalQwenAdapter,
)
from app.components.c3_misinformation.agents.qwen_translation_provider import (
    QwenTranslationProvider,
)
from app.components.c3_misinformation.pipeline.claim_translation import translate_claim
from app.components.c3_misinformation.pipeline.qwen_claim_extraction import (
    QwenClaimExtractionProvider,
)
from app.components.c3_misinformation.prompts.rag_prompt import ClaimInput
from app.components.c3_misinformation.rag_contract import RagResponseValidationError
from app.components.c3_misinformation.rag_service_async import (
    AsyncRagService,
    ThreadedGeminiProvider,
)


def _synthetic_packet(query: str) -> dict[str, object]:
    return {
        "query": query,
        "evidence": [
            {
                "evidence_id": "synthetic:statute-1",
                "text": "A tenant may be evicted only after the statutory notice period has expired.",
                "source_id": "synthetic-source",
                "page_number": 1,
            },
        ],
        "relationships": [],
        "context_resolution_notices": [],
        "limitations": ["Synthetic evidence used for this smoke check."],
        "legal_coverage": "synthetic",
    }


async def _verify() -> None:
    adapter = LocalQwenAdapter(
        base_url=DEFAULT_QWEN_BASE_URL,
        model=DEFAULT_QWEN_MODEL,
    )

    extracted = QwenClaimExtractionProvider(adapter=adapter).extract(
        "A tenant may be evicted only after the statutory notice period has expired."
    )
    assert extracted.status == "extracted", extracted
    assert extracted.extracted_claim
    assert extracted.method == f"qwen:{DEFAULT_QWEN_MODEL}", extracted.method
    print(f"extraction: passed ({extracted.method})")

    translation_provider = QwenTranslationProvider(adapter=adapter)
    for source_text, language, prefix in (
        ("සිංහල නීතිමය ප්‍රකාශයක්", "si", "සිංහල"),
        ("தமிழ் சட்ட அறிக்கை", "ta", "தமிழ்"),
    ):
        translated = translate_claim(
            source_text,
            language,
            provider=translation_provider,
        )
        assert translated.status == "success", translated
        assert translated.translated_text and translated.original_text.startswith(prefix)
        assert translated.method == f"qwen:{DEFAULT_QWEN_MODEL}", translated.method
        print(f"translation {language}: passed ({translated.method})")

    async def evidence_packet_builder(
        query: str,
        *,
        top_k: int,
        source_id: str | None,
        legal_domain: str | None,
        language: str | None,
    ) -> dict[str, object]:
        del top_k, source_id, legal_domain, language
        return _synthetic_packet(query)

    service = AsyncRagService(
        evidence_packet_builder=evidence_packet_builder,
        provider=ThreadedGeminiProvider(lambda: adapter),
        token_validator=lambda values: [len(value) for value in values],
    )
    try:
        verified = await service.run(
            ClaimInput(
                claim="A tenant may be evicted only after the statutory notice period has expired."
            ),
            top_k=1,
        )
    except RagResponseValidationError as exc:
        print(f"verification: failed ({exc})")
        raise
    response = verified.response
    assert verified.response_origin == "model"
    assert verified.model == DEFAULT_QWEN_MODEL
    assert verified.validation.get("status") == "passed"
    assert response["claim_verdict"] in {"True", "False", "Misleading", None}
    assert set(response["citations"]).issubset({"synthetic:statute-1"})
    assert "gemini" not in (verified.model or "").casefold()
    print(
        "verification: passed "
        f"(model={verified.model}, verdict={response['claim_verdict']}, "
        f"citations={response['citations']})"
    )


if __name__ == "__main__":
    asyncio.run(_verify())
