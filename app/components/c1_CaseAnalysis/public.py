"""The ONLY module other components may import from c1."""

from __future__ import annotations

from app.components.c1_CaseAnalysis.schemas import CaseIntakeRequest, ExtractedLegalInfo
from app.components.c1_CaseAnalysis.service import stage2_service
from app.shared.contracts.legal_info import StructuredLegalInfo


async def analyse_text(text: str, dispute_category: str | None = None) -> StructuredLegalInfo:
    from app.components.c1_CaseAnalysis.schemas import ExtractTextRequest

    return await stage2_service.extract_from_text(
        ExtractTextRequest(text=text, dispute_category=dispute_category)  # type: ignore[arg-type]
    )


async def get_structured_info(document_id: str) -> StructuredLegalInfo:
    status = await stage2_service.get_intake_status(document_id)
    if not status.result:
        raise LookupError(f"No extraction for intake_id={document_id}")
    return status.result.structured_legal_info


async def ingest_case(intake: CaseIntakeRequest) -> ExtractedLegalInfo:
    """Stage 2 facade: structured case intake → extracted legal information."""
    return await stage2_service.process_intake(intake)
