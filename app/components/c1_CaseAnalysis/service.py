"""Stage 2 orchestration: intake → document process → legal info extraction."""

from __future__ import annotations

import uuid
from pathlib import Path

from app.components.c1_CaseAnalysis.config import MODEL_VERSION, c1_settings
from app.components.c1_CaseAnalysis.pipeline.document_processor import (
    process_document_text,
    process_pdf,
)
from app.components.c1_CaseAnalysis.pipeline.info_extractor import (
    extract_legal_info,
    intake_to_source_text,
)
from app.components.c1_CaseAnalysis.repository import (
    create_intake,
    ensure_c1_schema,
    get_intake,
    mark_failed,
    save_extraction,
)
from app.components.c1_CaseAnalysis.schemas import (
    CaseIntakeRequest,
    ExtractedLegalInfo,
    ExtractTextRequest,
    IntakeStatusResponse,
)
from app.core.config import settings
from app.core.database import DirectSessionLocal, direct_engine
from app.shared.contracts.legal_info import StructuredLegalInfo


class Stage2Service:
    async def ensure_ready(self) -> None:
        if direct_engine is None:
            raise RuntimeError("DATABASE_URL / DATABASE_URL_DIRECT is not configured")
        await ensure_c1_schema(direct_engine)

    async def process_intake(
        self,
        intake: CaseIntakeRequest,
        *,
        pdf_path: Path | None = None,
        filename: str | None = None,
        mime: str | None = None,
    ) -> ExtractedLegalInfo:
        await self.ensure_ready()
        assert DirectSessionLocal is not None

        document_text = ""
        storage_path = str(pdf_path) if pdf_path is not None else None
        pdf_error: str | None = None
        if pdf_path is not None:
            try:
                processed_pdf = process_pdf(pdf_path)
                document_text = processed_pdf["text"]
            except Exception as exc:  # noqa: BLE001 — corrupt/unreadable PDF
                pdf_error = f"PDF_EXTRACTION_FAILED: {exc}"

        merged = intake_to_source_text(intake, document_text)
        processed = process_document_text(merged)

        async with DirectSessionLocal() as session:
            row = await create_intake(
                session,
                dispute_category=intake.dispute_category,
                intake_json=intake.model_dump(),
                status="processing",
                filename=filename,
                mime=mime,
                storage_path=storage_path,
                raw_text=processed["text"],
            )
            intake_id = str(row.id)
            try:
                if pdf_error and not (intake.description or intake.text or intake.legal_issue):
                    # PDF failed and there is no alternate narrative to extract from.
                    await mark_failed(session, row, pdf_error)
                    await session.commit()
                    raise ValueError(pdf_error)

                result = extract_legal_info(
                    intake_id=intake_id,
                    intake=intake,
                    processed_text=processed["text"],
                )
                await save_extraction(
                    session,
                    intake=row,
                    structured_json=result.model_dump(mode="json"),
                    model_version=MODEL_VERSION,
                    confidence=result.structured_legal_info.confidence,
                    raw_text=processed["text"],
                )
                await session.commit()
                return result
            except ValueError:
                raise
            except Exception as exc:
                await mark_failed(session, row, str(exc))
                await session.commit()
                raise

    async def extract_from_text(self, body: ExtractTextRequest) -> StructuredLegalInfo:
        """Architecture POST /extract-text — returns shared StructuredLegalInfo."""
        intake = CaseIntakeRequest(
            dispute_category=body.dispute_category or "other_civil",
            description=body.text,
            text=body.text,
        )
        result = await self.process_intake(intake)
        return result.structured_legal_info

    async def get_intake_status(self, intake_id: str) -> IntakeStatusResponse:
        await self.ensure_ready()
        assert DirectSessionLocal is not None
        async with DirectSessionLocal() as session:
            row = await get_intake(session, uuid.UUID(intake_id))
            if row is None:
                return IntakeStatusResponse(
                    intake_id=intake_id,
                    status="failed",
                    error="INTAKE_NOT_FOUND",
                )
            result = None
            if row.extraction and row.extraction.structured_json:
                result = ExtractedLegalInfo.model_validate(row.extraction.structured_json)
            return IntakeStatusResponse(
                intake_id=str(row.id),
                status=row.status,  # type: ignore[arg-type]
                result=result,
                error=row.error,
            )

    def store_upload(self, filename: str, data: bytes) -> Path:
        root = Path(settings.file_storage_path) / c1_settings.storage_subdir
        root.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename) or "upload.pdf"
        path = root / f"{uuid.uuid4().hex}_{safe}"
        path.write_bytes(data)
        return path


stage2_service = Stage2Service()
