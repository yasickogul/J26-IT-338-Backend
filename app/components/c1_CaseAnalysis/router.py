"""C1 Case Analysis routes — Stage 2 intake & extraction (`/api/v1/c1`)."""

from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.components.c1_CaseAnalysis.schemas import (
    CaseIntakeRequest,
    ExtractedLegalInfo,
    ExtractTextRequest,
    IntakeStatusResponse,
)
from app.components.c1_CaseAnalysis.service import stage2_service
from app.shared.contracts.legal_info import StructuredLegalInfo

router = APIRouter(tags=["C1 – Case Analysis"])


@router.get("/ping")
async def ping():
    return {"component": "c1", "status": "ok", "stage": 2}


@router.post("/intake", response_model=ExtractedLegalInfo)
async def create_intake(body: CaseIntakeRequest) -> ExtractedLegalInfo:
    """Structured current-case intake → Stage 2 legal information extraction."""
    try:
        return await stage2_service.process_intake(body)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "INTAKE_PROCESSING_FAILED", "message": str(exc)}},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/intake/document", response_model=ExtractedLegalInfo)
async def create_intake_with_document(
    file: UploadFile = File(...),  # noqa: B008 — FastAPI UploadFile dependency
    intake_json: str = Form(
        ...,
        description="JSON string of CaseIntakeRequest fields (dispute_category required)",
    ),
) -> ExtractedLegalInfo:
    """Multipart: PDF/document + intake JSON form field."""
    try:
        payload = CaseIntakeRequest.model_validate(json.loads(intake_json))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid intake_json: {exc}") from exc

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    content_type = (file.content_type or "").lower()
    name = (file.filename or "").lower()
    if content_type and "pdf" not in content_type and not name.endswith(".pdf"):
        # Soft check — still attempt extraction for allowed legal uploads later
        pass

    path = stage2_service.store_upload(file.filename or "document.pdf", data)
    try:
        return await stage2_service.process_intake(
            payload,
            pdf_path=path,
            filename=file.filename,
            mime=file.content_type,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "PDF_EXTRACTION_FAILED", "message": str(exc)}},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/intake/{intake_id}", response_model=IntakeStatusResponse)
async def get_intake(intake_id: str) -> IntakeStatusResponse:
    result = await stage2_service.get_intake_status(intake_id)
    if result.error == "INTAKE_NOT_FOUND":
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "INTAKE_NOT_FOUND", "message": "Intake not found"}},
        )
    return result


@router.post("/extract-text", response_model=StructuredLegalInfo)
async def extract_text(body: ExtractTextRequest) -> StructuredLegalInfo:
    """Free-text extraction → shared StructuredLegalInfo contract."""
    try:
        return await stage2_service.extract_from_text(body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/health/stage2")
async def stage2_health():
    try:
        await stage2_service.ensure_ready()
        return {"status": "ok", "component": "c1", "stage": 2}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(exc)})
