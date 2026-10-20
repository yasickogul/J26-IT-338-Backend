"""C1 Case Analysis routes — Stage 2 intake + Stage 3 semantic retrieval (`/api/v1/c1`)."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.components.c1_CaseAnalysis.retrieval.semantic import SemanticRetrievalError
from app.components.c1_CaseAnalysis.schemas import (
    CaseIntakeRequest,
    ExtractedLegalInfo,
    ExtractTextRequest,
    IntakeSimilarCasesRequest,
    IntakeSimilarCasesResponse,
    IntakeStatusResponse,
    SimilarCasesRequest,
    SimilarCasesResponse,
)
from app.components.c1_CaseAnalysis.service import stage2_service
from app.shared.contracts.legal_info import StructuredLegalInfo

logger = logging.getLogger("uvicorn.error")
router = APIRouter(tags=["C1 – Case Analysis"])


@router.get("/ping")
async def ping():
    return {
        "component": "c1",
        "status": "ok",
        "stage": 3,
        "capabilities": {
            "intake_extraction": True,
            "semantic_retrieval": True,
            "hybrid_retrieval": False,
            "legal_relevance_score": False,
            "lm_studio_optional": True,
        },
    }


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


@router.post("/similar-cases", response_model=SimilarCasesResponse)
async def similar_cases(body: SimilarCasesRequest) -> SimilarCasesResponse:
    """Standalone semantic similar-judgment search over the shared Neon corpus."""
    try:
        return await stage2_service.search_similar_cases(body)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "INVALID_SIMILAR_CASES_REQUEST", "message": str(exc)}},
        ) from exc
    except SemanticRetrievalError as exc:
        logger.exception("C1 similar-cases retrieval failed code=%s", exc.code)
        raise HTTPException(
            status_code=503,
            detail={"error": {"code": exc.code, "message": exc.message}},
        ) from exc
    except Exception as exc:
        logger.exception("C1 similar-cases unexpected failure")
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "SIMILAR_CASES_FAILED", "message": str(exc)}},
        ) from exc


@router.post("/intake/{intake_id}/similar-cases", response_model=IntakeSimilarCasesResponse)
async def intake_similar_cases(
    intake_id: str,
    body: IntakeSimilarCasesRequest | None = None,
) -> IntakeSimilarCasesResponse:
    """Retrieve similar judgments for a previously saved Stage 2 intake."""
    try:
        return await stage2_service.search_similar_cases_for_intake(intake_id, body)
    except ValueError as exc:
        code = str(exc)
        status = 404 if code == "INTAKE_NOT_FOUND" else 422
        logger.warning("C1 intake similar-cases rejected intake_id=%s code=%s", intake_id, code)
        raise HTTPException(
            status_code=status,
            detail={"error": {"code": code, "message": code.replace("_", " ").title()}},
        ) from exc
    except SemanticRetrievalError as exc:
        logger.exception(
            "C1 intake similar-cases retrieval failed intake_id=%s code=%s",
            intake_id,
            exc.code,
        )
        raise HTTPException(
            status_code=503,
            detail={"error": {"code": exc.code, "message": exc.message}},
        ) from exc
    except Exception as exc:
        logger.exception("C1 intake similar-cases unexpected failure intake_id=%s", intake_id)
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "INTAKE_SIMILAR_CASES_FAILED", "message": str(exc)}},
        ) from exc


@router.get("/health/stage2")
async def stage2_health():
    try:
        await stage2_service.ensure_ready()
        return {"status": "ok", "component": "c1", "stage": 2}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(exc)})


@router.get("/health/stage3")
async def stage3_health():
    try:
        await stage2_service.ensure_ready()
        return {
            "status": "ok",
            "component": "c1",
            "stage": 3,
            "retrieval": "semantic_pgvector_cosine",
            "hybrid_retrieval": False,
        }
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(exc)})
