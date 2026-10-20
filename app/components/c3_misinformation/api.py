"""Anonymous claim submission and owner-scoped result retrieval."""

import os
from collections.abc import Callable
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session

from . import repository
from .anonymous_claim_session import (
    COOKIE_NAME,
    SECURE_COOKIE_NAME,
    new_session_token,
    session_token_hash,
    valid_session_token,
)
from .api_schemas import (
    ClaimHistoryResponse,
    ClaimResultResponse,
    ClaimSubmissionRequest,
    ClaimSubmissionResponse,
    LegalSourceResponse,
)
from .claim_lifecycle_contract import ClaimStorageError
from .config import language_detection_enabled
from .pipeline.claim_input_tokens import (
    EmbeddingInputTooLongError,
    TokenizerUnavailableError,
    validate_claim_input_tokens,
)
from .pipeline.fasttext_language_identifier import get_fasttext_language_identifier
from .pipeline.language_detection import detect_language
from .pipeline.text_preprocessing import clean_text

router = APIRouter(tags=["C3 claims"])
ClaimTokenValidator = Callable[[list[str]], list[int]]


def get_claim_token_validator() -> ClaimTokenValidator:
    return validate_claim_input_tokens


def _error_response(
    status_code: int, code: str, *, category: str | None = None,
) -> JSONResponse:
    detail = {"code": code}
    if category is not None:
        detail["category"] = category
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail},
        headers={"Cache-Control": "no-store"},
    )


def _detect_claim_language(text: str) -> str | None:
    if not language_detection_enabled():
        return None
    result = detect_language(
        text,
        language_identifier=get_fasttext_language_identifier(),
    )
    return result.label if result.status == "detected" else None


def _cookie_name(request: Request) -> tuple[str, bool]:
    secure = request.url.scheme == "https" or os.getenv("COMPNET_SECURE_COOKIES") == "1"
    return (SECURE_COOKIE_NAME if secure else COOKIE_NAME), secure


@router.post(
    "/claims/analyze",
    status_code=202,
    response_model=ClaimSubmissionResponse,
)
async def submit_claim(
    body: ClaimSubmissionRequest,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_session),  # noqa: B008 - FastAPI DI
    token_validator: ClaimTokenValidator = Depends(  # noqa: B008 - FastAPI DI
        get_claim_token_validator
    ),
) -> ClaimSubmissionResponse | JSONResponse:
    """Persist one validated claim without running retrieval or a worker."""
    response.headers["Cache-Control"] = "no-store"
    if not body.text.strip():
        return _error_response(400, "invalid_claim_text", category="input")

    cleaned_text = clean_text(body.text)
    if not cleaned_text:
        return _error_response(400, "invalid_claim_text", category="input")
    try:
        token_validator([body.text, cleaned_text])
    except EmbeddingInputTooLongError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "detail": {
                    "code": "claim_text_too_long",
                    "category": "input",
                    "limit": exc.limit,
                },
            },
            headers={"Cache-Control": "no-store"},
        )
    except TokenizerUnavailableError:
        return _error_response(
            503, "claim_tokenizer_unavailable", category="preparation",
        )
    except Exception:  # noqa: BLE001 - API boundary returns a safe error
        return _error_response(500, "claim_validation_failed", category="preparation")
    try:
        detected_language = _detect_claim_language(cleaned_text)
    except Exception:  # noqa: BLE001 - API boundary returns a safe error
        return _error_response(500, "claim_language_detection_failed", category="preparation")

    cookie_name, secure_cookie = _cookie_name(request)
    incoming_token = request.cookies.get(cookie_name)
    issue_cookie = True
    owner_hash: str
    if valid_session_token(incoming_token):
        owner_hash = session_token_hash(incoming_token)
    else:
        owner_hash = ""

    try:
        async with session.begin():
            if owner_hash:
                issue_cookie = not await repository.anonymous_session_has_claim(
                    owner_hash, session=session,
                )
            if issue_cookie:
                token = new_session_token()
                owner_hash = session_token_hash(token)

            claim_id = await repository.create_text_claim(
                body.text,
                cleaned_text=cleaned_text,
                anonymous_session_hash=owner_hash,
                detected_language=detected_language,
                session=session,
            )
    except Exception:  # noqa: BLE001 - API boundary returns a safe error
        return _error_response(500, "claim_persistence_failed", category="storage")

    if issue_cookie:
        response.set_cookie(
            key=cookie_name,
            value=token,
            httponly=True,
            secure=secure_cookie,
            samesite="lax",
            path="/",
        )
    return ClaimSubmissionResponse(claim_id=claim_id, status="processing")


@router.get(
    "/results/{claim_id}",
    response_model=ClaimResultResponse,
    responses={401: {"description": "Invalid anonymous claim session"}},
)
async def read_claim_result(
    claim_id: UUID,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_session),  # noqa: B008 - FastAPI DI
) -> ClaimResultResponse | JSONResponse:
    """Return lifecycle state and the exact stored result to its owner."""
    response.headers["Cache-Control"] = "no-store"
    cookie_name, _ = _cookie_name(request)
    token = request.cookies.get(cookie_name)
    if not valid_session_token(token):
        return _error_response(401, "invalid_claim_session")

    try:
        claim = await repository.get_owned_claim_result(
            claim_id, session_token_hash(token), session=session,
        )
    except ClaimStorageError:
        return _error_response(500, "result_lookup_failed")
    except Exception:  # noqa: BLE001 - API boundary returns a safe error
        return _error_response(500, "result_lookup_failed")

    if claim is None:
        return _error_response(404, "result_not_found")

    try:
        state = claim["processing_status"]
        if state not in {"queued", "processing", "completed", "failed"}:
            return _error_response(500, "result_state_unavailable")
        result: dict[str, Any] | None = (
            claim["result_payload"] if state == "completed" else None
        )
        if state == "completed" and result is None:
            return _error_response(500, "result_unavailable")

        return ClaimResultResponse(
            claim_id=claim["claim_id"],
            status=state,
            queued_at=claim["queued_at"],
            started_at=claim["started_at"],
            completed_at=claim["completed_at"],
            failed_at=claim["failed_at"],
            result=result,
        )
    except (KeyError, TypeError, ValueError):
        return _error_response(500, "result_state_unavailable")


@router.get("/history", response_model=ClaimHistoryResponse)
async def read_claim_history(
    request: Request,
    response: Response,
    page: int = 1,
    page_size: int = 20,
    session: AsyncSession = Depends(get_session),  # noqa: B008 - FastAPI DI
) -> ClaimHistoryResponse | JSONResponse:
    """Return only the current anonymous session's prior submissions."""
    response.headers["Cache-Control"] = "no-store"
    cookie_name, _ = _cookie_name(request)
    token = request.cookies.get(cookie_name)
    if not valid_session_token(token):
        return _error_response(401, "invalid_claim_session")
    if page < 1 or page_size < 1 or page_size > 100:
        return _error_response(400, "invalid_pagination", category="input")
    try:
        rows = await repository.list_owned_claim_history(
            session_token_hash(token), page=page, page_size=page_size, session=session,
        )
        return ClaimHistoryResponse(
            items=[
                {
                    "claim_id": row["claim_id"],
                    "extracted_claim": row["extracted_claim"],
                    "status": row["processing_status"],
                    "verdict": row["verdict"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ],
            page=page,
            page_size=page_size,
        )
    except ClaimStorageError:
        return _error_response(500, "history_lookup_failed")
    except Exception:  # noqa: BLE001 - API boundary returns a safe error
        return _error_response(500, "history_lookup_failed")


@router.get("/legal-sources", response_model=list[LegalSourceResponse])
async def read_legal_sources(
    response: Response,
    legal_domain: str | None = None,
    session: AsyncSession = Depends(get_session),  # noqa: B008 - FastAPI DI
) -> list[LegalSourceResponse] | JSONResponse:
    """Expose source metadata for transparency; legal text stays protected."""
    response.headers["Cache-Control"] = "public, max-age=60"
    try:
        rows = await repository.list_legal_sources(
            legal_domain=legal_domain, session=session,
        )
        return [LegalSourceResponse(**row) for row in rows]
    except ValueError:
        return _error_response(400, "invalid_legal_domain", category="input")
    except ClaimStorageError:
        return _error_response(500, "legal_sources_lookup_failed")
    except Exception:  # noqa: BLE001 - API boundary returns a safe error
        return _error_response(500, "legal_sources_lookup_failed")
