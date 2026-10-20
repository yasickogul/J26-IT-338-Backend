"""Async claim submission, owner-scoped reads, and worker lifecycle writes.

For submission and owner-scoped reads, the caller owns the session and transaction, including commit,
rollback on errors, and close. A returned claim ID is not a commit receipt:
acknowledge submission only after the caller has committed successfully.
The worker lifecycle methods below instead own short transactions on an idle
injected session; they reject active transactions and leave session cleanup to
the caller. Use a dedicated session per worker operation (including renewal).
Core table statements avoid ORM autoflush during statement execution.
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, insert, literal_column, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .claim_lifecycle_contract import (
    FAILURE_CATEGORIES,
    ClaimedTextClaim,
    ClaimStateError,
    ClaimStorageError,
    _require_uuid,
    _result_snapshot,
)
from .models import Claim, LegalSource, VerificationResult


class ClaimPersistenceError(RuntimeError):
    """The claim could not be written to storage."""


_DETECTED_LANGUAGE_LABELS = frozenset({"en", "si", "ta", "mixed"})


def _require_owner_hash(value: str) -> None:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("anonymous_session_hash must be a SHA-256 hex digest")


async def create_text_claim(
    raw_input_text: str,
    *,
    anonymous_session_hash: str,
    session: AsyncSession,
    cleaned_text: str | None = None,
    detected_language: str | None = None,
) -> UUID:
    """Insert a queued anonymous claim; the caller must commit the transaction.

    Preserve supplied text verbatim. Database defaults set queued_at and
    created_at. Token hashing and text preprocessing belong to the caller.
    """
    if not isinstance(raw_input_text, str) or not raw_input_text.strip():
        raise ValueError("raw_input_text must be a nonblank string")
    if cleaned_text is not None and (
        not isinstance(cleaned_text, str) or not cleaned_text.strip()
    ):
        raise ValueError("cleaned_text must be a nonblank string when provided")
    _require_owner_hash(anonymous_session_hash)
    if detected_language is not None and detected_language not in _DETECTED_LANGUAGE_LABELS:
        raise ValueError("detected_language must be en, si, ta, or mixed")
    claim_id = uuid4()
    statement = insert(Claim.__table__).values(
        claim_id=claim_id, input_type="text", raw_input_text=raw_input_text,
        cleaned_text=cleaned_text, anonymous_session_hash=anonymous_session_hash,
        detected_language=detected_language, processing_status="queued",
    )
    try:
        await session.execute(statement)
    except Exception as exc:
        raise ClaimPersistenceError("Unable to save the claim.") from exc
    return claim_id


async def get_owned_claim_state(
    claim_id: UUID, anonymous_session_hash: str, *, session: AsyncSession,
) -> dict[str, Any] | None:
    """Return state only for its owner; missing and foreign claims return None."""
    _require_uuid(claim_id)
    _require_owner_hash(anonymous_session_hash)
    claims = Claim.__table__.c
    statement = select(
        claims.claim_id, claims.processing_status, claims.queued_at,
        claims.started_at, claims.completed_at, claims.failed_at, claims.failure_category,
    ).where(claims.claim_id == claim_id, claims.anonymous_session_hash == anonymous_session_hash)
    try:
        result = await session.execute(statement)
        row = result.mappings().one_or_none()
        return dict(row) if row is not None else None
    except Exception as exc:
        raise ClaimStorageError("Unable to read claim storage.") from exc


async def anonymous_session_has_claim(
    anonymous_session_hash: str, *, session: AsyncSession,
) -> bool:
    """Check whether an anonymous session already owns a claim."""
    _require_owner_hash(anonymous_session_hash)
    claims = Claim.__table__.c
    statement = select(claims.claim_id).where(
        claims.anonymous_session_hash == anonymous_session_hash,
    ).limit(1)
    try:
        result = await session.execute(statement)
        return result.scalar_one_or_none() is not None
    except Exception as exc:
        raise ClaimStorageError("Unable to read claim storage.") from exc


async def get_owned_claim_result(
    claim_id: UUID, anonymous_session_hash: str, *, session: AsyncSession,
) -> dict[str, Any] | None:
    """Return owned lifecycle metadata and the exact stored response snapshot.

    A missing/foreign claim returns None. An owned claim without a result
    returns metadata with result_payload=None, preserving the source contract.
    Consumers must check processing_status before presenting a completed result.
    """
    _require_uuid(claim_id)
    _require_owner_hash(anonymous_session_hash)
    claims = Claim.__table__
    results = VerificationResult.__table__
    statement = select(
        claims.c.claim_id, claims.c.processing_status, claims.c.queued_at,
        claims.c.started_at, claims.c.completed_at, claims.c.failed_at,
        results.c.result_payload,
    ).select_from(claims.outerjoin(results, results.c.claim_id == claims.c.claim_id)).where(
        claims.c.claim_id == claim_id,
        claims.c.anonymous_session_hash == anonymous_session_hash,
    )
    try:
        result = await session.execute(statement)
        row = result.mappings().one_or_none()
        return dict(row) if row is not None else None
    except Exception as exc:
        raise ClaimStorageError("Unable to read claim storage.") from exc


async def list_owned_claim_history(
    anonymous_session_hash: str, *, page: int, page_size: int,
    session: AsyncSession,
) -> list[dict[str, Any]]:
    """Return one owner's claims with the persisted verdict, newest first."""
    _require_owner_hash(anonymous_session_hash)
    if type(page) is not int or page < 1:
        raise ValueError("page must be a positive integer")
    if type(page_size) is not int or not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")
    claims = Claim.__table__
    results = VerificationResult.__table__
    statement = select(
        claims.c.claim_id, claims.c.extracted_claim, claims.c.processing_status,
        claims.c.created_at, results.c.verdict,
    ).select_from(
        claims.outerjoin(results, results.c.claim_id == claims.c.claim_id)
    ).where(
        claims.c.anonymous_session_hash == anonymous_session_hash,
    ).order_by(
        claims.c.created_at.desc(), claims.c.claim_id.desc(),
    ).offset((page - 1) * page_size).limit(page_size)
    try:
        result = await session.execute(statement)
        return [dict(row) for row in result.mappings().all()]
    except Exception as exc:
        raise ClaimStorageError("Unable to read claim history.") from exc


async def list_legal_sources(
    *, legal_domain: str | None, session: AsyncSession,
) -> list[dict[str, Any]]:
    """List C3-owned legal-source metadata without exposing full source text."""
    if legal_domain is not None and (
        not isinstance(legal_domain, str) or not legal_domain.strip()
    ):
        raise ValueError("legal_domain must be nonblank when provided")
    documents = LegalSource.__table__
    statement = select(
        documents.c.source_id, documents.c.title, documents.c.document_type.label("source_type"),
        documents.c.citation, documents.c.language, documents.c.source_url,
        documents.c.metadata,
    ).order_by(documents.c.title, documents.c.source_id)
    if legal_domain is not None:
        statement = statement.where(
            documents.c.legal_domain == legal_domain.strip()
        )
    try:
        result = await session.execute(statement)
        return [dict(row) for row in result.mappings().all()]
    except Exception as exc:
        raise ClaimStorageError("Unable to read legal-source storage.") from exc


@asynccontextmanager
async def _worker_write_transaction(session: AsyncSession) -> AsyncIterator[None]:
    """Commit a worker operation, or roll it back without closing its session.

    Reject caller-managed/autobegun transactions before touching them. The
    caller must supply a dedicated idle session, never share it with a heartbeat.
    """
    if session.in_transaction():
        raise ValueError("worker lifecycle operations require an idle session")
    try:
        async with session.begin():
            yield
    except ClaimStateError:
        raise
    except Exception as exc:
        raise ClaimStorageError("Unable to update claim storage.") from exc


def _validate_lease_seconds(lease_seconds: int) -> None:
    if type(lease_seconds) is not int or not 1 <= lease_seconds <= 86400:
        raise ValueError("lease_seconds must be an integer from 1 to 86400")


def _lease_deadline(lease_seconds: int):
    return func.clock_timestamp() + lease_seconds * literal_column("INTERVAL '1 second'")


def _owned_lease(claim_id: UUID, worker_token: UUID):
    claims = Claim.__table__.c
    return (
        claims.claim_id == claim_id,
        claims.processing_status == "processing",
        claims.worker_token == worker_token,
        claims.lease_expires_at > func.clock_timestamp(),
    )


async def _lock_claim(session: AsyncSession, claim_id: UUID) -> None:
    # Separate statement: evaluate the live expiry only AFTER waiting for the
    # row lock, matching CompNet's protection against stale UPDATE predicates.
    claims = Claim.__table__.c
    await session.execute(select(claims.claim_id).where(
        claims.claim_id == claim_id,
    ).with_for_update())


async def claim_next_queued_text_claim(
    *, session: AsyncSession, lease_seconds: int = 3600,
) -> ClaimedTextClaim | None:
    """Commit one exclusive lease before returning work; None means no work."""
    _validate_lease_seconds(lease_seconds)
    worker_token = uuid4()
    claims = Claim.__table__
    next_claim = select(claims.c.claim_id).where(
        claims.c.processing_status == "queued", claims.c.input_type == "text",
    ).order_by(claims.c.queued_at, claims.c.claim_id).limit(1).with_for_update(
        skip_locked=True,
    ).cte("next_claim")
    statement = update(claims).where(claims.c.claim_id == next_claim.c.claim_id).values(
        processing_status="processing", started_at=func.current_timestamp(),
        worker_token=worker_token, lease_expires_at=_lease_deadline(lease_seconds),
    ).returning(claims.c.claim_id, claims.c.raw_input_text, claims.c.cleaned_text)
    async with _worker_write_transaction(session):
        result = await session.execute(statement)
        row = result.mappings().one_or_none()
        if row is None:
            return None
        return ClaimedTextClaim(row["claim_id"], row["raw_input_text"], worker_token, row["cleaned_text"])


async def renew_claim_lease(
    claim_id: UUID, worker_token: UUID, *, session: AsyncSession, lease_seconds: int = 3600,
) -> None:
    """Commit a non-shortening renewal only for this worker's unexpired lease."""
    _require_uuid(claim_id)
    _require_uuid(worker_token)
    _validate_lease_seconds(lease_seconds)
    claims = Claim.__table__
    async with _worker_write_transaction(session):
        await _lock_claim(session, claim_id)
        result = await session.execute(update(claims).where(*_owned_lease(claim_id, worker_token)).values(
            lease_expires_at=func.greatest(claims.c.lease_expires_at, _lease_deadline(lease_seconds)),
        ).returning(claims.c.claim_id))
        if result.scalar_one_or_none() is None:
            raise ClaimStateError("Claim lease is expired or no longer owned by this worker.")


async def record_extracted_claim(
    claim_id: UUID, extracted_claim: str, worker_token: UUID, *, session: AsyncSession,
) -> None:
    """Commit extraction text only while this worker owns an active lease."""
    _require_uuid(claim_id)
    _require_uuid(worker_token)
    if not isinstance(extracted_claim, str) or not extracted_claim.strip():
        raise ValueError("extracted_claim must be nonblank text")
    claims = Claim.__table__
    async with _worker_write_transaction(session):
        await _lock_claim(session, claim_id)
        result = await session.execute(update(claims).where(*_owned_lease(claim_id, worker_token)).values(
            extracted_claim=extracted_claim,
        ).returning(claims.c.claim_id))
        if result.scalar_one_or_none() is None:
            raise ClaimStateError("Claim lease is expired or no longer owned by this worker.")


async def record_completed_result(
    claim_id: UUID, result_payload: Mapping[str, Any], *, retrieval_top_k: int,
    worker_token: UUID, session: AsyncSession,
) -> UUID:
    """Commit snapshot insertion and completion together under a valid lease.

    A worker token is mandatory; legacy unleased processing is not exposed.
    The unique claim_id constraint rejects duplicate results. Any insert failure
    rolls back the status transition in the same transaction.
    """
    _require_uuid(claim_id)
    if type(retrieval_top_k) is not int or not 1 <= retrieval_top_k <= 50:
        raise ValueError("retrieval_top_k must be an integer from 1 to 50")
    snapshot, verdict = _result_snapshot(result_payload)
    _require_uuid(worker_token)
    result_id = uuid4()
    claims = Claim.__table__
    async with _worker_write_transaction(session):
        await _lock_claim(session, claim_id)
        result = await session.execute(update(claims).where(*_owned_lease(claim_id, worker_token)).values(
            processing_status="completed", completed_at=func.current_timestamp(),
            worker_token=None, lease_expires_at=None,
        ).returning(claims.c.claim_id))
        if result.scalar_one_or_none() is None:
            raise ClaimStateError("Claim is not processing under this worker lease.")
        await session.execute(insert(VerificationResult.__table__).values(
            result_id=result_id, claim_id=claim_id, verdict=verdict, confidence_score=None,
            explanation=snapshot["explanation"], llm_model_name=snapshot["model"],
            retrieval_top_k=retrieval_top_k, result_payload=snapshot,
        ))
    return result_id


async def record_claim_failure(
    claim_id: UUID, failure_category: str, *, worker_token: UUID, session: AsyncSession,
) -> None:
    """Commit a safe failure category only for the current unexpired worker."""
    _require_uuid(claim_id)
    if failure_category not in FAILURE_CATEGORIES:
        raise ValueError("unsupported failure category")
    _require_uuid(worker_token)
    claims = Claim.__table__
    async with _worker_write_transaction(session):
        await _lock_claim(session, claim_id)
        result = await session.execute(update(claims).where(*_owned_lease(claim_id, worker_token)).values(
            processing_status="failed", failed_at=func.current_timestamp(),
            failure_category=failure_category, worker_token=None, lease_expires_at=None,
        ).returning(claims.c.claim_id))
        if result.scalar_one_or_none() is None:
            raise ClaimStateError("Claim cannot be failed or does not exist.")


async def recover_expired_claims(*, session: AsyncSession) -> list[UUID]:
    """Commit interrupted failures for expired leased work; never requeue it."""
    claims = Claim.__table__
    async with _worker_write_transaction(session):
        result = await session.execute(update(claims).where(
            claims.c.processing_status == "processing", claims.c.worker_token.is_not(None),
            claims.c.lease_expires_at <= func.clock_timestamp(),
        ).values(
            processing_status="failed", failed_at=func.current_timestamp(),
            failure_category="interrupted", worker_token=None, lease_expires_at=None,
        ).returning(claims.c.claim_id))
        return list(result.scalars())
