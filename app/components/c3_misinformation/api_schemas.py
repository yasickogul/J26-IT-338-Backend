"""HTTP request and response schemas for C3 claim lifecycle endpoints."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StrictStr


class ClaimSubmissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: StrictStr


class ClaimSubmissionResponse(BaseModel):
    claim_id: UUID
    status: Literal["processing"]


class ClaimResultResponse(BaseModel):
    claim_id: UUID
    status: Literal["queued", "processing", "completed", "failed"]
    queued_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    failed_at: datetime | None
    result: dict[str, Any] | None


class ClaimHistoryItem(BaseModel):
    claim_id: UUID
    extracted_claim: str | None
    status: Literal["queued", "processing", "completed", "failed"]
    verdict: Literal["True", "False", "Misleading", "insufficient_evidence"] | None
    created_at: datetime


class ClaimHistoryResponse(BaseModel):
    items: list[ClaimHistoryItem]
    page: int
    page_size: int


class LegalSourceResponse(BaseModel):
    source_id: str
    title: str
    source_type: str
    citation: str | None = None
    language: str | None = None
    source_url: str | None = None
    metadata: dict[str, Any]
