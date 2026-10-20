"""Independent claim worker outcome and extraction error contracts."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID


@dataclass(frozen=True)
class WorkerOutcome:
    claim_id: UUID
    status: Literal["completed", "failed", "ownership_lost"]


class ClaimExtractionWorkerError(RuntimeError):
    """Extraction could not safely yield one claim for verification."""

    def __init__(self, reason: str, *, failure_category: str) -> None:
        super().__init__(reason)
        self.failure_category = failure_category
