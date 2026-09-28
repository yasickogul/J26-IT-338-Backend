from typing import Optional

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    source_id: str
    title: str
    source_type: str  # act | judgment | regulation | other
    citation: Optional[str] = None
    section: Optional[str] = None
    url: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    citation: SourceCitation
    metadata: dict = Field(default_factory=dict)
