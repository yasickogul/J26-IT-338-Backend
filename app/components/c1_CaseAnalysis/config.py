"""C1 Case Analysis settings (Stage 2: intake + extraction)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

# Frontend / intake dispute categories (distinct from corpus case_type="civil")
DISPUTE_CATEGORIES: dict[str, str] = {
    "contract": "Contract",
    "property_land": "Property",
    "employment": "Employment",
    "rental": "Rental",
    "other_civil": "Other civil",
}

DISPUTE_CATEGORY_LABELS: dict[str, str] = {
    "contract": "Contract dispute",
    "property_land": "Property or land dispute",
    "employment": "Employment dispute",
    "rental": "Rental dispute",
    "other_civil": "Other civil dispute",
}

OCR_MIN_CHARS = 200
MODEL_VERSION = "stage2-rules-v1"
ANALYSIS_MODEL_VERSION = "stage3-semantic-v1"
DISCLAIMER = "This is informational and not legal advice."
RETRIEVAL_DISCLAIMER = (
    "Retrieval similarity measures semantic closeness in the judgment corpus. "
    "It is not a legal relevance percentage, binding-authority score, or prediction of case outcome."
)


class C1Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="C1_", env_file=".env", extra="ignore")

    storage_subdir: str = "c1_case_analysis"
    ocr_min_chars: int = OCR_MIN_CHARS
    retrieval_top_k_default: int = 5
    retrieval_top_k_max: int = 20
    # Optional C1-owned LM Studio OpenAI-compatible endpoint (does not change app.core.llm).
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = ""
    lm_studio_timeout_seconds: float = 45.0
    lm_studio_enabled: bool = False


c1_settings = C1Settings()
