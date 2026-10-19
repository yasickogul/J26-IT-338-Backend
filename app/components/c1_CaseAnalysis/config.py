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
DISCLAIMER = "This is informational and not legal advice."


class C1Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="C1_", env_file=".env", extra="ignore")

    storage_subdir: str = "c1_case_analysis"
    ocr_min_chars: int = OCR_MIN_CHARS


c1_settings = C1Settings()
