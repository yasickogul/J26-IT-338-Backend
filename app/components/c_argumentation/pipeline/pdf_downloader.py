"""Rate-limited, resumable downloads for official C4 judgment PDFs."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx


@dataclass(slots=True)
class DownloadResult:
    source_document_key: str
    path: Optional[Path]
    status: str
    attempts: int
    error: str | None = None


def download_pdf(
    *,
    source_document_key: str,
    pdf_url: str,
    output_dir: str | Path,
    timeout: float = 45.0,
    max_retries: int = 3,
    max_bytes: int = 50 * 1024 * 1024,
    min_delay: float = 1.0,
    max_delay: float = 2.5,
    user_agent: str = "LexLanka-C4-Research/1.0",
) -> DownloadResult:
    """Download one PDF without bypassing server access controls."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{source_document_key}.pdf"
    if target.exists() and target.stat().st_size > 0:
        return DownloadResult(source_document_key, target, "skipped", 0)

    headers = {"User-Agent": user_agent, "Accept": "application/pdf"}
    last_error = "unknown download error"
    attempts = max(1, max_retries + 1)
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        for attempt in range(1, attempts + 1):
            try:
                response = client.get(pdf_url)
                if response.status_code in {401, 403}:
                    return DownloadResult(source_document_key, None, "access_restricted", attempt, f"HTTP {response.status_code}")
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"retryable HTTP {response.status_code}", request=response.request, response=response
                    )
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > max_bytes:
                    raise ValueError(f"response exceeds {max_bytes} bytes")
                if len(response.content) > max_bytes:
                    raise ValueError(f"response exceeds {max_bytes} bytes")
                if len(response.content) < 100 or not response.content.startswith(b"%PDF-"):
                    raise ValueError("response is not a valid PDF signature")
                content_type = response.headers.get("content-type", "").lower()
                if content_type and "pdf" not in content_type and "octet-stream" not in content_type:
                    raise ValueError(f"unexpected content type: {content_type}")
                temporary = target.with_suffix(".part")
                temporary.write_bytes(response.content)
                if temporary.stat().st_size < 100 or temporary.read_bytes()[:5] != b"%PDF-":
                    temporary.unlink(missing_ok=True)
                    raise ValueError("downloaded file failed PDF validation")
                temporary.replace(target)
                return DownloadResult(source_document_key, target, "downloaded", attempt)
            except (httpx.HTTPError, ValueError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= attempts:
                    break
                time.sleep((2 ** (attempt - 1)) + random.uniform(0.0, 0.5))
            finally:
                if attempt < attempts:
                    time.sleep(max(0.0, random.uniform(min_delay, max_delay)))
    return DownloadResult(source_document_key, None, "failed", attempts, last_error)
