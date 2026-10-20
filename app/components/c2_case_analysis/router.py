"""C2 upload, progress, and report endpoints."""
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from . import config, jobs
from .pdf_export import export_report_pdf
from .runtime import warmup

router = APIRouter(tags=["C2 - Civil Case Intelligence"])


@router.get("/health")
def health():
    return {
        "status": "ok",
        "warmup": warmup["status"],
        "warmup_error": warmup["error"],
        "llm_model": config.LLM_MODEL,
        "max_upload_mb": config.MAX_UPLOAD_MB,
        "allowed_extensions": sorted(config.ALLOWED_EXTENSIONS),
    }


@router.post("/analyze", status_code=202)
async def analyze(file: UploadFile = File(...)):
    original_name = os.path.basename(file.filename or "document")
    ext = original_name.lower().rsplit(".", 1)[-1] if "." in original_name else ""
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '.{ext}'. Upload one of: "
                   + ", ".join(sorted(config.ALLOWED_EXTENSIONS)),
        )

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    fd, tmp_path = tempfile.mkstemp(suffix=f".{ext}", dir=config.UPLOAD_DIR)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail=f"File is larger than {config.MAX_UPLOAD_MB} MB.")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    job_id = jobs.create_job(tmp_path, original_name)
    return {"job_id": job_id}


def _get_job_or_404(job_id):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found (it may have expired).")
    return job


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    return jobs.public_status(_get_job_or_404(job_id))


@router.delete("/jobs/{job_id}", status_code=202)
def cancel_job(job_id: str):
    _get_job_or_404(job_id)
    jobs.cancel_job(job_id)
    return {"cancelled": True}


@router.get("/jobs/{job_id}/extraction")
def job_extraction(job_id: str):
    job = _get_job_or_404(job_id)
    if job["extraction"] is None:
        raise HTTPException(status_code=409, detail="Extraction is not ready yet.")
    return job["extraction"]


@router.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    job = _get_job_or_404(job_id)
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail="The report is not ready yet.")
    return {
        "document_id": job["document_id"],
        "extraction": job["extraction"],
        "report": job["report"],
        "markdown": job["markdown"],
        "warnings": job["warnings"],
    }


@router.get("/jobs/{job_id}/report.pdf")
def job_pdf(job_id: str):
    job = _get_job_or_404(job_id)
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail="The report is not ready yet.")
    pdf_bytes = export_report_pdf(job["report"])
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="civil_case_intelligence_report.pdf"'},
    )


@router.get("/jobs/{job_id}/report.md")
def job_markdown(job_id: str):
    job = _get_job_or_404(job_id)
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail="The report is not ready yet.")
    return Response(
        content=job["markdown"],
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="civil_case_intelligence_report.md"'},
    )
