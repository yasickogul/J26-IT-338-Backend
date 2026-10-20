"""In-memory job store + background runner. Analysing a document takes minutes (OCR + several
LLM calls), so the API is asynchronous: POST a file, poll the job."""
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from .pipeline import DocumentError, run_analysis

logger = logging.getLogger(__name__)

JOB_TTL_SECONDS = 6 * 60 * 60
_executor = ThreadPoolExecutor(max_workers=2)
_jobs = {}
_lock = threading.Lock()


def _now():
    return time.time()


def create_job(file_path, document_id):
    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "document_id": document_id,
        "status": "queued",       # queued | running | complete | failed | cancelled
        "stage": "queued",        # queued | reading | extracting | screening | researching | insights | complete
        "message": "Waiting to start",
        "progress": 0,
        "warnings": [],
        "error": None,
        "extraction": None,
        "report": None,
        "markdown": None,
        "created_at": _now(),
        "updated_at": _now(),
    }
    with _lock:
        _purge_old_jobs()
        _jobs[job_id] = job
    _executor.submit(_run, job_id, file_path, document_id)
    return job_id


def _purge_old_jobs():
    cutoff = _now() - JOB_TTL_SECONDS
    for jid in [j for j, v in _jobs.items() if v["updated_at"] < cutoff]:
        _jobs.pop(jid, None)


def _update(job_id, **fields):
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.update(fields)
            job["updated_at"] = _now()


def get_job(job_id):
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def public_status(job):
    """Light payload for polling (no big report/extraction bodies)."""
    return {
        "job_id": job["job_id"],
        "document_id": job["document_id"],
        "status": job["status"],
        "stage": job["stage"],
        "message": job["message"],
        "progress": job["progress"],
        "warnings": job["warnings"],
        "error": job["error"],
        "extraction_ready": job["extraction"] is not None,
        "report_ready": job["report"] is not None,
    }


class JobCancelled(Exception):
    pass


def cancel_job(job_id):
    """Ask a running job to stop at its next checkpoint (between pages / chunks / issues)."""
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return False
        job["cancelled"] = True
        if job["status"] in ("queued", "running"):
            job["message"] = "Cancelling"
        job["updated_at"] = _now()
        return True


def _is_cancelled(job_id):
    with _lock:
        job = _jobs.get(job_id)
        return bool(job and job.get("cancelled"))


def _run(job_id, file_path, document_id):
    def progress(stage, message, pct):
        if _is_cancelled(job_id):
            raise JobCancelled()
        _update(job_id, status="running", stage=stage, message=message, progress=pct)

    try:
        _update(job_id, status="running", stage="reading", message="Reading the document", progress=1)
        result = run_analysis(
            file_path, document_id, progress,
            on_extraction=lambda unified: _update(job_id, extraction=unified),
        )
        _update(
            job_id,
            status="complete", stage="complete", message="Report ready", progress=100,
            extraction=result["extraction"], report=result["report"],
            markdown=result["markdown"], warnings=result["warnings"],
        )
    except JobCancelled:
        _update(job_id, status="cancelled", message="Cancelled", error=None)
    except DocumentError as e:
        _update(job_id, status="failed", message="Could not read the document", error=str(e))
    except Exception as e:  # noqa: BLE001 - surface any pipeline error to the UI
        logger.exception("Job %s failed", job_id)
        _update(job_id, status="failed", message="Analysis failed",
                error=f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass
