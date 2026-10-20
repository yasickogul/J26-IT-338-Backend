"""Startup warm-up for the standalone C2 application."""
import logging
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

warmup = {"status": "loading", "error": None}


def _warm_up():
    """Load few-shot examples, the Qdrant connection + embedding model, and the OCR reader in the
    background so the first analysis is not slow."""
    try:
        from .fewshot import load_fewshot
        from .ocr import get_ocr_reader
        from .retrieval import init_retrieval

        load_fewshot()
        init_retrieval()
        try:
            get_ocr_reader()
        except Exception as e:  # OCR is only needed for scanned pages; don't fail warm-up on it
            logger.warning("OCR warm-up skipped: %s", e)
        warmup["status"] = "ready"
    except Exception as e:  # noqa: BLE001
        logger.exception("Warm-up failed")
        warmup.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


