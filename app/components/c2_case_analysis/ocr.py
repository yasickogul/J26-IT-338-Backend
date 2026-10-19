"""Document text extraction (notebook cells 5 and 13).

PDF pages with fewer than `min_chars_per_page` characters of embedded text are rasterised
at 200 dpi and read with EasyOCR, exactly like the notebook. Heavy imports are lazy so the
API can start quickly.
"""
import threading

_reader = None
_reader_lock = threading.Lock()


def get_ocr_reader():
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr
                import torch

                _reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    return _reader


def extract_text_from_pdf(path, min_chars_per_page=20, on_page=None):
    import fitz  # PyMuPDF
    import numpy as np
    from PIL import Image

    doc = fitz.open(path)
    try:
        total = len(doc)
        pages_text = []
        for i, page in enumerate(doc, start=1):
            if on_page:
                on_page(i, total)
            text = page.get_text().strip()
            if len(text) < min_chars_per_page:
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_lines = get_ocr_reader().readtext(np.array(img), detail=0, paragraph=True)
                text = "\n".join(ocr_lines)
            pages_text.append(text)
    finally:
        doc.close()
    return "\n".join(pages_text)


def extract_text_from_image(path):
    lines = get_ocr_reader().readtext(path, detail=0, paragraph=True)
    return "\n".join(lines)


def extract_document_text(path, on_page=None):
    ext = path.lower().split(".")[-1]
    if ext == "pdf":
        return extract_text_from_pdf(path, on_page=on_page)
    elif ext in ("png", "jpg", "jpeg", "tiff", "bmp"):
        return extract_text_from_image(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
