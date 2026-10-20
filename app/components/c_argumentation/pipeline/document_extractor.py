"""Temporary, in-memory extraction for user-uploaded C4 documents."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath
import re
from typing import Final
from uuid import uuid4
import zipfile
import xml.etree.ElementTree as ET

import pymupdf


MAX_UPLOAD_BYTES: Final = 10 * 1024 * 1024
MAX_PDF_PAGES: Final = 500
MAX_DOCX_UNCOMPRESSED_BYTES: Final = 40 * 1024 * 1024
MAX_DOCX_ENTRIES: Final = 1_000


class DocumentUploadError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ExtractedSection:
    label: str
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    document_id: str
    sanitized_filename: str
    file_type: str
    file_size: int
    extraction_status: str
    page_count: int | None
    sections: list[ExtractedSection]
    extracted_text: str
    character_count: int
    warnings: list[str]
    ocr_required: bool


def sanitize_filename(filename: str | None) -> str:
    raw_name = PurePath(filename or "").name
    safe_name = re.sub(r"[^A-Za-z0-9._() -]", "_", raw_name).strip(" .")
    return safe_name[:255] or "uploaded-document"


def extract_uploaded_document(filename: str | None, payload: bytes) -> ExtractedDocument:
    sanitized_filename = sanitize_filename(filename)
    suffix = PurePath(sanitized_filename).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise DocumentUploadError("Only PDF and DOCX files are supported.", 415)
    if not payload:
        raise DocumentUploadError("The uploaded file is empty.", 400)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise DocumentUploadError("The uploaded file exceeds the 10 MB limit.", 413)

    document_id = str(uuid4())
    if suffix == ".pdf":
        return _extract_pdf(document_id, sanitized_filename, payload)
    return _extract_docx(document_id, sanitized_filename, payload)


def _extract_pdf(document_id: str, filename: str, payload: bytes) -> ExtractedDocument:
    if not payload.startswith(b"%PDF-"):
        raise DocumentUploadError("The file is not a valid PDF by signature.", 400)
    try:
        document = pymupdf.open(stream=payload, filetype="pdf")
    except Exception as exc:
        raise DocumentUploadError("The PDF could not be opened or is corrupt.", 400) from exc

    try:
        page_count = len(document)
        if page_count > MAX_PDF_PAGES:
            raise DocumentUploadError(f"PDFs are limited to {MAX_PDF_PAGES} pages.", 413)
        sections: list[ExtractedSection] = []
        warnings: list[str] = []
        image_only_pages: list[int] = []
        for index, page in enumerate(document, start=1):
            text = (page.get_text("text") or "").strip()
            if text:
                sections.append(ExtractedSection(f"page-{index}", text))
            elif page.get_images(full=True):
                image_only_pages.append(index)
            else:
                warnings.append(f"Page {index} has no extractable text.")

        if image_only_pages:
            joined_pages = ", ".join(str(page) for page in image_only_pages)
            warnings.append(f"Pages {joined_pages} contain scanned images without extractable text; OCR is required.")
        elif not sections:
            warnings.append("No extractable text was found; OCR or a different source document may be required.")

        extracted_text = "\n\n".join(f"[Page {section.label.removeprefix('page-')}]\n{section.text}" for section in sections)
        return ExtractedDocument(
            document_id=document_id,
            sanitized_filename=filename,
            file_type="pdf",
            file_size=len(payload),
            extraction_status="ocr_required" if image_only_pages or not sections else "extracted",
            page_count=page_count,
            sections=sections,
            extracted_text=extracted_text,
            character_count=len(extracted_text),
            warnings=warnings,
            ocr_required=bool(image_only_pages or not sections),
        )
    finally:
        document.close()


def _extract_docx(document_id: str, filename: str, payload: bytes) -> ExtractedDocument:
    if not payload.startswith(b"PK"):
        raise DocumentUploadError("The file is not a valid DOCX ZIP package by signature.", 400)
    try:
        archive = zipfile.ZipFile(BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise DocumentUploadError("The DOCX package is corrupt.", 400) from exc

    with archive:
        names = archive.namelist()
        if len(names) > MAX_DOCX_ENTRIES or sum(info.file_size for info in archive.infolist()) > MAX_DOCX_UNCOMPRESSED_BYTES:
            raise DocumentUploadError("The DOCX package exceeds safe extraction limits.", 413)
        if any(name.lower().endswith("vbaproject.bin") for name in names):
            raise DocumentUploadError("Macro-enabled document content is not accepted.", 415)
        if "[Content_Types].xml" not in names or "word/document.xml" not in names:
            raise DocumentUploadError("The uploaded ZIP is not a valid DOCX document.", 400)
        try:
            root = ET.fromstring(archive.read("word/document.xml"))
        except (ET.ParseError, KeyError) as exc:
            raise DocumentUploadError("The DOCX document XML is invalid.", 400) from exc

    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    sections: list[ExtractedSection] = []
    paragraph_number = 0
    table_number = 0
    body = root.find(f"{namespace}body")
    if body is not None:
        for child in body:
            if child.tag == f"{namespace}p":
                text = "".join(node.text or "" for node in child.iter(f"{namespace}t")).strip()
                if text:
                    paragraph_number += 1
                    sections.append(ExtractedSection(f"paragraph-{paragraph_number}", text))
            elif child.tag == f"{namespace}tbl":
                rows: list[str] = []
                for row in child.findall(f"{namespace}tr"):
                    cells = [" ".join(node.text or "" for node in cell.iter(f"{namespace}t")).strip() for cell in row.findall(f"{namespace}tc")]
                    rows.append("\t".join(cells).strip())
                table_text = "\n".join(row for row in rows if row)
                if table_text:
                    table_number += 1
                    sections.append(ExtractedSection(f"table-{table_number}", table_text))

    extracted_text = "\n\n".join(section.text for section in sections)
    status = "extracted" if sections else "failed"
    warnings = [] if sections else ["No readable paragraph or table text was found in the DOCX document."]
    return ExtractedDocument(
        document_id=document_id,
        sanitized_filename=filename,
        file_type="docx",
        file_size=len(payload),
        extraction_status=status,
        page_count=None,
        sections=sections,
        extracted_text=extracted_text,
        character_count=len(extracted_text),
        warnings=warnings,
        ocr_required=False,
    )
