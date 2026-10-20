from io import BytesIO
import base64
from zipfile import ZipFile

import pymupdf
from fastapi.testclient import TestClient

from app.components.c_argumentation.dev_app import app
from app.components.c_argumentation import router
from app.components.c_argumentation.schemas_multi_agent import MultiAgentAssessment


def pdf_bytes(*pages: str) -> bytes:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text)
    payload = document.tobytes()
    document.close()
    return payload


def docx_bytes(paragraphs: list[str], rows: list[list[str]]) -> bytes:
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    body += "<w:tbl>" + "".join(
        "<w:tr>" + "".join(f"<w:tc><w:p><w:r><w:t>{cell}</w:t></w:r></w:p></w:tc>" for cell in row) + "</w:tr>"
        for row in rows
    ) + "</w:tbl>"
    document_xml = f'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{body}<w:sectPr/></w:body></w:document>'
    content_types = '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("word/document.xml", document_xml)
    return output.getvalue()


def upload(filename: str, payload: bytes):
    return TestClient(app).post(
        "/api/v1/c_argumentation/documents/extract",
        files={"document": (filename, payload, "application/octet-stream")},
    )


def test_valid_multipage_pdf_preserves_pages_and_uses_temporary_id():
    response = upload("../land-case.pdf", pdf_bytes("First page evidence", "Second page evidence"))
    data = response.json()
    assert response.status_code == 200
    assert data["extraction_status"] == "extracted"
    assert data["page_count"] == 2
    assert [section["label"] for section in data["sections"]] == ["page-1", "page-2"]
    assert "[Page 1]" in data["extracted_text"] and "Second page evidence" in data["extracted_text"]
    assert len(data["document_id"]) == 36
    assert data["sanitized_filename"] == "land-case.pdf"
    assert data["content_status"] == "unverified_extracted_text"


def test_valid_docx_extracts_paragraphs_and_tables_without_persistent_storage():
    response = upload("evidence.docx", docx_bytes(["Heading", "A legal fact supplied by the user."], [["Date", "Value"]]))
    data = response.json()
    assert response.status_code == 200
    assert data["file_type"] == "docx"
    assert [section["label"] for section in data["sections"]] == ["paragraph-1", "paragraph-2", "table-1"]
    assert "Date\tValue" in data["extracted_text"]
    assert "storage" not in data


def test_empty_file_is_rejected():
    response = upload("empty.pdf", b"")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_unsupported_extension_is_rejected():
    response = upload("evidence.txt", b"plain text")
    assert response.status_code == 415


def test_invalid_pdf_signature_is_rejected():
    response = upload("evidence.pdf", b"not a pdf")
    assert response.status_code == 400
    assert "signature" in response.json()["detail"].lower()


def test_oversized_upload_is_rejected():
    response = upload("large.pdf", b"%PDF-" + b"x" * (10 * 1024 * 1024))
    assert response.status_code == 413


def test_scanned_pdf_is_marked_for_ocr():
    document = pymupdf.open()
    page = document.new_page()
    one_pixel_png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
    page.insert_image(pymupdf.Rect(72, 72, 100, 100), stream=one_pixel_png)
    image_pdf = document.tobytes()
    document.close()
    response = upload("scanned.pdf", image_pdf)
    assert response.status_code == 200
    assert response.json()["ocr_required"] is True
    assert response.json()["extraction_status"] == "ocr_required"


def test_existing_assessment_route_remains_registered(monkeypatch):
    async def fake_assess(_case):
        return MultiAgentAssessment(
            case_summary="Compatibility check",
            stated_facts="The existing assessment contract remains available.",
            plaintiff_arguments={"arguments": []},
            defense_counterarguments={"counterarguments": []},
            evidence_sufficiency="insufficient",
            neutral_case_assessment={"assessment": "Insufficient.", "evidence_sufficiency": "insufficient"},
        )

    monkeypatch.setattr(router, "assess_case", fake_assess)
    response = TestClient(app).post(
        "/api/v1/c_argumentation/assess",
        json={
            "case_title": "Property dispute",
            "dispute_type": "Property ownership",
            "case_facts": "A deed is disputed and possession is contested by both parties.",
            "user_position": "The user claims ownership.",
            "opposing_position": "The other party disputes the deed.",
        },
    )
    assert response.status_code == 200
