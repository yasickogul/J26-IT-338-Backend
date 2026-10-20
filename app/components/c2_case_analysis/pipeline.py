"""End-to-end pipeline (notebook cells 9, 24): extract text -> LLM categories -> report."""
from .extraction import build_unified_output, extract_all_categories
from .fewshot import build_fewshot_block
from .ocr import extract_document_text
from .report import build_civil_case_intelligence_report, render_report_markdown


class DocumentError(Exception):
    """A problem with the uploaded document itself (shown to the user as-is)."""


def run_analysis(file_path, document_id, progress, on_extraction=None):
    """`progress(stage, message, pct)` reports progress; `on_extraction(unified)` is called as
    soon as the nine categories are available. Returns a dict with extraction/report/markdown."""
    progress("reading", "Reading the document", 2)
    text = extract_document_text(
        file_path,
        on_page=lambda i, n: progress("reading", f"Reading page {i} of {n}", 2 + int(12 * (i - 1) / max(n, 1))),
    )
    if not text or not text.strip():
        raise DocumentError(
            "No text could be read from this file. Try a clearer scan or a text-based PDF."
        )

    progress("extracting", "Extracting case details", 15)
    fewshot_block = build_fewshot_block()
    extracted, failed, total = extract_all_categories(
        text, fewshot_block,
        on_chunk=lambda i, n: progress("extracting", f"Extracting case details (part {i} of {n})",
                                       15 + int(30 * (i - 1) / max(n, 1))),
    )
    warnings = []
    if failed:
        warnings.append(
            f"{failed} of {total} sections of the document could not be interpreted and were skipped, "
            "so the extracted details may be incomplete."
        )

    unified = build_unified_output(document_id, extracted)
    if on_extraction:
        on_extraction(unified)

    report = build_civil_case_intelligence_report(unified, on_progress=progress)
    markdown = render_report_markdown(report)
    return {"extraction": unified, "report": report, "markdown": markdown, "warnings": warnings}
