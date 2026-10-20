"""Branded PDF export (notebook cell 26). Same layout and styles; returns bytes instead of
writing to /kaggle/working."""
import io
import re
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer,
)

from .report import DISCLAIMER


def _p(text):
    """Escape text for ReportLab's mini-XML and turn **bold** into <b>bold</b>."""
    text = escape(str(text or ""))
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return text.replace("\n", "<br/>")


def export_report_pdf(report):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=20, spaceAfter=4)
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=9,
                                textColor=colors.grey, spaceAfter=10)
    h1 = ParagraphStyle("H1", parent=styles["Heading2"], textColor=colors.HexColor("#1F3A5F"),
                        spaceBefore=14, spaceAfter=6)
    h2 = ParagraphStyle("H2", parent=styles["Heading3"], spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14)
    small = ParagraphStyle("Small", parent=body, fontSize=9, textColor=colors.HexColor("#444444"))
    disclaimer_style = ParagraphStyle("Disclaimer", parent=body, fontSize=8.5,
                                      textColor=colors.HexColor("#555555"))

    def bullets(items, style=body):
        items = [i for i in items if str(i).strip()]
        if not items:
            return [Paragraph("None identified.", small)]
        return [ListFlowable(
            [ListItem(Paragraph(_p(i), style), leftIndent=12) for i in items],
            bulletType="bullet", start="•", leftIndent=12,
        )]

    story = []
    story.append(Paragraph("Civil Case Intelligence Report", title_style))
    story.append(Paragraph(
        f"Document: {_p(report.get('document_id', 'N/A'))} &nbsp;|&nbsp; "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", meta_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#1F3A5F")))

    # Blocked documents
    if report.get("blocked"):
        story.append(Paragraph("This document could not be processed", h1))
        story.append(Paragraph(f"<b>Reason:</b> {_p(report.get('reason'))}", body))
        story.append(Paragraph(_p(report.get("message")), body))
    else:
        story.append(Paragraph("Parties", h1))
        story += bullets(report.get("parties", []))

        story.append(Paragraph("Case Summary - Key Facts", h1))
        story += bullets(report.get("facts", []))

        story.append(Paragraph("Party Positions", h1))
        positions = report.get("party_positions", [])
        if positions:
            for pp in positions:
                story.append(Paragraph(
                    f"<b>{_p(pp.get('party', 'Party'))}:</b> {_p(pp.get('position_summary', ''))}", body))
                story.append(Spacer(1, 4))
        else:
            story.append(Paragraph("None identified.", small))

        story.append(Paragraph("Potential Evidence / Information Gaps", h1))
        gaps = report.get("evidence_gaps", [])
        if gaps:
            story += bullets([
                f"**Claim/Argument:** {g.get('claim_or_argument', '')}  -  "
                f"**Gap noted:** {g.get('issue', '')}" for g in gaps
            ])
        else:
            story.append(Paragraph(
                "No clear evidence gaps were identified based on the extracted information.", small))

        story.append(Paragraph("Legal Issues, Relevant Law, Precedents &amp; Explanations", h1))
        for i, ir in enumerate(report.get("legal_issue_reports", []), 1):
            story.append(Paragraph(f"Issue {i}: {_p(ir['issue'])}", h2))
            story.append(Paragraph(_p(ir["explanation"]), body))

            if ir.get("similar_cases"):
                story.append(Spacer(1, 4))
                story.append(Paragraph("<b>Similar Historical Cases</b>", body))
                story += bullets([
                    f"{c.get('case_title')} - {c.get('court')} ({c.get('decided_date')}), "
                    f"relevance score {c.get('score')}" for c in ir["similar_cases"]
                ], small)

            if ir.get("relevant_sources"):
                story.append(Spacer(1, 4))
                story.append(Paragraph("<b>Sources Consulted</b>", body))
                story += bullets([
                    f"[{s.get('source_type')}] {s.get('label')} (relevance score {s.get('score')})"
                    for s in ir["relevant_sources"]
                ], small)

        story.append(Spacer(1, 14))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        story.append(Spacer(1, 6))
        story.append(Paragraph(_p(report.get("disclaimer", DISCLAIMER)), disclaimer_style))

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(20 * mm, 10 * mm, "Civil Case Intelligence Report - not legal advice")
        canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title="Civil Case Intelligence Report",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
