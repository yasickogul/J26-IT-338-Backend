"""Civil Case Intelligence Report (notebook cells 22 and 23)."""
from .insights import generate_case_insights
from .rag import generate_issue_explanation, heuristic_groundedness_check
from .retrieval import retrieve_legal_evidence, retrieve_similar_cases
from .safety import screen_document_categories

DISCLAIMER = (
    "This report is generated from the uploaded case document and retrieved Sri Lankan legal "
    "sources for general information and decision-support purposes only. It is not formal legal "
    "advice and does not predict the outcome of any case. Please consult a qualified lawyer for "
    "advice on your specific situation."
)


def build_civil_case_intelligence_report(extracted_result, on_progress=None):
    """`on_progress(stage, message, pct)` is optional and only used by the web layer."""
    def progress(stage, message, pct):
        if on_progress:
            on_progress(stage, message, pct)

    categories = extracted_result["categories"]

    progress("screening", "Checking the document content is safe to process", 46)
    screen = screen_document_categories(categories)
    if not screen["allowed"]:
        return {
            "document_id": extracted_result.get("document_id"),
            "blocked": True,
            "reason": screen["reason"],
            "message": "The uploaded document content could not be safely processed.",
        }

    legal_issues = [i.get("text", "") for i in categories.get("Legal Issue", {}).get("items", [])]

    issue_reports = []
    total = len(legal_issues)
    for idx, issue_text in enumerate(legal_issues):
        progress("researching", f"Researching legal issue {idx + 1} of {total}",
                 48 + int(42 * idx / max(total, 1)))
        similar_cases = retrieve_similar_cases(issue_text)
        law_passages = retrieve_legal_evidence(issue_text)
        gen = generate_issue_explanation(issue_text, law_passages)

        grounded_ok, overlap = (None, None)
        if gen["grounded"]:
            grounded_ok, overlap = heuristic_groundedness_check(gen["explanation"], law_passages)

        issue_reports.append({
            "issue": issue_text,
            "explanation": gen["explanation"],
            "groundedness_overlap": overlap,
            "low_confidence": (grounded_ok is False),
            "similar_cases": [
                {
                    "case_title": p["metadata"].get("case_title"),
                    "court": p["metadata"].get("court"),
                    "decided_date": p["metadata"].get("decided_date"),
                    "score": round(p["score"], 3),
                    "excerpt": p["metadata"].get("text", "")[:300],
                }
                for p in similar_cases
            ],
            "relevant_sources": [
                {
                    "label": p["metadata"].get("case_title") or p["metadata"].get("file_name"),
                    "source_type": p["metadata"].get("source_type"),
                    "score": round(p["score"], 3),
                }
                for p in law_passages
            ],
        })

    progress("insights", "Summarising party positions and evidence gaps", 92)
    insights = generate_case_insights(categories)

    return {
        "document_id": extracted_result.get("document_id"),
        "blocked": False,
        "parties": [i.get("text", "") for i in categories.get("Party", {}).get("items", [])],
        "facts": [i.get("text", "") for i in categories.get("Fact", {}).get("items", [])],
        "party_positions": insights.get("party_positions", []),
        "evidence_gaps": insights.get("evidence_gaps", []),
        "legal_issue_reports": issue_reports,
        "disclaimer": DISCLAIMER,
    }


def render_report_markdown(report):
    if report.get("blocked"):
        return f"**This document could not be processed.** Reason: {report['reason']}\n\n{report['message']}"

    lines = []
    lines.append("# Civil Case Intelligence Report")
    lines.append(f"*Document: {report.get('document_id', 'N/A')}*\n")

    lines.append("## Parties")
    for p in report["parties"]:
        lines.append(f"- {p}")
    lines.append("")

    lines.append("## Case Summary — Key Facts")
    for f in report["facts"]:
        lines.append(f"- {f}")
    lines.append("")

    lines.append("## Party Positions")
    for pp in report["party_positions"]:
        lines.append(f"**{pp.get('party', 'Party')}:** {pp.get('position_summary', '')}\n")

    lines.append("## Potential Evidence / Information Gaps")
    if report["evidence_gaps"]:
        for g in report["evidence_gaps"]:
            lines.append(f"- **Claim/Argument:** {g.get('claim_or_argument','')}\n  **Gap noted:** {g.get('issue','')}")
    else:
        lines.append("- No clear evidence gaps were identified based on the extracted information.")
    lines.append("")

    lines.append("## Legal Issues, Relevant Law, Precedents & Explanations")
    for i, ir in enumerate(report["legal_issue_reports"], 1):
        lines.append(f"### Issue {i}: {ir['issue']}")
        lines.append(ir["explanation"])

        if ir["similar_cases"]:
            lines.append("\n**Similar Historical Cases:**")
            for c in ir["similar_cases"]:
                lines.append(f"- {c['case_title']} — {c['court']} ({c['decided_date']}), relevance score {c['score']}")

        if ir["relevant_sources"]:
            lines.append("\n**Sources Consulted:**")
            for s in ir["relevant_sources"]:
                lines.append(f"- [{s['source_type']}] {s['label']} (relevance score {s['score']})")
        lines.append("")

    lines.append("---")
    lines.append(report["disclaimer"])

    return "\n".join(lines)
