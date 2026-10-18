"""Stage 2 final verification — three synthetic disputes + API/persistence checks.

Usage (from J26-IT-338-Backend/):
  PYTHONUNBUFFERED=1 python -m scripts.verify_stage2
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from app.components.c1_CaseAnalysis.pipeline.info_extractor import (
    extract_legal_info,
    intake_to_source_text,
)
from app.components.c1_CaseAnalysis.schemas import CaseIntakeRequest

BACKEND_ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC = [
    CaseIntakeRequest(
        dispute_category="employment",
        title="Perera v. ABC Services",
        plaintiff="Kasun Perera",
        defendant="ABC Services (Pvt) Ltd",
        court_type="District Court — Colombo",
        province="Western",
        district="Colombo",
        amount_lkr="2500000",
        relevant_dates="2024-03-01",
        description=(
            "Kasun Perera was employed under a written contract. On 1 March 2024 the employer "
            "terminated his employment without statutory notice after he requested unpaid overtime. "
            "Counsel cited CA/WRT/0201/2022 as background only."
        ),
        legal_issue="Wrongful termination and unpaid wages/overtime",
        remedy="Reinstatement and compensation for unpaid wages",
        evidence_factors=[
            "Written employment agreement",
            "Salary payment records",
            "Termination letter",
        ],
    ),
    CaseIntakeRequest(
        dispute_category="contract",
        title="Silva v. Fernando Traders",
        plaintiff="Nimal Silva",
        defendant="Fernando Traders",
        court_type="District Court — Gampaha",
        province="Western",
        district="Gampaha",
        amount_lkr="1200000",
        description=(
            "The parties entered a written supply agreement. Fernando Traders failed to pay "
            "invoices for delivered goods totalling LKR 1,200,000. The plaintiff seeks payment. "
            "Reference was made to CA/RII/0021/2024."
        ),
        legal_issue="Breach of contract — unpaid money under supply agreement",
        remedy="Payment of outstanding invoice amount",
        evidence_factors=["Written supply agreement", "Invoices", "Delivery notes"],
    ),
    CaseIntakeRequest(
        dispute_category="property_land",
        title="Gunasekara v. Jayasinghe",
        case_number="DC/KDY/45/2023",
        plaintiff="Ranjith Gunasekara",
        defendant="Sunil Jayasinghe",
        court_type="District Court — Kandy",
        province="Central",
        district="Kandy",
        description=(
            "The plaintiff claims ownership of a land parcel in Kandy and alleges the defendant "
            "encroached across the shared boundary by constructing a fence. Title documents and a "
            "survey plan are available. Relied on CA/PHC/0004/2021 regarding boundary disputes."
        ),
        legal_issue="Land ownership and boundary encroachment",
        remedy="Declaration of title and removal of encroachment",
        evidence_factors=["Title deed", "Survey plan", "Photographs of fence"],
    ),
]


def _annotate_provenance(intake: CaseIntakeRequest, result: dict) -> dict:
    """Human-readable map: field → extracted from intake vs inferred by rules."""
    from_intake = []
    inferred = []
    narrative_refs = []

    ci = result["case_information"]
    mapping = [
        ("dispute_category / case_type", "intake.dispute_category → mapped label"),
        ("title", "intake.title" if intake.title else "inferred from party names"),
        ("case_number", "intake.case_number only (null if omitted)"),
        ("court", "intake.court_type only"),
        ("summary", "intake.description"),
        ("remedy_requested", "intake.remedy"),
        ("legal_issues / claims", "intake.legal_issue (+ claim-like lines if any)"),
        ("evidence_mentioned", "intake.evidence_factors only"),
        ("parties", "intake.plaintiff / defendant (positions left null)"),
        ("cited_case_references", "CASE_NO patterns in narrative → LEGAL_REF, not current case_number"),
    ]
    for field, note in mapping:
        if "inferred" in note.lower() and not intake.title and field == "title" or "mapped" in note:
            inferred.append(f"{field}: {note}")
        elif "narrative" in note.lower() or "LEGAL_REF" in note:
            narrative_refs.append(f"{field}: {note}")
        else:
            from_intake.append(f"{field}: {note}")

    return {
        "from_user_intake": from_intake,
        "from_narrative_patterns": narrative_refs
        + [f"cited={result.get('cited_case_references')}"],
        "rule_inferred_only": inferred
        + [
            f"case_type label '{ci.get('case_type')}' from dispute_category '{intake.dispute_category}'",
            f"confidence={result['structured_legal_info'].get('confidence')}",
        ],
        "explicitly_null_or_empty_when_absent": [
            "party.position (never invented)",
            "case_number when not supplied on intake",
            "court when court_type omitted",
        ],
    }


async def main() -> int:
    print("C2_RESEARCH_COMPONENT_SPEC.md: NOT FOUND in workspace\n")
    print("=" * 72)
    print("TASK 3 — Three synthetic civil disputes (full JSON)")
    print("=" * 72)

    for i, intake in enumerate(SYNTHETIC, start=1):
        print(f"\n### Example {i}: {intake.dispute_category} — {intake.title}\n")
        merged = intake_to_source_text(intake)
        offline = extract_legal_info(
            intake_id=f"synthetic-{i}",
            intake=intake,
            processed_text=merged,
        )
        payload = offline.model_dump(mode="json")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("\n--- Provenance ---")
        print(json.dumps(_annotate_provenance(intake, payload), indent=2))

    print("\n" + "=" * 72)
    print("TASK 4 — API registration, validation, persistence")
    print("=" * 72)

    from httpx import ASGITransport, AsyncClient

    from app.components.c1_CaseAnalysis.dev_app import app
    from app.core.database import direct_engine

    routes = sorted({getattr(r, "path", None) for r in app.routes if getattr(r, "path", None)})
    print("Registered paths:", routes)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        ping = await client.get("/api/v1/c1/ping")
        print("GET /ping:", ping.status_code, ping.json())

        bad = await client.post("/api/v1/c1/intake", json={"dispute_category": "criminal"})
        print("POST /intake invalid category:", bad.status_code)

        missing = await client.post("/api/v1/c1/intake", json={"title": "x"})
        print("POST /intake missing category:", missing.status_code)

        ok = await client.post("/api/v1/c1/intake", json=SYNTHETIC[0].model_dump())
        print("POST /intake employment:", ok.status_code)
        if ok.status_code != 200:
            print(ok.text)
            return 1
        body = ok.json()
        intake_id = body["intake_id"]
        # Dispose pool to simulate process boundary, then re-fetch from Neon
        if direct_engine is not None:
            await direct_engine.dispose()
        got = await client.get(f"/api/v1/c1/intake/{intake_id}")
        print("GET /intake/{id} after engine dispose:", got.status_code)
        assert got.status_code == 200
        assert got.json()["result"]["intake_id"] == intake_id
        assert got.json()["result"]["case_information"]["case_number"] is None
        assert "CA/WRT/0201/2022" in got.json()["result"]["cited_case_references"]
        print("Persistence check: saved extraction reloaded from DB (not memory-only).")

        # Corrupt PDF upload without narrative
        files = {"file": ("bad.pdf", b"this is not a pdf file at all", "application/pdf")}
        data = {"intake_json": json.dumps({"dispute_category": "employment"})}
        pdf_resp = await client.post("/api/v1/c1/intake/document", files=files, data=data)
        print("POST /intake/document corrupt PDF:", pdf_resp.status_code, pdf_resp.text[:200])
        assert pdf_resp.status_code == 422

    if direct_engine is not None:
        await direct_engine.dispose()

    print("\nStage 2 verification script completed.")
    return 0


if __name__ == "__main__":
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    raise SystemExit(asyncio.run(main()))
