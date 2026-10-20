"""Party positions + evidence-gap analysis (notebook cell 21)."""
import json

from .llm import chat
from .utils import strip_json_fences

INSIGHT_SYSTEM_PROMPT = """You are a legal document analysis assistant. You will be given structured
information already extracted from a Sri Lankan civil case document: parties, facts, claims,
arguments, evidence, and legal issues.

Your task is to:
1. Summarize each party's position in plain language, based only on their claims and arguments.
2. For each claim or argument, check whether it is supported by at least one item in the evidence
   list. Flag claims/arguments that appear to lack clearly corresponding evidence.
3. Note this is about information gaps, not a judgment on who is right.

Rules:
- Use ONLY the structured information provided below. Do not invent facts, documents, or evidence
  that is not listed.
- NEVER predict who will win the case or how a court would likely rule.
- NEVER state this is formal legal advice.
- Return ONLY valid JSON matching the schema given, no commentary outside the JSON.
"""

INSIGHT_OUTPUT_SCHEMA = """{
  "party_positions": [
    {"party": "...", "position_summary": "..."}
  ],
  "evidence_gaps": [
    {"claim_or_argument": "...", "issue": "no corresponding evidence found in the extracted evidence list"}
  ]
}"""


def generate_case_insights(extracted_categories):
    def items_text(cat):
        return [i.get("text", "") for i in extracted_categories.get(cat, {}).get("items", [])]

    payload = {
        "Party": items_text("Party"),
        "Fact": items_text("Fact"),
        "Claim": items_text("Claim"),
        "Argument": items_text("Argument"),
        "Evidence": items_text("Evidence"),
    }

    user_prompt = f"""Extracted case information:
{json.dumps(payload, indent=2)}

Output schema:
{INSIGHT_OUTPUT_SCHEMA}
"""
    raw = strip_json_fences(chat(
        [
            {"role": "system", "content": INSIGHT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    ))
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("not an object", raw, 0)
        return parsed
    except json.JSONDecodeError:
        return {"party_positions": [], "evidence_gaps": [], "_raw_response": raw}
