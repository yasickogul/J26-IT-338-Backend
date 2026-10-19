"""LLM category extraction (notebook cells 7, 8, 9)."""
import difflib
import json
import logging

from .chunking import chunk_document
from .llm import chat
from .utils import strip_json_fences

logger = logging.getLogger(__name__)

ALL_CATEGORIES = [
    "Party", "Fact", "Claim", "Argument", "Evidence",
    "Legal Issue", "Law Reference", "Judicial Precedent", "Decision",
]

CATEGORY_DEFINITIONS = """
- "Party": names and roles of the parties involved (plaintiff, defendant, appellant, respondent, etc.)
- "Fact": undisputed factual background of the case (dates, amounts, events agreed by both sides)
- "Claim": a statement of what a party is asking for or asserting as their position
- "Argument": legal or factual reasoning presented by a party or the court in support of a position
- "Evidence": documents, testimony, or exhibits referenced as supporting a claim or finding
- "Legal Issue": the main legal question(s) the court needed to resolve
- "Law Reference": any Act, Ordinance, statute, section, or legal provision mentioned
- "Judicial Precedent": any previous case cited within the document
- "Decision": the court's finding, ruling, or final order
"""

EXTRACTION_PROMPT = """You are a legal document analysis assistant specializing in Sri Lankan civil case documents.

Extract information from the case text below into the following nine categories. Return ONLY valid JSON — no commentary, no markdown fences, no explanation outside the JSON.

Category definitions:
{definitions}

Here are real examples from Sri Lankan case law showing the expected style and granularity for some categories (for calibration only — do not copy these into your output, only use them as style/format reference):
{fewshot}

Output schema (use this exact structure):
{{
  "Party": [{{"text": "...", "role": "plaintiff|defendant|appellant|respondent|other"}}],
  "Fact": [{{"text": "..."}}],
  "Claim": [{{"text": "...", "party": "plaintiff|defendant|other"}}],
  "Argument": [{{"text": "...", "party": "plaintiff|defendant|court|other"}}],
  "Evidence": [{{"text": "..."}}],
  "Legal Issue": [{{"text": "..."}}],
  "Law Reference": [{{"text": "...", "law": "...", "section": "..."}}],
  "Judicial Precedent": [{{"text": "...", "case_title": "..."}}],
  "Decision": [{{"text": "..."}}]
}}

Rules:
- Use an empty list if a category has no relevant content in this text.
- Do not invent information that is not present in the text.
- Each item's "text" should be a faithful excerpt or close paraphrase of the source, not a summary that loses specific facts (names, dates, amounts, section numbers).
- If this is a partial excerpt of a larger document, still extract whatever is present; do not note that it is partial.

Case text:
\"\"\"
{document_text}
\"\"\"
"""


def _empty_result():
    return {cat: [] for cat in ALL_CATEGORIES}


def extract_llm_categories(document_text, fewshot_block=""):
    """Run the extraction prompt on one chunk. On unparseable output returns an empty result
    carrying `_raw_response` (same as the notebook)."""
    prompt = EXTRACTION_PROMPT.format(
        definitions=CATEGORY_DEFINITIONS,
        fewshot=fewshot_block,
        document_text=document_text,
    )
    raw = strip_json_fences(chat([{"role": "user", "content": prompt}], temperature=0.0))
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("not an object", raw, 0)
        for cat in ALL_CATEGORIES:
            if not isinstance(parsed.get(cat), list):
                parsed[cat] = []
        return parsed
    except json.JSONDecodeError:
        result = _empty_result()
        result["_raw_response"] = raw
        return result


def is_near_duplicate(a, b, threshold=0.85):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def merge_chunk_results(chunk_results):
    """Merge per-chunk results, dropping items whose text is a near-duplicate (>= 0.85)
    of one already kept in the same category."""
    merged = {cat: [] for cat in ALL_CATEGORIES}
    for result in chunk_results:
        for cat in ALL_CATEGORIES:
            for item in result.get(cat, []):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "") or "").strip()
                if not text:
                    continue
                is_dup = any(
                    is_near_duplicate(text, existing.get("text", ""))
                    for existing in merged[cat]
                )
                if not is_dup:
                    merged[cat].append(item)
    return merged


def extract_all_categories(document_text, fewshot_block="", on_chunk=None):
    """Returns (merged_categories, failed_chunk_count, total_chunk_count)."""
    chunks = chunk_document(document_text)
    chunk_results = []
    failed = 0
    for i, c in enumerate(chunks, start=1):
        if on_chunk:
            on_chunk(i, len(chunks))
        res = extract_llm_categories(c, fewshot_block)
        if "_raw_response" in res:
            failed += 1
            logger.warning("Chunk %d/%d: model output was not valid JSON", i, len(chunks))
        chunk_results.append(res)
    return merge_chunk_results(chunk_results), failed, len(chunks)


def build_unified_output(document_id, extracted):
    return {
        "document_id": document_id,
        "categories": {
            cat: {"items": extracted.get(cat, []), "source": "llm"}
            for cat in ALL_CATEGORIES
        },
    }
