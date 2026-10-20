"""Grounded legal explanation (notebook cell 20)."""
import difflib

from .llm import chat
from .retrieval import format_context_block

RAG_SYSTEM_PROMPT = """You are a legal information assistant for a Sri Lankan civil-law decision-support tool. Follow these rules strictly:

1. Answer ONLY using the numbered sources provided in the context. Do not use outside knowledge, even if you believe it to be correct.
2. Every substantive legal statement must be followed by a citation to the source number(s) it came from, e.g. [1], [2].
3. If the sources do not contain enough information, say so explicitly rather than filling the gap yourself.
4. NEVER predict the outcome of the case, state who will "win" or "lose", or give a probability of success. Explain the relevant law and precedent only.
5. NEVER state or imply this is formal legal advice.
6. Write in plain, clear language a non-lawyer can understand, while remaining accurate to the sources.
7. Do not fabricate case names, statute sections, or citations not present in the sources.
"""


def generate_issue_explanation(issue_text, law_passages):
    if not law_passages:
        return {
            "explanation": (
                "No sufficiently relevant legal material was found in the knowledge base for this "
                "issue. This may fall outside the current knowledge base's coverage."
            ),
            "grounded": False,
        }
    context_block = format_context_block(law_passages)
    user_prompt = f"""Context sources:
{context_block}

Legal issue identified in the case under review:
{issue_text}

Explain, in plain language, what the relevant law and precedent say about this issue, using only the context sources above."""

    explanation = chat(
        [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return {"explanation": explanation, "grounded": True}


def heuristic_groundedness_check(answer_text, passages, min_overlap_ratio=0.15):
    context_text = " ".join(p["metadata"].get("text", "") for p in passages).lower()
    overlap = difflib.SequenceMatcher(None, answer_text.lower(), context_text).ratio()
    return overlap >= min_overlap_ratio, round(overlap, 3)
