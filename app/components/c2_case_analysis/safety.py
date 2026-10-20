"""Input screening (notebook cell 18). The source document is untrusted input, so every
extracted text field is screened for prompt-injection phrases before it reaches the LLM."""
import re

INJECTION_PATTERNS = [
    r"ignore (all|any|previous|above) instructions",
    r"disregard (all|any|previous|above) instructions",
    r"you are now",
    r"pretend (you|to) (are|be)",
    r"act as (an? )?(unfiltered|unrestricted|jailbroken)",
    r"reveal your (prompt|instructions|system prompt)",
]

OUTCOME_PREDICTION_PATTERNS = [
    r"will i win", r"will i lose", r"am i going to win", r"what will the court decide",
    r"predict the outcome", r"what is the verdict", r"will the judge rule in my favou?r",
    r"chances of winning", r"probability (of|that) (i|we) (win|lose)",
]


def screen_text(text):
    t = text.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, t):
            return {"allowed": False, "reason": "prompt_injection"}
    flags = []
    if any(re.search(p, t) for p in OUTCOME_PREDICTION_PATTERNS):
        flags.append("outcome_prediction")
    return {"allowed": True, "flags": flags}


def screen_document_categories(extracted_categories):
    """Screen every extracted text field before it reaches the LLM, since the source
    document is untrusted input (could itself contain an injection attempt)."""
    for cat, payload in extracted_categories.items():
        for item in payload.get("items", []):
            check = screen_text(item.get("text", ""))
            if not check["allowed"]:
                return check
    return {"allowed": True, "flags": []}
