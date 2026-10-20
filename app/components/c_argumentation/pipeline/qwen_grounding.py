"""Prompt construction and citation validation for C4-grounded Qwen answers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Protocol, Sequence

from .retrieval_repository import RetrievedPassage


class TextGenerator(Protocol):
    def generate(self, prompt: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class CitationValidation:
    valid_source_numbers: tuple[int, ...]
    invalid_source_numbers: tuple[int, ...]
    invalid_case_numbers: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.invalid_source_numbers and not self.invalid_case_numbers


def build_grounded_prompt(
    question: str,
    passages: Sequence[RetrievedPassage],
    *,
    max_chars_per_passage: int = 1400,
) -> str:
    if not question.strip():
        raise ValueError("question must not be empty")
    if not passages:
        evidence = "No passages were retrieved. State that the available evidence is insufficient."
    else:
        blocks = []
        for index, passage in enumerate(passages, start=1):
            page_range = f"{passage.start_page}-{passage.end_page}"
            blocks.append(
                f"[Source {index}] Case: {passage.case_number}; Pages: {page_range}\n"
                f"{passage.chunk_text.strip()[:max_chars_per_passage]}"
            )
        evidence = "\n\n".join(blocks)
    return (
        "Answer the user's question using only the supplied judgment passages. "
        "If they do not establish an answer, say the evidence is insufficient. "
        "Do not invent cases, statutes, legal rules, citations, or page references. "
        "Cite supplied material only with [Source N].\n\n"
        f"Question: {question.strip()}\n\nEvidence:\n{evidence}"
    )


def validate_citations(response: str, passages: Sequence[RetrievedPassage]) -> CitationValidation:
    source_numbers = [int(value) for value in re.findall(r"\[Source\s+(\d+)\]", response, re.IGNORECASE)]
    valid_range = set(range(1, len(passages) + 1))
    invalid_sources = tuple(sorted({value for value in source_numbers if value not in valid_range}))
    valid_sources = tuple(sorted({value for value in source_numbers if value in valid_range}))
    case_numbers = {passage.case_number for passage in passages}
    mentioned_cases = {
        value.rstrip(".,;:)")
        for value in re.findall(r"\bCA/[A-Z0-9()/{}.-]+", response, re.IGNORECASE)
    }
    invalid_cases = tuple(sorted(case for case in mentioned_cases if case not in case_numbers))
    return CitationValidation(valid_sources, invalid_sources, invalid_cases)


def generate_grounded_answer(
    generator: TextGenerator,
    question: str,
    passages: Sequence[RetrievedPassage],
    *,
    max_tokens: int = 256,
) -> tuple[str, CitationValidation]:
    prompt = build_grounded_prompt(question, passages)
    result = generator.generate(
        prompt,
        system_prompt="You are a cautious legal-research assistant. Do not give unsupported conclusions.",
        max_tokens=max_tokens,
        temperature=0.1,
    )
    response = result.text if hasattr(result, "text") else str(result)
    return response, validate_citations(response, passages)
