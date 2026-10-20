"""Conservative language detection for FR-05-cleaned claim text.

Native Sinhala/Tamil script can be identified without a model. Latin-script
and mixed-script text require an injected language identifier; script presence
alone is not sufficient evidence of English or a meaningful language mixture.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, Protocol

_MIN_MIXED_SCRIPT_WORDS = 2


LanguageLabel = Literal["en", "si", "ta", "mixed"]
ModelLabel = Literal["en", "si", "ta", "mixed", "unsupported", "uncertain"]
DetectionStatus = Literal[
    "detected", "uncertain", "unsupported", "invalid", "detector_error",
]
DetectionReason = Literal[
    "native_script", "statistical_id", "mixed_language_evidence",
    "insufficient_language_evidence", "model_unavailable",
    "unsupported_language", "unsupported_or_unrecognized_script",
    "unsupported_language_candidate", "insufficient_mixed_language_evidence",
    "romanized_language_uncertain", "language_evidence_conflict",
    "invalid_input", "model_error",
]


class LanguageIdentifierUnavailableError(RuntimeError):
    """Raised by a configured language identifier when its model is unavailable."""


class LanguageIdentifierError(RuntimeError):
    """Raised when a configured language identifier cannot classify text."""


@dataclass(frozen=True)
class LanguageIdentification:
    """A language-ID decision; score is an uncalibrated model output."""

    label: ModelLabel
    model_name: str
    score: float | None = None


class LanguageIdentifier(Protocol):
    def identify(self, text: str) -> LanguageIdentification:
        """Classify text or explicitly report uncertainty/unsupported language."""


@dataclass(frozen=True)
class LanguageDetectionResult:
    """Detection outcome; model_score is diagnostic and uncalibrated."""

    label: LanguageLabel | None
    status: DetectionStatus
    reason: DetectionReason
    evidence_languages: tuple[str, ...] = ()
    model_name: str | None = None
    model_score: float | None = None


def _letter_scripts(text: str) -> set[str]:
    scripts: set[str] = set()
    for char in text:
        if not unicodedata.category(char).startswith("L"):
            continue
        name = unicodedata.name(char, "")
        if name.startswith("SINHALA "):
            scripts.add("si")
        elif name.startswith("TAMIL "):
            scripts.add("ta")
        elif name.startswith("LATIN "):
            scripts.add("latin")
        else:
            scripts.add("other")
    return scripts


def _word_tokens(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    for token in re.findall(r"[^\W\d_]+", text, flags=re.UNICODE):
        scripts = _letter_scripts(token)
        if len(scripts) == 1:
            tokens.append((next(iter(scripts)), token))
    return tokens


def _tokens_for(tokens: list[tuple[str, str]], script: str) -> list[str]:
    return [token for token_script, token in tokens if token_script == script]


def _english_segments(tokens: list[tuple[str, str]]) -> list[str]:
    segments: list[str] = []
    current: list[str] = []
    for script, token in tokens:
        if script == "latin":
            current.append(token)
        elif current:
            if len(current) >= _MIN_MIXED_SCRIPT_WORDS:
                segments.append(" ".join(current))
            current = []
    if len(current) >= _MIN_MIXED_SCRIPT_WORDS:
        segments.append(" ".join(current))
    return segments


def _uncertain(
    reason: DetectionReason,
    *,
    evidence_languages: tuple[str, ...] = (),
    model_name: str | None = None,
    model_score: float | None = None,
) -> LanguageDetectionResult:
    return LanguageDetectionResult(
        label=None,
        status="uncertain",
        reason=reason,
        evidence_languages=evidence_languages,
        model_name=model_name,
        model_score=model_score,
    )


def _identification_result(
    label: LanguageLabel,
    reason: DetectionReason,
    identification: LanguageIdentification,
    *,
    evidence_languages: tuple[str, ...] | None = None,
) -> LanguageDetectionResult:
    return LanguageDetectionResult(
        label=label,
        status="detected",
        reason=reason,
        evidence_languages=(
            evidence_languages
            if evidence_languages is not None
            else (() if label == "mixed" else (label,))
        ),
        model_name=identification.model_name,
        model_score=identification.score,
    )


def _validate_identification(
    identification: LanguageIdentification,
) -> LanguageIdentification:
    if not isinstance(identification, LanguageIdentification):
        raise TypeError("language_identifier must return LanguageIdentification")
    if identification.label not in {
        "en", "si", "ta", "mixed", "unsupported", "uncertain",
    }:
        raise ValueError("language_identifier returned an unsupported label")
    if not isinstance(identification.model_name, str) or not identification.model_name.strip():
        raise ValueError("language_identifier must return a nonblank model_name")
    if identification.score is not None and (
        isinstance(identification.score, bool)
        or not isinstance(identification.score, (int, float))
        or not math.isfinite(identification.score)
    ):
        raise ValueError("language_identifier score must be a finite number or None")
    return identification


def detect_language(
    cleaned_text: str,
    *,
    language_identifier: LanguageIdentifier | None = None,
) -> LanguageDetectionResult:
    """Identify language from cleaned text without modifying the input.

    The optional identifier is called for Latin-script or conflicting /
    unsupported script input. It must return a supported class or explicit
    uncertainty; model scores are diagnostic only and are not thresholded.
    """
    if not isinstance(cleaned_text, str):
        raise TypeError("cleaned_text must be a string")
    if not cleaned_text.strip():
        return LanguageDetectionResult(
            label=None, status="invalid", reason="invalid_input",
        )

    scripts = _letter_scripts(cleaned_text)
    tokens = _word_tokens(cleaned_text)
    native = tuple(language for language in ("si", "ta") if language in scripts)
    native_tokens = {
        language: _tokens_for(tokens, language) for language in ("si", "ta")
    }
    latin_tokens = _tokens_for(tokens, "latin")

    if not tokens:
        return _uncertain("insufficient_language_evidence")

    if native and "other" in scripts:
        return _uncertain(
            "unsupported_or_unrecognized_script",
            evidence_languages=native,
        )

    if len(native) == 1 and "latin" not in scripts and "other" not in scripts:
        language = native[0]
        return LanguageDetectionResult(
            label=language, status="detected", reason="native_script",
            evidence_languages=(language,),
        )

    if len(native) == 2 and "other" not in scripts:
        if all(
            len(native_tokens[language]) >= _MIN_MIXED_SCRIPT_WORDS
            for language in native
        ):
            return LanguageDetectionResult(
                label="mixed", status="detected",
                reason="mixed_language_evidence",
                evidence_languages=native,
            )
        return _uncertain(
            "insufficient_mixed_language_evidence",
            evidence_languages=native,
        )

    if native and "latin" in scripts and (
        len(native_tokens[native[0]]) < _MIN_MIXED_SCRIPT_WORDS
        or len(latin_tokens) < _MIN_MIXED_SCRIPT_WORDS
    ):
        # This is a structural minimum, not a model-confidence threshold.
        return _uncertain(
            "insufficient_mixed_language_evidence",
            evidence_languages=native,
        )

    if language_identifier is None:
        reason: DetectionReason = (
            "unsupported_or_unrecognized_script"
            if "other" in scripts
            else "language_evidence_conflict"
            if native
            else "model_unavailable"
        )
        return _uncertain(reason, evidence_languages=native)

    try:
        identification = language_identifier.identify(cleaned_text)
    except LanguageIdentifierUnavailableError:
        return _uncertain(
            "model_unavailable", evidence_languages=native,
        )
    except LanguageIdentifierError:
        return LanguageDetectionResult(
            label=None, status="detector_error", reason="model_error",
            evidence_languages=native,
        )
    except Exception:  # noqa: BLE001 - identifier boundary must be contained
        return LanguageDetectionResult(
            label=None, status="detector_error", reason="model_error",
            evidence_languages=native,
        )

    try:
        identification = _validate_identification(identification)
    except (TypeError, ValueError):
        return LanguageDetectionResult(
            label=None,
            status="detector_error",
            reason="model_error",
            evidence_languages=native,
        )

    model_label = identification.label
    if model_label == "unsupported":
        if not native and "latin" not in scripts and "other" in scripts:
            return LanguageDetectionResult(
                label=None, status="unsupported", reason="unsupported_language",
                evidence_languages=(), model_name=identification.model_name,
                model_score=identification.score,
            )
        return _uncertain(
            "unsupported_language_candidate",
            evidence_languages=native,
            model_name=identification.model_name,
            model_score=identification.score,
        )
    if model_label == "uncertain":
        return _uncertain(
            "language_evidence_conflict" if native else "insufficient_language_evidence",
            evidence_languages=native, model_name=identification.model_name,
            model_score=identification.score,
        )

    if native:
        if "latin" in scripts:
            if not latin_tokens or not native_tokens[native[0]]:
                return _uncertain(
                    "insufficient_mixed_language_evidence",
                    evidence_languages=native,
                    model_name=identification.model_name,
                    model_score=identification.score,
                )
            latin_identifications: list[LanguageIdentification] = []
            try:
                for segment in _english_segments(tokens):
                    latin_identifications.append(
                        _validate_identification(language_identifier.identify(segment))
                    )
            except LanguageIdentifierUnavailableError:
                return _uncertain(
                    "model_unavailable", evidence_languages=native,
                )
            except LanguageIdentifierError:
                return LanguageDetectionResult(
                    label=None, status="detector_error", reason="model_error",
                    evidence_languages=native,
                )
            except Exception:  # noqa: BLE001 - identifier boundary must be contained
                return LanguageDetectionResult(
                    label=None,
                    status="detector_error",
                    reason="model_error",
                    evidence_languages=native,
                )
            if any(item.label == "en" for item in latin_identifications):
                return _identification_result(
                    "mixed", "mixed_language_evidence", identification,
                    evidence_languages=(*native, "en"),
                )
            return _uncertain(
                "language_evidence_conflict",
                evidence_languages=tuple(dict.fromkeys((*native, model_label))),
                model_name=identification.model_name,
                model_score=identification.score,
            )

        if "other" in scripts:
            if model_label in native:
                return _uncertain(
                    "unsupported_or_unrecognized_script",
                    evidence_languages=native,
                    model_name=identification.model_name,
                    model_score=identification.score,
                )
            return _uncertain(
                "language_evidence_conflict",
                evidence_languages=tuple(dict.fromkeys((*native, model_label))),
                model_name=identification.model_name,
                model_score=identification.score,
            )

        if model_label in native:
            return _identification_result(
                model_label, "statistical_id", identification,
                evidence_languages=native,
            )
        return _uncertain(
            "language_evidence_conflict",
            evidence_languages=tuple(dict.fromkeys((*native, model_label))),
            model_name=identification.model_name,
            model_score=identification.score,
        )

    if "other" in scripts:
        return _uncertain(
            "unsupported_or_unrecognized_script",
            evidence_languages=(model_label,),
            model_name=identification.model_name,
            model_score=identification.score,
        )

    if model_label in ("si", "ta") and "latin" in scripts:
        return _uncertain(
            "romanized_language_uncertain",
            evidence_languages=(model_label,),
            model_name=identification.model_name,
            model_score=identification.score,
        )

    if model_label == "mixed":
        return _uncertain(
            "insufficient_mixed_language_evidence",
            model_name=identification.model_name,
            model_score=identification.score,
        )

    if model_label == "en" and len(latin_tokens) < _MIN_MIXED_SCRIPT_WORDS:
        return _uncertain(
            "insufficient_language_evidence",
            model_name=identification.model_name,
            model_score=identification.score,
        )

    return _identification_result(model_label, "statistical_id", identification)
