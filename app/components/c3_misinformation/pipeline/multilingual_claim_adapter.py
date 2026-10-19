"""Optional multilingual claim handling for English-language legal retrieval."""
from __future__ import annotations

import asyncio
import os
import threading

from .claim_translation import TranslationProvider, TranslationProviderError, translate_claim
from .language_detection import LanguageIdentifier, detect_language
from ..prompts.rag_prompt import ClaimInput, Input


def _language_detection_enabled() -> bool:
    value = os.getenv("COMPNET_LANGUAGE_DETECTION_ENABLED", "false").strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise RuntimeError("COMPNET_LANGUAGE_DETECTION_ENABLED must be a boolean value.")


class MultilingualClaimAdapter:
    """Adapt claim text to an English retrieval query without replacing the source claim."""

    def __init__(
        self,
        *,
        language_identifier: LanguageIdentifier | None = None,
        translation_provider: TranslationProvider | None = None,
        detection_enabled: bool | None = None,
    ) -> None:
        if detection_enabled is None:
            detection_enabled = _language_detection_enabled()
        if not isinstance(detection_enabled, bool):
            raise TypeError("detection_enabled must be a boolean or None")
        self._language_identifier = language_identifier
        self._translation_provider = translation_provider
        self._detection_enabled = detection_enabled

    def prepare_input(self, input_data: Input) -> Input:
        """Return the original claim with an English retrieval query where applicable."""
        if not isinstance(input_data, ClaimInput):
            return input_data
        if input_data.retrieval_query is not None:
            return input_data

        claim_text = input_data.claim
        if not isinstance(claim_text, str):
            raise TypeError("claim text must be a string")
        if not claim_text.strip():
            return ClaimInput(claim=claim_text)

        detected_language = self._detect_language(claim_text)
        if detected_language in {"si", "ta"} and self._translation_provider is not None:
            try:
                translation = translate_claim(
                    claim_text,
                    detected_language,
                    provider=self._translation_provider,
                )
            except Exception:
                # Fallback belongs at this integration boundary. Exception
                # excludes asyncio cancellation; untranslated retrieval does
                # not guarantee equivalent coverage of the English corpus.
                return ClaimInput(claim=claim_text, retrieval_query=claim_text)
            if (
                translation.status == "success"
                and isinstance(translation.translated_text, str)
                and translation.translated_text.strip()
            ):
                return ClaimInput(
                    claim=claim_text,
                    retrieval_query=translation.translated_text,
                )

        return ClaimInput(claim=claim_text, retrieval_query=claim_text)

    async def aprepare_input(self, input_data: Input) -> Input:
        """Prepare input without allowing a synchronous provider to block asyncio.

        Translation providers are intentionally dependency-injected and may be
        synchronous (the existing Gemini provider is one).  Running the
        compatibility method in a worker thread keeps network/model work out of
        the event loop and preserves cancellation of the awaiting RAG task.
        """
        asyncio.get_running_loop()
        value: Input | None = None
        error: BaseException | None = None
        finished = threading.Event()

        def run() -> None:
            nonlocal value, error
            try:
                value = self.prepare_input(input_data)
            except BaseException as exc:
                error = exc
            finally:
                finished.set()

        thread = threading.Thread(
            target=run, name="c3-multilingual-adapter", daemon=True,
        )
        thread.start()
        try:
            while not finished.is_set():
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            raise
        thread.join(timeout=0.1)
        if error is not None:
            raise error
        assert value is not None
        return value

    def _detect_language(self, text: str) -> str | None:
        if not self._detection_enabled:
            return None
        try:
            result = detect_language(text, language_identifier=self._language_identifier)
        except Exception:
            return None
        if result.status == "detected" and result.label in {"en", "si", "ta", "mixed"}:
            return result.label
        return None


__all__ = ["MultilingualClaimAdapter"]
