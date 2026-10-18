"""Lazy, local-only adapter for the FastText language-ID model."""

from __future__ import annotations

import math
import os
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal

from app.components.c3_misinformation.pipeline.language_detection import (
    LanguageIdentification,
    LanguageIdentifierError,
    LanguageIdentifierUnavailableError,
)


MODEL_PATH_ENV = "COMPNET_FASTTEXT_MODEL_PATH"
_SUPPORTED_LABELS: dict[str, Literal["en", "si", "ta"]] = {
    "en": "en",
    "si": "si",
    "ta": "ta",
}


class FastTextLanguageIdentifier:
    """Load one local FastText model and reuse it for subsequent predictions.

    The adapter never downloads model files. FastText scores are retained as
    uncalibrated diagnostics; they are not used as decision thresholds here.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        model_loader: Callable[[str], Any] | None = None,
        model_name: str | None = None,
    ) -> None:
        configured_path = model_path or os.environ.get(MODEL_PATH_ENV)
        self._model_path = Path(configured_path).expanduser() if configured_path else None
        self._model_loader = model_loader
        self._model_name = model_name
        self._model: Any | None = None
        self._load_lock = Lock()

    def identify(self, text: str) -> LanguageIdentification:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise LanguageIdentifierError("FastText cannot classify empty text.")

        model = self._get_model()
        try:
            labels, scores = self._predict(model, text)
        except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
            raise LanguageIdentifierError(
                f"FastText prediction failed: {type(exc).__name__}."
            ) from exc

        if len(labels) != 1 or len(scores) != 1:
            raise LanguageIdentifierError("FastText returned malformed prediction output.")
        if not isinstance(labels[0], str):
            raise LanguageIdentifierError("FastText returned a non-text language label.")
        raw_label = labels[0].removeprefix("__label__")
        score = scores[0]
        if not raw_label or not math.isfinite(score):
            raise LanguageIdentifierError("FastText returned invalid prediction output.")

        label = _SUPPORTED_LABELS.get(raw_label, "unsupported")

        return LanguageIdentification(
            label=label,
            model_name=self._model_name or f"fasttext:{self._model_path.name}",
            score=score,
        )

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            if self._model_path is None:
                raise LanguageIdentifierUnavailableError(
                    f"Set {MODEL_PATH_ENV} or provide a local model_path."
                )
            if not self._model_path.is_file():
                raise LanguageIdentifierUnavailableError(
                    f"FastText model file does not exist: {self._model_path}."
                )
            loader = self._model_loader or self._default_model_loader
            try:
                self._model = loader(str(self._model_path))
            except LanguageIdentifierUnavailableError:
                raise
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                raise LanguageIdentifierError(
                    f"FastText model could not be loaded: {type(exc).__name__}."
                ) from exc
            if self._model is None:
                raise LanguageIdentifierError("FastText model loader returned no model.")
            return self._model

    @staticmethod
    def _default_model_loader(model_path: str) -> Any:
        try:
            import fasttext
        except ImportError as exc:
            raise LanguageIdentifierUnavailableError(
                "FastText Python bindings are unavailable."
            ) from exc
        return fasttext.load_model(model_path)

    @staticmethod
    def _predict(model: Any, text: str) -> tuple[list[str], list[float]]:
        model_text = text.replace("\n", " ")
        native_model = getattr(model, "f", None)
        if native_model is not None:
            # fasttext-wheel 0.9.2's wrapper uses np.array(copy=False), which
            # fails with NumPy 2; its native binding avoids that conversion.
            pairs = list(native_model.predict(model_text + "\n", 1, 0.0, "strict"))
            labels = [label for _, label in pairs]
            scores = [float(score) for score, _ in pairs]
            return labels, scores

        labels, scores = model.predict(model_text, k=1)
        return list(labels), [float(score) for score in scores]
