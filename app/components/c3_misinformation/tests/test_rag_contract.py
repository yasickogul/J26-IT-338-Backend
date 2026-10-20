"""Offline coverage of the original RAG data and exception contracts."""

from dataclasses import FrozenInstanceError, MISSING, asdict, fields
from typing import Any, Literal, get_type_hints
import unittest

from app.components.c3_misinformation import rag_contract as contract
from app.components.c3_misinformation.claim_lifecycle_contract import FAILURE_CATEGORIES


def result_values():
    return dict(
        mode="claim", evidence_packet={"evidence": []},
        response={"claim_verdict": None, "insufficient_evidence": True},
        response_origin="application", validation={"status": "passed"},
        model=None, usage=None, attempts=0, finish_reason=None,
        limitations=["Insufficient evidence"],
        transaction_status={"read_only_requested": True, "rolled_back": True, "closed": True},
    )


class RagContractTests(unittest.TestCase):
    def test_result_preserves_claim_and_question_values(self):
        for mode in ("claim", "question"):
            with self.subTest(mode=mode):
                values = result_values()
                values["mode"] = mode
                if mode == "question":
                    values.update(response={"answer": "Fixture answer"}, response_origin="model",
                                  model="fixture", usage={"tokens": 1}, attempts=1, finish_reason="STOP")
                result = contract.RagResult(**values)
                self.assertEqual(asdict(result), values)
                self.assertIs(result.evidence_packet, values["evidence_packet"])

    def test_prepared_preserves_fields_and_nested_objects(self):
        packet, prompt, schema = {}, {"input": "text"}, {"type": "object"}
        prepared = contract.RagPrepared("claim", packet, prompt, schema, {"closed": True})
        self.assertEqual(asdict(prepared), {
            "mode": "claim", "evidence_packet": {}, "prompt": prompt,
            "schema": schema, "transaction_status": {"closed": True},
        })
        self.assertIs(prepared.prompt, prompt)
        with self.assertRaises(FrozenInstanceError):
            prepared.mode = "question"

    def test_nullable_fields_are_required_and_dataclasses_are_shallow_frozen(self):
        for cls in (contract.RagResult, contract.RagPrepared):
            self.assertTrue(all(f.default is MISSING and f.default_factory is MISSING for f in fields(cls)))
        values = result_values()
        result = contract.RagResult(**values)
        with self.assertRaises(FrozenInstanceError):
            result.attempts = 2
        result.response["extra"] = "preserved"
        self.assertEqual(values["response"]["extra"], "preserved")
        del values["model"]
        with self.assertRaises(TypeError):
            contract.RagResult(**values)

    def test_annotations_and_lack_of_runtime_dataclass_validation(self):
        hints = get_type_hints(contract.RagResult)
        self.assertEqual(hints["mode"], Literal["question", "claim"])
        self.assertEqual(hints["usage"], dict[str, Any] | None)
        self.assertEqual(hints["transaction_status"], dict[str, bool])
        values = result_values()
        values.update(mode="unknown", attempts=-1)
        self.assertEqual(contract.RagResult(**values).mode, "unknown")

    def test_error_inheritance_message_category_and_attempt_defaults(self):
        classes = (
            contract.RagServiceError, contract.RagInputError, contract.RagRetrievalError,
            contract.RagPreparationError, contract.RagProviderError, contract.RagResponseValidationError,
        )
        for cls in classes:
            with self.subTest(cls=cls.__name__):
                error = cls("fixture message", category="provider_error")
                self.assertIsInstance(error, RuntimeError)
                self.assertIsInstance(error, contract.RagServiceError)
                self.assertEqual(error.args, ("fixture message",))
                self.assertEqual(str(error), "fixture message")
                self.assertEqual(error.category, "provider_error")
                self.assertEqual(error.attempts, 0)
                self.assertEqual(cls("message", category="custom", attempts=3).attempts, 3)
        self.assertIsInstance(contract.RagInputError("input", category="input"), ValueError)

    def test_token_limit_error_preserves_exact_message_and_attributes(self):
        error = contract.RagInputTooLongError(token_count=513, limit=512)
        self.assertIsInstance(error, contract.RagInputError)
        self.assertIsInstance(error, ValueError)
        self.assertEqual((error.token_count, error.limit, error.attempts), (513, 512, 0))
        self.assertEqual(error.category, "input_too_long")
        self.assertEqual(str(error),
                         "The question or claim exceeds the LaBSE token limit "
                         "(513 tokens; limit 512). Please shorten the question or claim.")

    def test_request_budget_error_preserves_exact_message_and_attributes(self):
        error = contract.RagRequestTooLargeError(
            estimated_input_tokens=17000, project_budget=16384,
            model_input_limit=100000, output_allowance=2048,
        )
        self.assertEqual((error.estimated_input_tokens, error.project_budget,
                          error.model_input_limit, error.output_allowance), (17000, 16384, 100000, 2048))
        self.assertEqual(error.category, "request_too_large")
        self.assertEqual(error.attempts, 0)
        self.assertEqual(str(error),
                         "The Gemini request is too large for the configured request budget "
                         "(estimated input 17000 tokens; project budget 16384; model input limit "
                         "100000; output allowance 2048). Please shorten the question or reduce the evidence scope.")

    def test_constructor_requirements_and_source_nonvalidation(self):
        with self.assertRaises(TypeError):
            contract.RagServiceError("missing category")
        with self.assertRaises(TypeError):
            contract.RagServiceError("message", "input")
        with self.assertRaises(TypeError):
            contract.RagInputTooLongError(513, 512)
        with self.assertRaises(TypeError):
            contract.RagRequestTooLargeError(estimated_input_tokens=1)
        # Error objects describe failures; their constructors do not enforce numeric bounds.
        self.assertEqual(contract.RagInputTooLongError(token_count=-1, limit=0).token_count, -1)
        error = contract.RagServiceError("", category="unknown", attempts=-1)
        self.assertEqual((str(error), error.category, error.attempts), ("", "unknown", -1))

    def test_worker_fallback_categories_are_not_normalized(self):
        for category in ("provider_error", "request_preparation", "request_size_counting"):
            error = contract.RagProviderError("fixture", category=category)
            self.assertEqual(error.category, category)
            self.assertNotIn(error.category, FAILURE_CATEGORIES)
        self.assertIn("provider", FAILURE_CATEGORIES)

    def test_no_orchestration_or_serialization_is_exposed(self):
        for name in ("RagService", "_read_only_packet", "_response_model", "GeminiAdapter"):
            self.assertFalse(hasattr(contract, name))


if __name__ == "__main__":
    unittest.main()
