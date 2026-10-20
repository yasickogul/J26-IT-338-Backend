"""Offline tests for independent claim worker contracts."""

from dataclasses import FrozenInstanceError, MISSING, asdict, fields
from typing import Literal, get_type_hints
import unittest
from uuid import UUID

from app.components.c3_misinformation import claim_worker_contract as contract


CLAIM_ID = UUID("03030303-0303-4303-8303-030303030303")


class ClaimWorkerContractTests(unittest.TestCase):
    def test_worker_outcome_preserves_all_source_statuses(self):
        for status in ("completed", "failed", "ownership_lost"):
            with self.subTest(status=status):
                outcome = contract.WorkerOutcome(CLAIM_ID, status)
                self.assertEqual(asdict(outcome), {"claim_id": CLAIM_ID, "status": status})
                self.assertEqual(outcome, contract.WorkerOutcome(claim_id=CLAIM_ID, status=status))

    def test_worker_outcome_fields_have_no_defaults_and_are_frozen(self):
        outcome = contract.WorkerOutcome(CLAIM_ID, "completed")
        self.assertEqual([field.name for field in fields(outcome)], ["claim_id", "status"])
        self.assertTrue(all(field.default is MISSING and field.default_factory is MISSING
                            for field in fields(outcome)))
        self.assertEqual(get_type_hints(contract.WorkerOutcome)["claim_id"], UUID)
        self.assertEqual(get_type_hints(contract.WorkerOutcome)["status"],
                         Literal["completed", "failed", "ownership_lost"])
        with self.assertRaises(FrozenInstanceError):
            outcome.status = "failed"
        with self.assertRaises(TypeError):
            contract.WorkerOutcome(CLAIM_ID)

    def test_worker_outcome_has_no_runtime_status_or_uuid_validation(self):
        # The source dataclass relies on annotations but does not validate at runtime.
        self.assertEqual(contract.WorkerOutcome(CLAIM_ID, "unexpected").status, "unexpected")
        self.assertEqual(contract.WorkerOutcome("not-a-uuid", "failed").claim_id, "not-a-uuid")

    def test_extraction_error_keeps_reason_and_category(self):
        for reason, category in (("ambiguous", "input"),
                                 ("invalid_provider_response", "response_validation"),
                                 ("provider unavailable", "provider")):
            with self.subTest(category=category):
                error = contract.ClaimExtractionWorkerError(reason, failure_category=category)
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(str(error), reason)
                self.assertEqual(error.args, (reason,))
                self.assertEqual(error.failure_category, category)

    def test_extraction_error_preserves_keyword_only_category_and_no_validation(self):
        with self.assertRaises(TypeError):
            contract.ClaimExtractionWorkerError("reason", "input")
        with self.assertRaises(TypeError):
            contract.ClaimExtractionWorkerError("reason")
        error = contract.ClaimExtractionWorkerError("", failure_category="unrecognized")
        self.assertEqual(str(error), "")
        self.assertEqual(error.failure_category, "unrecognized")

    def test_worker_execution_and_source_specific_helpers_are_absent(self):
        for name in ("process_one_claim", "get_claim_extraction_enabled", "_extraction_provenance",
                     "claim_next_queued_text_claim", "renew_claim_lease"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(contract, name))


if __name__ == "__main__":
    unittest.main()
