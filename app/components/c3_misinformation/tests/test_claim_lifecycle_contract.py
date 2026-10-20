"""Offline tests for independent claim lifecycle contracts."""

from dataclasses import FrozenInstanceError, MISSING, asdict, fields
from math import inf, nan
from types import MappingProxyType
from typing import get_type_hints
import unittest
from uuid import UUID, uuid4

from app.components.c3_misinformation import claim_lifecycle_contract as contract


def claim_response(*, verdict=None, insufficient=True):
    return {
        "mode": "claim", "response_origin": "application", "claim_verdict": verdict,
        "explanation": "No evidence was retrieved.", "citations": [],
        "insufficient_evidence": insufficient,
        "evidence": {"passages": [], "ranked_evidence": []},
        "limitations": [], "validation": {"status": "passed"},
        "model": None, "usage": None, "attempts": 0, "finish_reason": None,
    }


class ClaimLifecycleContractTests(unittest.TestCase):
    def test_failure_categories_are_exact_and_immutable(self):
        self.assertEqual(contract.FAILURE_CATEGORIES, frozenset({
            "input", "input_too_long", "request_too_large", "retrieval",
            "preparation", "provider", "response_validation", "service",
            "result_storage", "interrupted",
        }))
        self.assertIsInstance(contract.FAILURE_CATEGORIES, frozenset)

    def test_claimed_text_claim_fields_default_and_frozen_behavior(self):
        claim_id, token = uuid4(), uuid4()
        legacy = contract.ClaimedTextClaim(claim_id, "raw", token)
        cleaned = contract.ClaimedTextClaim(claim_id, "raw", token, "cleaned")
        self.assertEqual(asdict(legacy), {
            "claim_id": claim_id, "raw_input_text": "raw",
            "worker_token": token, "cleaned_text": None,
        })
        self.assertEqual(cleaned.cleaned_text, "cleaned")
        self.assertEqual([field.name for field in fields(legacy)], [
            "claim_id", "raw_input_text", "worker_token", "cleaned_text",
        ])
        self.assertTrue(all(field.default is MISSING for field in fields(legacy)[:3]))
        self.assertIsNone(fields(legacy)[3].default)
        self.assertEqual(get_type_hints(contract.ClaimedTextClaim)["claim_id"], UUID)
        self.assertEqual(get_type_hints(contract.ClaimedTextClaim)["cleaned_text"], str | None)
        with self.assertRaises(FrozenInstanceError):
            legacy.cleaned_text = "changed"

    def test_error_classes_keep_bases_and_messages(self):
        storage = contract.ClaimStorageError("storage failed")
        state = contract.ClaimStateError("state unavailable")
        self.assertIsInstance(storage, RuntimeError)
        self.assertIsInstance(state, ValueError)
        self.assertEqual(str(storage), "storage failed")
        self.assertEqual(str(state), "state unavailable")

    def test_require_uuid_accepts_uuid_and_rejects_other_values(self):
        self.assertIsNone(contract._require_uuid(uuid4()))
        for value in (None, "00000000-0000-0000-0000-000000000000", 0, b"id", False):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"^claim_id must be a UUID$"):
                    contract._require_uuid(value)

    def test_insufficient_evidence_snapshot_is_json_copy_with_extra_fields(self):
        payload = claim_response()
        payload["extra"] = {"numbers": (1, 2)}
        snapshot, verdict = contract._result_snapshot(MappingProxyType(payload))
        self.assertEqual(verdict, "insufficient_evidence")
        self.assertEqual(snapshot["extra"], {"numbers": [1, 2]})
        self.assertEqual(snapshot["evidence"]["passages"], [])
        self.assertIsNone(snapshot["claim_verdict"])
        self.assertIsNot(snapshot, payload)
        self.assertIsNot(snapshot["evidence"], payload["evidence"])

    def test_supported_verdicts_preserve_complete_snapshot(self):
        for verdict in ("True", "False", "Misleading"):
            with self.subTest(verdict=verdict):
                payload = claim_response(verdict=verdict, insufficient=False)
                payload["citations"] = ["chunk:one"]
                payload["evidence"]["passages"] = [{"evidence_id": "chunk:one"}]
                payload["model"] = "fixture"
                self.assertEqual(contract._result_snapshot(payload), (payload, verdict))

    def test_nonmapping_or_nonserializable_payload_is_rejected(self):
        for value in (None, [], "claim", 4):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"^result payload must be an object$"):
                    contract._result_snapshot(value)
        for value in (object(), nan, inf):
            payload = claim_response()
            payload["extra"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"^result payload must be JSON serializable$"):
                    contract._result_snapshot(payload)
        cycle = claim_response()
        cycle["extra"] = cycle
        with self.assertRaisesRegex(ValueError, r"^result payload must be JSON serializable$"):
            contract._result_snapshot(cycle)

    def test_required_shape_and_boolean_validation(self):
        variations = (
            lambda p: p.pop("usage"),
            lambda p: p.update(mode="question"),
            lambda p: p.update(insufficient_evidence=1),
            lambda p: p.update(citations="chunk:one"),
            lambda p: p.update(evidence=[]),
            lambda p: p["evidence"].update(passages=None),
        )
        for index, mutate in enumerate(variations):
            payload = claim_response()
            mutate(payload)
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, r"^result payload is not a complete claim response$"):
                    contract._result_snapshot(payload)

    def test_insufficient_evidence_requires_null_verdict(self):
        with self.assertRaisesRegex(ValueError, r"^insufficient evidence must have a null claim verdict$"):
            contract._result_snapshot(claim_response(verdict="False"))

    def test_supported_result_requires_source_verdict(self):
        for verdict in (None, "Unknown", 1, False):
            with self.subTest(verdict=verdict):
                with self.assertRaisesRegex(ValueError, r"^supported claim result has no valid verdict$"):
                    contract._result_snapshot(claim_response(verdict=verdict, insufficient=False))

    def test_explanation_and_model_types(self):
        for field, message in (("explanation", "explanation must be text or null"),
                               ("model", "model must be text or null")):
            payload = claim_response()
            payload[field] = 123
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"^{message}$"):
                    contract._result_snapshot(payload)

    def test_database_operations_are_absent(self):
        for name in ("claim_next_queued_text_claim", "renew_claim_lease", "record_completed_result", "record_claim_failure"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(contract, name))


if __name__ == "__main__":
    unittest.main()
