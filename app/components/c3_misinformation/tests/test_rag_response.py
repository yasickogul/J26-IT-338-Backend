import json
import unittest

from pydantic import ValidationError

from app.components.c3_misinformation.rag_contract import RagResult
from app.components.c3_misinformation.rag_response import (
    ClaimRagResponse,
    QuestionRagResponse,
    serialize_rag_result,
)


PACKET = {
    "ranked_evidence": [{"rank": 1, "evidence_id": "e1", "distance": 0.1, "similarity": 0.9}],
    "context_evidence_ids": ["e2"],
    "evidence": [{
        "evidence_id": "e1", "chunk_id": "c1", "original_chunk_id": "c1",
        "text": "fixture", "metadata": {"source_id": "s1"}, "source": {"id": "s1"},
        "parent_context": None, "paragraph": None, "roles": ["ranked"],
    }],
    "relationships": [],
    "context_resolution_notices": [],
}


def result(mode, response, *, origin="model"):
    return RagResult(
        mode=mode, evidence_packet=PACKET, response=response,
        response_origin=origin, validation={"status": "passed", "validator": "Draft202012Validator"},
        model="fixture-model" if origin == "model" else None,
        usage={"total_tokens": 2} if origin == "model" else None,
        attempts=1 if origin == "model" else 0,
        finish_reason="STOP" if origin == "model" else None,
        limitations=["fixture limitation"], transaction_status={},
    )


class RagResponseTests(unittest.TestCase):
    def test_claim_serialization_preserves_contract(self):
        value = serialize_rag_result(result("claim", {
            "claim_verdict": "True", "citations": ["e1"],
            "insufficient_evidence": False, "explanation": "supported",
        }))
        self.assertIsInstance(value, ClaimRagResponse)
        self.assertEqual(value.model_dump(mode="json")["claim_verdict"], "True")
        self.assertEqual(value.model_dump(mode="json")["evidence"]["passages"][0]["evidence_id"], "e1")

    def test_question_and_insufficient_evidence(self):
        value = serialize_rag_result(result("question", {
            "answer": None, "citations": [], "insufficient_evidence": True,
            "explanation": "No evidence was retrieved for this request.",
        }, origin="application"))
        self.assertIsInstance(value, QuestionRagResponse)
        payload = value.model_dump(mode="json")
        self.assertTrue(payload["insufficient_evidence"])
        self.assertIsNone(payload["answer"])
        json.dumps(payload)

    def test_missing_required_response_field_fails(self):
        with self.assertRaises(KeyError):
            serialize_rag_result(result("claim", {
                "claim_verdict": None, "citations": [], "explanation": None,
            }))

    def test_invalid_evidence_shape_is_rejected(self):
        bad = dict(PACKET)
        bad["evidence"] = [{"evidence_id": "only-id"}]
        value = result("claim", {
            "claim_verdict": "True", "citations": [], "insufficient_evidence": False,
            "explanation": None,
        })
        value = RagResult(**{**value.__dict__, "evidence_packet": bad})
        with self.assertRaises(ValidationError):
            serialize_rag_result(value)


if __name__ == "__main__":
    unittest.main()
