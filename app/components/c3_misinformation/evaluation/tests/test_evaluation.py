"""Synthetic, offline tests for C3 retrieval and recorded-RAG evaluation."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.components.c3_misinformation.evaluation import (
    EvaluationInputError,
    build_report,
    evaluate_documents,
    evaluate_rag_response,
    validate_predictions,
    validate_query_dataset,
)
from app.components.c3_misinformation.evaluation.__main__ import main as runner_main


def query(query_id="Q1", *, targets=None, alternatives=None):
    target_ids = targets or ["C-TARGET"]
    return {
        "query_id": query_id,
        "query": f"Synthetic question {query_id}?",
        "language": "en",
        "provenance": "constructed_evaluation_query",
        "target_chunk_ids": target_ids,
        "background_chunk_ids": ["C-BACKGROUND"],
        "evidence": {
            "chunk_id": target_ids[0],
            "text": "Synthetic reference text.",
            "metadata": {"source_id": "SYNTHETIC"},
        },
        **({"alternative_evidence_sets": alternatives} if alternatives is not None else {}),
    }


def dataset(*queries_):
    return {
        "schema_version": "1.0",
        "dataset_name": "synthetic-only-test-fixture",
        "review_status": "synthetic",
        "legal_review_status": "not_applicable",
        "scope": {"query_count": len(queries_)},
        "queries": list(queries_),
    }


def result(rank, chunk_id, distance=None):
    distance = float(rank) / 10 if distance is None else distance
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "distance": distance,
        "similarity": 1.0 - distance,
    }


def passage(chunk_id):
    return {
        "evidence_id": f"chunk:{chunk_id}",
        "chunk_id": chunk_id,
        "original_chunk_id": f"SOURCE-{chunk_id}",
        "text": "Synthetic recorded evidence passage.",
        "metadata": {"source_id": "SYNTHETIC", "title": "Synthetic"},
        "source": {"source_id": "SYNTHETIC", "title": "Synthetic"},
        "parent_context": None,
        "paragraph": None,
        "roles": ["ranked"],
    }


def rag_record(*, mode="claim", ids=("C-TARGET",), response_changes=None,
               request_text="Synthetic request"):
    evidence = [passage(chunk_id) for chunk_id in ids]
    ranked = [
        {
            "rank": index + 1,
            "evidence_id": item["evidence_id"],
            "distance": 0.1,
            "similarity": 0.9,
        }
        for index, item in enumerate(evidence)
    ]
    serialized_evidence = {
        "ranked_evidence": ranked,
        "context_evidence_ids": [],
        "passages": evidence,
        "relationships": [],
        "context_resolution_notices": [],
    }
    response = {
        "mode": mode,
        "response_origin": "model" if evidence else "application",
        "explanation": "Synthetic technical fixture.",
        "citations": [evidence[0]["evidence_id"]] if evidence else [],
        "insufficient_evidence": not evidence,
        "evidence": serialized_evidence,
        "limitations": [],
        "validation": {"status": "passed", "validator": "Draft202012Validator"},
        "model": "synthetic-model" if evidence else None,
        "usage": {"total_tokens": 1} if evidence else None,
        "attempts": 1 if evidence else 0,
        "finish_reason": "STOP" if evidence else None,
    }
    if mode == "claim":
        response["claim_verdict"] = "False" if evidence else None
    else:
        response["answer"] = "Synthetic response." if evidence else None
    response.update(response_changes or {})
    return {
        "mode": mode,
        "request_text": request_text,
        "retrieval_query": request_text,
        "expected_request_text": request_text,
        "evidence_packet": {
            "query": request_text,
            "ranked_evidence": ranked,
            "context_evidence_ids": [],
            "evidence": evidence,
            "relationships": [],
        },
        "response": response,
    }


class RetrievalEvaluationTests(unittest.TestCase):
    def test_fully_relevant_order_hit_metrics_and_reciprocal_rank(self):
        queries = [query()]
        predictions = {"Q1": {"retrieval_results": [result(1, "C-TARGET")]}}
        report = evaluate_documents(dataset(*queries), {"predictions": [
            {"query_id": "Q1", **predictions["Q1"]},
        ]}, ks=[1, 5])
        per_query = report["retrieval"]["per_query"][0]
        self.assertEqual(per_query["designated_target_rank"], 1)
        self.assertEqual(report["retrieval"]["metrics"]["1"]["target_hit_at_k"],
                         {"hits": 1, "rate": 1.0})
        self.assertEqual(report["retrieval"]["metrics"]["1"]["target_mrr_at_k"], 1.0)
        self.assertEqual(
            report["retrieval"]["metrics"]["5"]
            ["labelled_condition_id_coverage_at_k"]["rate"],
            1.0,
        )

    def test_relevant_below_k_no_relevant_and_empty_rankings_have_explicit_denominator(self):
        queries = [query("Q1"), query("Q2", targets=["C-MISSING"]), query("Q3")]
        predictions = {"predictions": [
            {"query_id": "Q1", "retrieval_results": [
                result(1, "A"), result(2, "B"), result(3, "C-TARGET"),
            ]},
            {"query_id": "Q2", "retrieval_results": [result(1, "OTHER")]},
            {"query_id": "Q3", "retrieval_results": []},
        ]}
        report = evaluate_documents(dataset(*queries), predictions, ks=[2, 3, 5])["retrieval"]
        self.assertEqual(report["evaluated_retrieval_count"], 3)
        self.assertEqual(report["metrics"]["2"]["target_hit_at_k"]["hits"], 0)
        self.assertEqual(report["metrics"]["3"]["target_hit_at_k"]["hits"], 1)
        self.assertAlmostEqual(report["metrics"]["3"]["target_mrr_at_k"], 1 / 9)
        self.assertEqual(report["metrics"]["5"]["denominator"], 3)
        self.assertEqual(report["per_query"][2]["returned_chunk_ids"], [])

    def test_missing_prediction_is_not_scored_as_zero_and_order_is_deterministic(self):
        qset = dataset(query("Q2"), query("Q1"))
        predictions = {"predictions": [
            {"query_id": "Q1", "retrieval_results": [result(1, "C-TARGET")]},
        ]}
        report = evaluate_documents(qset, predictions)
        self.assertEqual(report["evaluation_scope"]["query_order"], ["Q2", "Q1"])
        self.assertEqual(report["retrieval"]["missing_retrieval_query_ids"], ["Q2"])
        self.assertEqual(report["retrieval"]["metrics"]["1"]["denominator"], 1)

    def test_multiple_direct_targets_are_independent_valid_hits(self):
        q = query(targets=["T1", "T2"])
        predictions = {"predictions": [
            {"query_id": "Q1", "retrieval_results": [
                result(1, "X"), result(2, "T2"),
            ]},
        ]}
        report = evaluate_documents(dataset(q), predictions, ks=[2, 5])
        self.assertEqual(report["retrieval"]["per_query"][0]["designated_target_rank"], 2)
        self.assertEqual(
            report["retrieval"]["per_query"][0]["metrics"]["2"]["completed_condition_ids"],
            ["target:T2"],
        )

    def test_q08_alternative_evidence_requires_both_and_remains_separate(self):
        q08 = query(
            "Q08",
            targets=["CON-003-0068"],
            alternatives=[{
                "required_chunk_ids": ["CON-003-0170", "CON-003-0172"],
                "reason": (
                    "Section 29(b) describes the clause; the section 29 opening "
                    "states that it is void."
                ),
            }],
        )
        predictions = {"predictions": [{
            "query_id": "Q08",
            "retrieval_results": [
                result(1, "CON-003-0170", 0.1),
                result(2, "OTHER", 0.2),
                result(3, "CON-003-0172", 0.3),
            ],
        }]}
        report = evaluate_documents(
            dataset(q08), predictions, ks=[2, 3],
        )["retrieval"]["per_query"][0]
        self.assertEqual(report["designated_target_rank"], None)
        alternative = report["evidence_conditions"][1]
        self.assertEqual(alternative["required_chunk_ids"],
                         ["CON-003-0170", "CON-003-0172"])
        self.assertEqual(
            alternative["reason"],
            "Section 29(b) describes the clause; the section 29 opening states that it is void.",
        )
        self.assertEqual(alternative["completion_rank"], 3)
        self.assertEqual(report["metrics"]["2"]["labelled_condition_id_coverage"], False)
        self.assertEqual(report["metrics"]["3"]["labelled_condition_id_coverage"], True)
        self.assertEqual(
            report["metrics"]["3"]["completed_condition_ids"],
            ["alternative:1"],
        )

    def test_invalid_dataset_and_predictions_are_rejected_not_repaired(self):
        with self.assertRaisesRegex(EvaluationInputError, "duplicate query_id"):
            validate_query_dataset(dataset(query("Q1"), query("Q1")))
        with self.assertRaisesRegex(EvaluationInputError, "duplicate prediction query_id"):
            validate_predictions(
                {"predictions": [
                    {"query_id": "Q1", "retrieval_results": []},
                    {"query_id": "Q1", "retrieval_results": []},
                ]},
                {"Q1"},
            )
        with self.assertRaisesRegex(EvaluationInputError, "unknown prediction query_id"):
            validate_predictions(
                {"predictions": [{"query_id": "QX", "retrieval_results": []}]},
                {"Q1"},
            )
        with self.assertRaisesRegex(EvaluationInputError, "duplicate retrieved"):
            validate_predictions(
                {"predictions": [{
                    "query_id": "Q1",
                    "retrieval_results": [result(1, "C"), result(2, "C", 0.2)],
                }]},
                {"Q1"},
            )
        with self.assertRaisesRegex(EvaluationInputError, "contiguous one-based"):
            validate_predictions(
                {"predictions": [{
                    "query_id": "Q1", "retrieval_results": [result(2, "C")],
                }]},
                {"Q1"},
            )
        with self.assertRaisesRegex(EvaluationInputError, "ordered by distance"):
            validate_predictions(
                {"predictions": [{
                    "query_id": "Q1", "retrieval_results": [
                        result(1, "Z", 0.2), result(2, "A", 0.1),
                    ],
                }]},
                {"Q1"},
            )
        with self.assertRaisesRegex(EvaluationInputError, "distance must be finite"):
            validate_predictions(
                {"predictions": [{
                    "query_id": "Q1", "retrieval_results": [{
                        **result(1, "C"), "distance": float("nan"),
                    }],
                }]},
                {"Q1"},
            )


class RagEvaluationTests(unittest.TestCase):
    def test_valid_claim_response_and_citation_ids(self):
        report = evaluate_rag_response(
            rag_record(), expected_request_text="Synthetic request",
        )
        self.assertEqual(report["status"], "valid")
        self.assertTrue(report["schema_valid"])
        self.assertTrue(report["citation_ids_valid"])
        self.assertTrue(report["request_text_matches_expected"])
        self.assertEqual(report["semantic_support"], "manual_review_required")

    def test_valid_question_response_uses_question_contract(self):
        report = evaluate_rag_response(rag_record(mode="question"))
        self.assertEqual(report["status"], "valid")
        self.assertTrue(report["schema_valid"])

    def test_invalid_citation_and_mismatched_mode_are_not_legal_judgments(self):
        invalid_citation = rag_record(response_changes={"citations": ["chunk:ABSENT"]})
        report = evaluate_rag_response(invalid_citation)
        self.assertEqual(report["status"], "invalid")
        self.assertIn("citation_id_not_in_evidence_packet", report["issues"])
        wrong_mode = rag_record(mode="question")
        wrong_mode["response"]["mode"] = "claim"
        wrong_mode["response"].pop("answer")
        wrong_mode["response"]["claim_verdict"] = "False"
        report = evaluate_rag_response(wrong_mode)
        self.assertIn("response_mode_mismatch", report["issues"])

    def test_empty_evidence_requires_application_abstention_for_claim_and_question(self):
        for mode in ("claim", "question"):
            with self.subTest(mode=mode):
                report = evaluate_rag_response(rag_record(mode=mode, ids=()))
                self.assertEqual(report["status"], "valid")
        invalid = rag_record(
            ids=(),
            response_changes={
                "response_origin": "model",
                "insufficient_evidence": True,
                "claim_verdict": "False",
                "citations": [],
            },
        )
        report = evaluate_rag_response(invalid)
        self.assertIn("empty_evidence_must_use_application_abstention", report["issues"])
        self.assertIn("insufficient_claim_must_have_null_verdict", report["issues"])

    def test_missing_explanation_and_unavailable_rag_are_reported_explicitly(self):
        missing = rag_record(response_changes={"explanation": None})
        del missing["response"]["explanation"]
        report = evaluate_rag_response(missing)
        self.assertIn("missing_or_blank_explanation", report["issues"])
        unavailable = evaluate_rag_response({"status": "unavailable"})
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertIsNone(unavailable["schema_valid"])

    def test_nonempty_insufficient_response_does_not_turn_into_a_verdict(self):
        invalid = rag_record(response_changes={
            "insufficient_evidence": True,
            "citations": [],
            "claim_verdict": "False",
        })
        report = evaluate_rag_response(invalid)
        self.assertIn("insufficient_claim_must_have_null_verdict", report["issues"])

    def test_packet_identity_and_request_text_must_be_consistent(self):
        record = rag_record()
        record["evidence_packet"]["evidence"][0]["evidence_id"] = "chunk:OTHER"
        record["request_text"] = "different"
        report = evaluate_rag_response(record, expected_request_text="expected")
        self.assertIn("evidence_id_mismatch:0", report["issues"])
        self.assertIn("request_text_does_not_match_expected", report["issues"])


class EvaluationRunnerTests(unittest.TestCase):
    def test_report_is_reproducible_and_contains_input_hashes(self):
        qset = dataset(query())
        predictions = {"predictions": [{
            "query_id": "Q1",
            "retrieval_results": [result(1, "C-TARGET")],
            "rag": rag_record(),
        }]}
        kwargs = {
            "query_sha256": "a" * 64,
            "predictions_sha256": "b" * 64,
            "ks": [1, 3],
        }
        first = build_report(qset, predictions, **kwargs)
        second = build_report(qset, predictions, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["inputs"]["query_dataset_sha256"], "a" * 64)
        self.assertEqual(first["rag"]["valid_count"], 1)

    def test_cli_writes_json_markdown_hashes_and_refuses_overwrite(self):
        qset = dataset(query())
        predictions = {"predictions": [{
            "query_id": "Q1", "retrieval_results": [result(1, "C-TARGET")],
        }]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            query_path = root / "queries.json"
            prediction_path = root / "predictions.json"
            output = root / "out" / "report.json"
            markdown = root / "out" / "report.md"
            query_path.write_text(json.dumps(qset), encoding="utf-8")
            prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = runner_main([
                    "--queries", str(query_path),
                    "--predictions", str(prediction_path),
                    "--output", str(output),
                    "--markdown", str(markdown),
                    "--k", "1",
                ])
            self.assertEqual(code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                report["inputs"]["query_dataset_sha256"],
                hashlib.sha256(query_path.read_bytes()).hexdigest(),
            )
            self.assertIn("Target MRR@K", markdown.read_text(encoding="utf-8"))

            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                runner_main([
                    "--queries", str(query_path),
                    "--predictions", str(prediction_path),
                    "--output", str(output),
                ])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("must not already exist", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
