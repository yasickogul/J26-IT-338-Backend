"""Offline tests for independent evidence packet assembly."""

from dataclasses import dataclass
from math import inf, nan
from types import SimpleNamespace
import unittest

from app.components.c3_misinformation.pipeline.evidence_packet import assemble_evidence_packet


@dataclass(frozen=True)
class Notice:
    child_chunk_id: str
    reason: str


def ranked(chunk_id="child", *, text="Child text", metadata=None, distance=0.2, similarity=0.8):
    if metadata is None:
        metadata = {"source_id": "act-1", "section_label": "Part > Section", "title": "Act"}
    return SimpleNamespace(
        chunk_id=chunk_id,
        original_chunk_id=f"original-{chunk_id}",
        text=text,
        metadata=metadata,
        parent_context={"heading": "Part"},
        paragraph="1",
        distance=distance,
        similarity=similarity,
    )


def parent(child_chunk_id="child", *, metadata=None, text="Parent text"):
    if metadata is None:
        metadata = {"source_id": "act-1", "section_label": "Part", "title": "Act"}
    return SimpleNamespace(
        parent_chunk_id="parent",
        child_chunk_id=child_chunk_id,
        text=text,
        metadata=metadata,
        parent_context=None,
        paragraph=None,
    )


def expansion(parents=(), unresolved=()):
    return SimpleNamespace(parents=parents, unresolved=unresolved)


class EvidencePacketTests(unittest.TestCase):
    def test_valid_evidence_preserves_order_scores_metadata_and_relationships(self):
        first = ranked()
        second = ranked("child-2")
        settings = {"top_k": 2, "filters": {"source_id": "act-1"}}
        packet = assemble_evidence_packet(
            "Legal question", [first, second],
            expansion([parent()], [Notice("child-2", "unresolved cross-reference")]),
            original_chunk_ids={"parent": "original-parent"},
            retrieval_settings=settings,
        )

        self.assertEqual(packet["schema_version"], "1.0")
        self.assertEqual(packet["query"], "Legal question")
        self.assertEqual(packet["retrieval_state"], "evidence_retrieved")
        self.assertEqual(packet["legal_coverage"], "not_assessed")
        self.assertIsNone(packet["verdict"])
        self.assertIsNone(packet["confidence"])
        self.assertEqual([item["evidence_id"] for item in packet["evidence"]],
                         ["chunk:child", "chunk:child-2", "chunk:parent"])
        self.assertEqual(packet["ranked_evidence"], [
            {"rank": 1, "evidence_id": "chunk:child", "distance": 0.2, "similarity": 0.8},
            {"rank": 2, "evidence_id": "chunk:child-2", "distance": 0.2, "similarity": 0.8},
        ])
        self.assertEqual(packet["context_evidence_ids"], ["chunk:parent"])
        self.assertEqual(packet["evidence"][0]["original_chunk_id"], "original-child")
        self.assertEqual(packet["evidence"][0]["source"]["title"], "Act")
        self.assertEqual(packet["evidence"][0]["parent_context"], {"heading": "Part"})
        self.assertEqual(packet["evidence"][0]["paragraph"], "1")
        self.assertEqual(packet["evidence"][2]["roles"], ["structural_context"])
        self.assertEqual(packet["relationships"], [
            {"relationship": "structural_parent_of", "parent_evidence_id": "chunk:parent", "child_evidence_id": "chunk:child", "source_id": "act-1"},
            {"relationship": "structural_parent_of", "parent_evidence_id": "chunk:parent", "child_evidence_id": "chunk:child-2", "source_id": "act-1"},
        ])
        self.assertEqual(packet["context_resolution_notices"],
                         [{"child_chunk_id": "child-2", "reason": "unresolved cross-reference"}])
        self.assertEqual(len(packet["limitations"]), 7)
        self.assertIn("untrusted data", packet["evidence_handling"])
        self.assertEqual(packet["retrieval_settings"], settings)
        settings["filters"]["source_id"] = "changed"
        first.metadata["title"] = "Changed"
        self.assertEqual(packet["retrieval_settings"]["filters"]["source_id"], "act-1")
        self.assertEqual(packet["evidence"][0]["metadata"]["title"], "Act")

    def test_empty_evidence(self):
        packet = assemble_evidence_packet("Question", [], expansion(), original_chunk_ids={})
        self.assertEqual(packet["retrieval_state"], "no_evidence_retrieved")
        for field in ("ranked_evidence", "context_evidence_ids", "evidence", "relationships", "context_resolution_notices"):
            self.assertEqual(packet[field], [])
        self.assertEqual(packet["retrieval_settings"], {})
        self.assertEqual(len(packet["limitations"]), 7)

    def test_citation_ids_reference_supplied_evidence_only(self):
        packet = assemble_evidence_packet(
            "Question", [ranked()], expansion([parent()]), original_chunk_ids={"parent": "original-parent"}
        )
        ids = {item["evidence_id"] for item in packet["evidence"]}
        self.assertEqual(ids, {"chunk:child", "chunk:parent"})
        self.assertTrue({item["evidence_id"] for item in packet["ranked_evidence"]} <= ids)
        self.assertTrue(set(packet["context_evidence_ids"]) <= ids)
        for relation in packet["relationships"]:
            self.assertIn(relation["parent_evidence_id"], ids)
            self.assertIn(relation["child_evidence_id"], ids)

    def test_query_and_ranked_evidence_validation(self):
        for query in ("", "  ", None, 123):
            with self.subTest(query=query), self.assertRaises(ValueError):
                assemble_evidence_packet(query, [], expansion(), original_chunk_ids={})
        cases = [
            ([ranked(text=" ")], "Missing legal text"),
            ([ranked(metadata=[])], "Metadata must be an object"),
            ([ranked(distance=nan)], "Non-finite retrieval score"),
            ([ranked(similarity=inf)], "Non-finite retrieval score"),
            ([ranked(), ranked()], "Repeated ranked result"),
        ]
        for results, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                assemble_evidence_packet("Question", results, expansion(), original_chunk_ids={})
        item = ranked()
        item.original_chunk_id = " "
        with self.assertRaisesRegex(ValueError, "evidence chunk ID must be nonblank"):
            assemble_evidence_packet("Question", [item], expansion(), original_chunk_ids={})

    def test_parent_and_relationship_validation(self):
        cases = [
            (parent(), {}, "Missing original-chunk reference"),
            (parent("missing-child"), {"parent": "original-parent"}, "references missing evidence"),
            (parent(metadata={"section_label": "Part"}), {"parent": "original-parent"}, "no source_id"),
            (parent(metadata={"source_id": "other", "section_label": "Part"}), {"parent": "original-parent"}, "crosses legal sources"),
            (parent(metadata={"source_id": "act-1", "section_label": "Other"}), {"parent": "original-parent"}, "structural label"),
            (parent(text=" "), {"parent": "original-parent"}, "Missing legal text"),
        ]
        for item, originals, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                assemble_evidence_packet("Question", [ranked()], expansion([item]), original_chunk_ids=originals)


if __name__ == "__main__":
    unittest.main()
