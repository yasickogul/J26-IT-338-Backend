"""Offline tests for adapting async RAG analysis to the existing claim worker."""
import unittest
from unittest.mock import AsyncMock

from app.components.c3_misinformation.prompts.rag_prompt import ClaimInput
from app.components.c3_misinformation.worker_rag_adapter import (
    create_worker_rag_analyzer,
)


class WorkerRagAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_forwards_claim_top_k_and_optional_filters(self):
        result = object()

        class Service:
            run = AsyncMock(return_value=result)

        service = Service()
        analyze = create_worker_rag_analyzer(
            service, source_id="source", legal_domain="civil", language="English",
        )

        self.assertIs(await analyze("Claim as received.", top_k=9), result)
        service.run.assert_awaited_once_with(
            ClaimInput(claim="Claim as received."),
            top_k=9,
            source_id="source",
            legal_domain="civil",
            language="English",
        )

    async def test_adapter_does_not_create_claim_retrieval_query(self):
        class Service:
            def __init__(self):
                self.request = None

            async def run(self, request, **kwargs):
                self.request = request
                return request

        service = Service()
        analyze = create_worker_rag_analyzer(service)
        request = await analyze("Claim text.")
        self.assertEqual(request, ClaimInput(claim="Claim text."))
        self.assertIsNone(request.retrieval_query)

    def test_service_run_is_required(self):
        with self.assertRaisesRegex(TypeError, "service must provide async run"):
            create_worker_rag_analyzer(object())


if __name__ == "__main__":
    unittest.main()
