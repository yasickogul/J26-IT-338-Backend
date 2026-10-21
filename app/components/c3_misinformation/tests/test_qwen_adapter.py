import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.components.c3_misinformation.agents import qwen_adapter
from app.components.c3_misinformation.agents.qwen_adapter import LocalQwenAdapter
from app.components.c3_misinformation.agents.qwen_translation_provider import (
    QwenTranslationProvider,
)
from app.components.c3_misinformation.config import create_structured_adapter
from app.components.c3_misinformation.pipeline.claim_translation import translate_claim

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}
PROMPT = {"trusted_instructions": "Use evidence only.", "input": {"question": "What is the law?"}}


class FakeResponse:
    def __init__(self, payload, *, finish="stop", status=200):
        self._payload = payload
        self.status_code = status
        self._finish = finish

    def raise_for_status(self):
        if self.status_code >= 400:
            raise qwen_adapter.httpx.HTTPStatusError("error", request=SimpleNamespace(), response=self)

    def json(self):
        if isinstance(self._payload, BaseException):
            raise self._payload
        return {"choices": [{"finish_reason": self._finish, "message": {"content": self._payload}}]}


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class QwenAdapterTests(unittest.TestCase):
    def test_valid_structured_response_and_request_contract(self):
        client = FakeClient(FakeResponse('{"answer":"Supported"}'))
        result = LocalQwenAdapter(client=client).generate_json(PROMPT, SCHEMA)
        self.assertEqual(result.response, {"answer": "Supported"})
        self.assertEqual(result.model, "qwen/qwen3-4b-2507")
        self.assertEqual(client.calls[0][0], "http://localhost:1234/v1/chat/completions")
        self.assertEqual(client.calls[0][1]["json"]["response_format"]["type"], "json_schema")

    def test_prompt_preserves_full_conditional_contract_for_qwen(self):
        schema = {
            "type": "object",
            "properties": {
                "claim_verdict": {"type": ["string", "null"]},
                "insufficient_evidence": {"type": "boolean"},
                "explanation": {"type": "string", "minLength": 1},
            },
            "required": ["claim_verdict", "insufficient_evidence", "explanation"],
            "allOf": [
                {
                    "if": {"properties": {"insufficient_evidence": {"const": True}}},
                    "then": {"properties": {"claim_verdict": {"type": "null"}}},
                },
            ],
            "additionalProperties": False,
        }
        system, _user = LocalQwenAdapter._contents(PROMPT, schema)
        self.assertIn('"allOf"', system)
        self.assertIn('"minLength":1', system)

    def test_invalid_empty_malformed_and_truncated_outputs_use_existing_errors(self):
        cases = (
            ("", qwen_adapter.GeminiEmptyResponseError),
            ("not-json", qwen_adapter.GeminiMalformedResponseError),
            ('{"answer": 3}', qwen_adapter.GeminiInvalidResponseError),
        )
        for payload, error in cases:
            with self.subTest(payload=payload), self.assertRaises(error):
                LocalQwenAdapter(client=FakeClient(FakeResponse(payload))).generate_json(PROMPT, SCHEMA)
        with self.assertRaises(qwen_adapter.GeminiTruncatedResponseError):
            LocalQwenAdapter(client=FakeClient(FakeResponse("", finish="length"))).generate_json(PROMPT, SCHEMA)

    def test_unreachable_server_is_safe_provider_error(self):
        client = FakeClient(qwen_adapter.httpx.ConnectError("offline"))
        with self.assertRaises(qwen_adapter.GeminiAdapterError) as caught:
            LocalQwenAdapter(client=client).generate_json(PROMPT, SCHEMA)
        self.assertEqual(caught.exception.failure_category, "provider_error")

    def test_qwen_translation_preserves_original_and_english_bypass(self):
        translated = "The court granted the application."
        client = FakeClient(FakeResponse(json.dumps({"translated_text": translated})))
        result = translate_claim("සිංහල claim", "si", provider=QwenTranslationProvider(adapter=LocalQwenAdapter(client=client)))
        self.assertEqual(result.translated_text, translated)
        self.assertEqual(result.method, "qwen:qwen/qwen3-4b-2507")
        self.assertEqual(translate_claim("English claim", "en").status, "skipped")

    def test_provider_selection_is_explicit_and_default_stays_gemini(self):
        with patch.dict("os.environ", {"C3_LLM_PROVIDER": "qwen"}, clear=False):
            self.assertIsInstance(create_structured_adapter(), LocalQwenAdapter)
        with patch.dict("os.environ", {"C3_LLM_PROVIDER": "gemini"}, clear=False), patch(
            "app.components.c3_misinformation.config.GeminiAdapter"
        ) as gemini:
            create_structured_adapter()
            gemini.assert_called_once()


if __name__ == "__main__":
    unittest.main()
