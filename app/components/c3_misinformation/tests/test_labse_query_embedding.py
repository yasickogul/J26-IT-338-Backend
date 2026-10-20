import unittest

import numpy as np

from app.components.c3_misinformation.pipeline.labse_query_embedding import (
    EXPECTED_DIMENSION,
    MODEL_NAME,
    EmbeddingInputTooLongError,
    embed_text,
    embed_texts,
    validate_input_token_limits,
)


class FakeTokenizer:
    def __init__(self, count=3):
        self.count = count
        self.calls = []

    def __call__(self, values, **kwargs):
        self.calls.append((values, kwargs))
        return {"input_ids": [[0] * self.count for _ in values]}


class FakeEncoder:
    def __init__(self, vectors=None, dimension=EXPECTED_DIMENSION, limit=16):
        self.tokenizer = FakeTokenizer()
        self.max_seq_length = limit
        self.dimension = dimension
        self.vectors = vectors
        self.calls = []

    def get_embedding_dimension(self):
        return self.dimension

    def encode(self, values, **kwargs):
        self.calls.append((values, kwargs))
        if self.vectors is not None:
            return self.vectors
        return np.tile(np.array([1.0] + [0.0] * (EXPECTED_DIMENSION - 1)), (len(values), 1))


class LabseEmbeddingTests(unittest.TestCase):
    def test_identity_and_valid_query_embedding(self):
        encoder = FakeEncoder()
        vector = embed_text("a valid legal query", encoder=encoder)
        self.assertEqual(MODEL_NAME, "sentence-transformers/LaBSE")
        self.assertEqual(vector.shape, (EXPECTED_DIMENSION,))
        self.assertTrue(np.isfinite(vector).all())
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0)
        self.assertTrue(encoder.calls[0][1]["normalize_embeddings"])

    def test_batch_order_and_tokenizer_no_truncation(self):
        encoder = FakeEncoder()
        vectors = embed_texts(["first", "second"], encoder=encoder)
        self.assertEqual(vectors.shape, (2, EXPECTED_DIMENSION))
        kwargs = encoder.tokenizer.calls[0][1]
        self.assertEqual(kwargs, {"truncation": False, "add_special_tokens": True, "verbose": False})

    def test_empty_and_invalid_texts_match_source_validation(self):
        encoder = FakeEncoder()
        for value in ([], "single string", ["   "], [42]):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    embed_texts(value, encoder=encoder)
        with self.assertRaises(TypeError):
            embed_text(42, encoder=encoder)

    def test_over_limit_input_raises_source_error(self):
        encoder = FakeEncoder(limit=2)
        with self.assertRaisesRegex(EmbeddingInputTooLongError, r"Input\[0\] has 3 tokens \(limit 2\)"):
            validate_input_token_limits(["query"], encoder=encoder)
        self.assertEqual(encoder.calls, [])

    def test_wrong_dimension_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "expected dimension 768, got 384"):
            embed_text("query", encoder=FakeEncoder(dimension=384))

    def test_wrong_vector_shape_is_rejected(self):
        vectors = np.zeros((1, EXPECTED_DIMENSION - 1))
        with self.assertRaisesRegex(RuntimeError, "Unexpected embedding shape"):
            embed_text("query", encoder=FakeEncoder(vectors=vectors))

    def test_nonfinite_vectors_are_rejected(self):
        vectors = np.zeros((1, EXPECTED_DIMENSION))
        vectors[0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            embed_text("query", encoder=FakeEncoder(vectors=vectors))

    def test_non_normalized_vectors_are_rejected(self):
        vectors = np.zeros((1, EXPECTED_DIMENSION))
        vectors[0, 0] = 2.0
        with self.assertRaisesRegex(RuntimeError, "not unit-normalized"):
            embed_text("query", encoder=FakeEncoder(vectors=vectors))

    def test_encoder_errors_are_preserved(self):
        class Broken(FakeEncoder):
            def encode(self, values, **kwargs):
                raise ValueError("private encoder detail")

        with self.assertRaisesRegex(ValueError, "private encoder detail") as caught:
            embed_text("query", encoder=Broken())


if __name__ == "__main__":
    unittest.main()
