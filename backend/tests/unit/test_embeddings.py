"""
Unit Tests for EmbeddingProvider

Tests cover:
- Hash-based fallback embeddings
- Deterministic output
- Unit vector property
- Batch embedding
- Dimension consistency
"""

import math

import pytest

from app.rag.embeddings import EmbeddingProvider


class TestEmbeddingProviderInit:
    def test_creates_without_api_key(self) -> None:
        provider = EmbeddingProvider()
        assert provider.api_key is None
        assert provider._client is None

    def test_creates_with_custom_dimension(self) -> None:
        provider = EmbeddingProvider(dimension=256)
        assert provider.dimension == 256


class TestEmbeddingProviderHash:
    def test_produces_correct_dimension(self) -> None:
        provider = EmbeddingProvider(dimension=128)
        embedding = provider.embed("test text")
        assert len(embedding) == 128

    def test_is_unit_vector(self) -> None:
        provider = EmbeddingProvider(dimension=64)
        embedding = provider.embed("hello")
        norm = math.sqrt(sum(x * x for x in embedding))
        assert abs(norm - 1.0) < 1e-6

    def test_deterministic(self) -> None:
        provider = EmbeddingProvider()
        e1 = provider.embed("same text")
        e2 = provider.embed("same text")
        assert e1 == e2

    def test_different_texts_produce_different_embeddings(self) -> None:
        provider = EmbeddingProvider(dimension=64)
        e1 = provider.embed("text one")
        e2 = provider.embed("text two")
        assert e1 != e2

    def test_values_in_range(self) -> None:
        provider = EmbeddingProvider(dimension=32)
        embedding = provider.embed("range check")
        for val in embedding:
            assert -1.0 <= val <= 1.0


class TestEmbeddingProviderBatch:
    def test_batch_embedding(self) -> None:
        provider = EmbeddingProvider(dimension=64)
        texts = ["hello", "world", "test"]
        embeddings = provider.embed_batch(texts)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 64

    def test_batch_matches_single(self) -> None:
        provider = EmbeddingProvider(dimension=64)
        single = provider.embed("compare me")
        batch = provider.embed_batch(["compare me"])
        assert single == batch[0]

    def test_empty_batch(self) -> None:
        provider = EmbeddingProvider()
        result = provider.embed_batch([])
        assert result == []
