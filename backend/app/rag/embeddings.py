"""
Embedding Provider

Generates vector embeddings for text.  Supports OpenAI embeddings
with a deterministic hash-based fallback for testing / offline use.
"""

import hashlib
import math
from typing import List, Optional

import structlog

logger = structlog.get_logger().bind(component="embedding_provider")

_DEFAULT_DIMENSION = 1536


class EmbeddingProvider:
    """
    Generates embeddings for text strings.

    When an OpenAI API key is configured the ``text-embedding-ada-002``
    model is used.  Otherwise a deterministic pseudo-embedding is
    generated from a SHA-256 hash so that similarity search can still
    be tested without external services.

    Parameters
    ----------
    api_key : str | None
        OpenAI-compatible API key.  ``None`` triggers hash fallback.
    api_base : str
        Base URL for the OpenAI-compatible API.
    model : str
        Embedding model name.
    dimension : int
        Expected embedding dimensionality.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://api.openai.com/v1",
        model: str = "text-embedding-ada-002",
        dimension: int = _DEFAULT_DIMENSION,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.dimension = dimension
        self._client = None

        if api_key:
            self._init_client()

    def _init_client(self) -> None:
        try:
            import openai

            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
            logger.info("OpenAI embedding client initialised", model=self.model)
        except Exception as exc:
            logger.warning("Failed to init OpenAI client, using fallback", error=str(exc))
            self._client = None

    def embed(self, text: str) -> List[float]:
        """Generate an embedding for a single text string."""
        if self._client is not None:
            return self._embed_openai(text)
        return self._embed_hash(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of text strings."""
        if self._client is not None and len(texts) > 0:
            return self._embed_openai_batch(texts)
        return [self._embed_hash(t) for t in texts]

    # ── OpenAI ────────────────────────────────────────────────

    def _embed_openai(self, text: str) -> List[float]:
        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as exc:
            logger.warning("OpenAI embedding failed, using fallback", error=str(exc))
            return self._embed_hash(text)

    def _embed_openai_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            logger.warning("OpenAI batch embedding failed, using fallback", error=str(exc))
            return [self._embed_hash(t) for t in texts]

    # ── Hash fallback ─────────────────────────────────────────

    def _embed_hash(self, text: str) -> List[float]:
        """
        Generate a deterministic pseudo-embedding from a SHA-256 hash.

        Produces a unit vector of the configured dimension so that
        cosine-similarity search still works in a limited fashion.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        raw: List[float] = []
        for i in range(self.dimension):
            byte_pair = digest[(i * 2) % len(digest) : (i * 2 + 2) % len(digest)]
            if len(byte_pair) < 2:
                byte_pair = digest[i % len(digest) : i % len(digest) + 2]
            raw.append((int(byte_pair, 16) / 255.0) * 2 - 1)

        norm = math.sqrt(sum(x * x for x in raw))
        if norm > 0:
            raw = [x / norm for x in raw]

        return raw
