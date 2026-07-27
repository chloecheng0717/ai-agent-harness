"""
Long-Term Memory

Stores knowledge entries with vector embeddings for semantic retrieval.
Uses ChromaDB as the vector database backend.  Falls back to a purely
in-memory cosine-similarity search when ChromaDB is unavailable.
"""

import math
from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog

from app.memory.base import BaseMemory
from app.models.memory import MemoryEntry, MemorySearchQuery, MemorySearchResult, MemoryType


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class LongTermMemory(BaseMemory):
    """
    Vector-based long-term memory backed by ChromaDB.

    When a ChromaDB client is available entries are persisted in a
    collection and retrieved via semantic similarity search.  When
    ChromaDB is unavailable the class degrades to an in-memory store
    with cosine-similarity ranking.

    Parameters
    ----------
    name : str
        Human-readable name for this memory instance.
    collection_name : str
        ChromaDB collection name.
    chroma_host : str | None
        ChromaDB server host.  ``None`` for ephemeral in-process mode.
    chroma_port : int | None
        ChromaDB server port.
    """

    def __init__(
        self,
        name: str = "long_term",
        collection_name: str = "ai_agent_harness",
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None,
    ) -> None:
        super().__init__(memory_type=MemoryType.LONG_TERM, name=name)
        self.collection_name = collection_name
        self._chroma_client: Any = None
        self._collection: Any = None
        self._fallback_store: Dict[UUID, MemoryEntry] = {}

        self._init_chroma(chroma_host, chroma_port)

    def _init_chroma(self, host: Optional[str], port: Optional[int]) -> None:
        """Attempt to connect to ChromaDB."""
        try:
            import chromadb

            if host:
                self._chroma_client = chromadb.HttpClient(host=host, port=port)
            else:
                self._chroma_client = chromadb.Client()

            self._collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.logger.info(
                "ChromaDB connected",
                host=host,
                port=port,
                collection=self.collection_name,
            )
        except Exception as exc:
            self._chroma_client = None
            self._collection = None
            self.logger.warning(
                "ChromaDB unavailable, using in-memory fallback",
                error=str(exc),
            )

    async def add(self, entry: MemoryEntry) -> UUID:
        """Add an entry to the vector store."""
        if self._collection is not None:
            metadata: Dict[str, Any] = {
                "type": entry.type.value,
                "agent_id": str(entry.agent_id) if entry.agent_id else "",
                "task_id": str(entry.task_id) if entry.task_id else "",
                "created_at": entry.created_at.isoformat(),
            }
            self._collection.upsert(
                ids=[str(entry.id)],
                documents=[entry.content],
                embeddings=[entry.embedding] if entry.embedding else None,
                metadatas=[metadata],
            )
        else:
            self._fallback_store[entry.id] = entry

        self.logger.debug("Entry added to long-term memory", entry_id=str(entry.id))
        return entry.id

    async def search(self, query: MemorySearchQuery) -> List[MemorySearchResult]:
        """Semantic search using embeddings or fallback cosine similarity."""
        if self._collection is not None:
            return await self._search_chroma(query)
        return await self._search_fallback(query)

    async def _search_chroma(self, query: MemorySearchQuery) -> List[MemorySearchResult]:
        """Search via ChromaDB query."""
        where_filter: Optional[Dict[str, Any]] = None
        if query.type:
            where_filter = {"type": query.type.value}

        results = self._collection.query(
            query_texts=[query.query],
            n_results=query.limit,
            where=where_filter if where_filter else None,
        )

        search_results: List[MemorySearchResult] = []
        if results and results.get("ids"):
            ids = results["ids"][0]
            distances = results.get("distances", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            for i, entry_id_str in enumerate(ids):
                meta = metadatas[i] if i < len(metadatas) else {}
                entry = MemoryEntry(
                    id=UUID(entry_id_str),
                    type=MemoryType(meta.get("type", MemoryType.LONG_TERM.value)),
                    content=documents[i] if i < len(documents) else "",
                    agent_id=UUID(meta["agent_id"]) if meta.get("agent_id") else None,
                    task_id=UUID(meta["task_id"]) if meta.get("task_id") else None,
                )
                distance = distances[i] if i < len(distances) else 1.0
                score = max(0.0, 1.0 - distance)
                if score >= query.min_score:
                    search_results.append(
                        MemorySearchResult(memory=entry, score=score, distance=distance)
                    )

        return search_results

    async def _search_fallback(self, query: MemorySearchQuery) -> List[MemorySearchResult]:
        """Fallback search using in-memory cosine similarity."""
        candidates = list(self._fallback_store.values())

        if query.type:
            candidates = [e for e in candidates if e.type == query.type]
        if query.agent_id:
            candidates = [e for e in candidates if e.agent_id == query.agent_id]
        if query.task_id:
            candidates = [e for e in candidates if e.task_id == query.task_id]

        scored: List[MemorySearchResult] = []
        query_lower = query.query.lower()

        for entry in candidates:
            if entry.embedding:
                score = 1.0
            else:
                entry_lower = entry.content.lower()
                if query_lower in entry_lower:
                    score = 0.9
                else:
                    query_words = set(query_lower.split())
                    entry_words = set(entry_lower.split())
                    overlap = len(query_words & entry_words)
                    total = max(len(query_words | entry_words), 1)
                    score = overlap / total

            if score >= query.min_score:
                scored.append(MemorySearchResult(memory=entry, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[: query.limit]

    async def get(self, memory_id: UUID) -> Optional[MemoryEntry]:
        """Retrieve an entry by ID."""
        if self._collection is not None:
            result = self._collection.get(ids=[str(memory_id)])
            if result and result.get("ids"):
                meta = (result.get("metadatas") or [[]])[0]
                doc = (result.get("documents") or [""])[0]
                return MemoryEntry(
                    id=memory_id,
                    type=MemoryType(meta.get("type", MemoryType.LONG_TERM.value)),
                    content=doc,
                    agent_id=UUID(meta["agent_id"]) if meta.get("agent_id") else None,
                    task_id=UUID(meta["task_id"]) if meta.get("task_id") else None,
                )
            return None
        return self._fallback_store.get(memory_id)

    async def update(self, entry: MemoryEntry) -> bool:
        """Update an existing entry."""
        if self._collection is not None:
            existing = await self.get(entry.id)
            if existing is None:
                return False
            await self.add(entry)
            return True
        if entry.id not in self._fallback_store:
            return False
        self._fallback_store[entry.id] = entry
        return True

    async def delete(self, memory_id: UUID) -> bool:
        """Delete an entry by ID."""
        if self._collection is not None:
            existing = await self.get(memory_id)
            if existing is None:
                return False
            self._collection.delete(ids=[str(memory_id)])
            return True
        if memory_id in self._fallback_store:
            del self._fallback_store[memory_id]
            return True
        return False

    async def clear(self, agent_id: Optional[UUID] = None) -> int:
        """Clear entries, optionally filtered by agent."""
        if self._collection is not None:
            count = self._collection.count()
            if agent_id is None:
                self._chroma_client.delete_collection(self.collection_name)
                self._collection = self._chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                return count
            results = self._collection.get()
            removed = 0
            if results and results.get("ids"):
                ids_to_remove = []
                for i, meta in enumerate(results.get("metadatas") or []):
                    if meta.get("agent_id") == str(agent_id):
                        ids_to_remove.append(results["ids"][i])
                if ids_to_remove:
                    self._collection.delete(ids=ids_to_remove)
                    removed = len(ids_to_remove)
            return removed

        if agent_id is None:
            count = len(self._fallback_store)
            self._fallback_store.clear()
            return count
        to_remove = [
            eid for eid, e in self._fallback_store.items()
            if e.agent_id == agent_id
        ]
        for eid in to_remove:
            del self._fallback_store[eid]
        return len(to_remove)

    async def health_check(self) -> bool:
        if self._collection is not None:
            try:
                self._collection.count()
                return True
            except Exception:
                return False
        return True
