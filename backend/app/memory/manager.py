"""
Memory Manager

Central coordinator that exposes a unified API over the three memory
backends (short-term, long-term, episodic).  Agents interact with
memory exclusively through this manager.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog

from app.memory.episodic import EpisodicMemory
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.models.memory import (
    EpisodicMemoryEntry,
    MemoryEntry,
    MemorySearchQuery,
    MemorySearchResult,
    MemoryType,
)


class MemoryManager:
    """
    Unified memory facade used by all agents.

    Parameters
    ----------
    short_term : ShortTermMemory | None
        Short-term conversation memory.  Created with defaults when ``None``.
    long_term : LongTermMemory | None
        Vector-based long-term memory.  Created with defaults when ``None``.
    episodic : EpisodicMemory | None
        Task experience memory.  Created with defaults when ``None``.
    enabled : bool
        Master switch — when ``False``, all operations become no-ops.
    """

    def __init__(
        self,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        episodic: Optional[EpisodicMemory] = None,
        enabled: bool = True,
    ) -> None:
        self.short_term = short_term or ShortTermMemory()
        self.long_term = long_term or LongTermMemory()
        self.episodic = episodic or EpisodicMemory()
        self.enabled = enabled
        self.logger = structlog.get_logger().bind(component="memory_manager")
        self.logger.info(
            "Memory manager initialised",
            enabled=enabled,
            short_term_size=len(self.short_term),
            long_term_available=self.long_term._collection is not None,
            episodic_size=len(self.episodic),
        )

    # ── Store ────────────────────────────────────────────────────

    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        agent_id: Optional[UUID] = None,
        task_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> UUID:
        """
        Store a memory entry in the appropriate backend.

        Returns the ID of the stored entry.
        """
        if not self.enabled:
            from uuid import UUID as _UUID
            return _UUID(int=0)

        entry = MemoryEntry(
            type=memory_type,
            content=content,
            agent_id=agent_id,
            task_id=task_id,
            metadata=metadata or {},
            embedding=embedding,
        )

        backend = self._backend_for(memory_type)
        entry_id = await backend.add(entry)

        self.logger.debug(
            "Memory stored",
            entry_id=str(entry_id),
            memory_type=memory_type.value,
        )
        return entry_id

    async def store_episode(self, episode: EpisodicMemoryEntry) -> UUID:
        """Store a full episodic memory entry."""
        if not self.enabled:
            from uuid import UUID as _UUID
            return _UUID(int=0)
        return await self.episodic.add_episode(episode)

    # ── Retrieve ─────────────────────────────────────────────────

    async def retrieve_memory(
        self,
        memory_id: UUID,
        memory_type: Optional[MemoryType] = None,
    ) -> Optional[MemoryEntry]:
        """
        Retrieve a single memory by ID.

        If ``memory_type`` is given, only that backend is queried.
        Otherwise all backends are tried.
        """
        if not self.enabled:
            return None

        if memory_type:
            backend = self._backend_for(memory_type)
            return await backend.get(memory_id)

        for backend in [self.short_term, self.long_term, self.episodic]:
            result = await backend.get(memory_id)
            if result is not None:
                return result
        return None

    async def retrieve_episode(
        self, episode_id: UUID
    ) -> Optional[EpisodicMemoryEntry]:
        """Retrieve a full episodic entry by ID."""
        if not self.enabled:
            return None
        return await self.episodic.get_episode(episode_id)

    # ── Search ───────────────────────────────────────────────────

    async def search_memory(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        agent_id: Optional[UUID] = None,
        task_id: Optional[UUID] = None,
        limit: int = 10,
        min_score: float = 0.5,
    ) -> List[MemorySearchResult]:
        """
        Search across one or all memory backends.

        Results from different backends are merged and sorted by score.
        """
        if not self.enabled:
            return []

        search_query = MemorySearchQuery(
            query=query,
            type=memory_type,
            agent_id=agent_id,
            task_id=task_id,
            limit=limit,
            min_score=min_score,
        )

        if memory_type:
            backend = self._backend_for(memory_type)
            return await backend.search(search_query)

        all_results: List[MemorySearchResult] = []
        for backend in [self.short_term, self.long_term, self.episodic]:
            results = await backend.search(search_query)
            all_results.extend(results)

        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:limit]

    # ── Delete ───────────────────────────────────────────────────

    async def delete_memory(
        self,
        memory_id: UUID,
        memory_type: Optional[MemoryType] = None,
    ) -> bool:
        """Delete a memory entry by ID."""
        if not self.enabled:
            return False

        if memory_type:
            backend = self._backend_for(memory_type)
            return await backend.delete(memory_id)

        deleted = False
        for backend in [self.short_term, self.long_term, self.episodic]:
            if await backend.delete(memory_id):
                deleted = True
        return deleted

    # ── Summarise ────────────────────────────────────────────────

    async def summarize_memory(
        self,
        memory_type: Optional[MemoryType] = None,
        agent_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        """
        Produce a summary of stored memories.

        Returns counts per backend and total, plus recent entries.
        """
        if not self.enabled:
            return {"total": 0, "short_term": 0, "long_term": 0, "episodic": 0}

        st = len(self.short_term)
        lt = len(self.long_term._fallback_store) if self.long_term._collection is None else self.long_term._collection.count()
        ep = len(self.episodic)

        summary: Dict[str, Any] = {
            "short_term": st,
            "long_term": lt,
            "episodic": ep,
            "total": st + lt + ep,
        }

        recent = await self.search_memory(
            query="",
            memory_type=memory_type,
            agent_id=agent_id,
            limit=5,
            min_score=0.0,
        )
        summary["recent_count"] = len(recent)

        self.logger.debug("Memory summary", **summary)
        return summary

    # ── Clear ────────────────────────────────────────────────────

    async def clear_memory(
        self,
        memory_type: Optional[MemoryType] = None,
        agent_id: Optional[UUID] = None,
    ) -> int:
        """Clear memories, optionally filtered by type and/or agent."""
        if not self.enabled:
            return 0

        if memory_type:
            backend = self._backend_for(memory_type)
            return await backend.clear(agent_id)

        total = 0
        for backend in [self.short_term, self.long_term, self.episodic]:
            total += await backend.clear(agent_id)
        return total

    # ── Health ───────────────────────────────────────────────────

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all backends."""
        return {
            "short_term": await self.short_term.health_check(),
            "long_term": await self.long_term.health_check(),
            "episodic": await self.episodic.health_check(),
        }

    # ── Internal ─────────────────────────────────────────────────

    def _backend_for(self, memory_type: MemoryType):
        if memory_type == MemoryType.SHORT_TERM:
            return self.short_term
        elif memory_type == MemoryType.LONG_TERM:
            return self.long_term
        elif memory_type == MemoryType.EPISODIC:
            return self.episodic
        raise ValueError(f"Unknown memory type: {memory_type}")
