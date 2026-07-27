"""
Short-Term Memory

Stores current conversation context and recent messages using an
in-memory sliding window.  Optionally backed by Redis for persistence
across process restarts.
"""

from collections import OrderedDict
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

import structlog

from app.memory.base import BaseMemory
from app.models.memory import MemoryEntry, MemorySearchQuery, MemorySearchResult, MemoryType


class ShortTermMemory(BaseMemory):
    """
    In-memory sliding-window short-term memory.

    Entries are stored in insertion order.  When the window is full the
    oldest entry is evicted.  An optional ``redis_url`` can be supplied
    for future Redis-backed persistence (currently unused — the in-memory
    store is always used for correctness and testability).

    Parameters
    ----------
    name : str
        Human-readable name for this memory instance.
    max_size : int
        Maximum number of entries to retain (default 100).
    redis_url : str | None
        Reserved for future Redis integration.
    """

    def __init__(
        self,
        name: str = "short_term",
        max_size: int = 100,
        redis_url: Optional[str] = None,
    ) -> None:
        super().__init__(memory_type=MemoryType.SHORT_TERM, name=name)
        self.max_size = max_size
        self.redis_url = redis_url
        self._store: OrderedDict[UUID, MemoryEntry] = OrderedDict()
        self.logger.info(
            "Short-term memory created",
            max_size=max_size,
            redis_url=bool(redis_url),
        )

    async def add(self, entry: MemoryEntry) -> UUID:
        """Add a memory entry, evicting the oldest if the window is full."""
        if len(self._store) >= self.max_size:
            evicted_id, _ = self._store.popitem(last=False)
            self.logger.debug("Evicted oldest entry", evicted_id=str(evicted_id))

        self._store[entry.id] = entry
        self.logger.debug("Entry added", entry_id=str(entry.id))
        return entry.id

    async def search(self, query: MemorySearchQuery) -> List[MemorySearchResult]:
        """
        Search entries by substring match on content.

        Returns results sorted by creation time (newest first).
        All matching entries receive a score of 1.0 since short-term
        memory does not support semantic similarity.
        """
        results: List[MemorySearchResult] = []
        for entry in reversed(self._store.values()):
            if query.type and entry.type != query.type:
                continue
            if query.agent_id and entry.agent_id != query.agent_id:
                continue
            if query.task_id and entry.task_id != query.task_id:
                continue
            if query.query.lower() in entry.content.lower():
                results.append(
                    MemorySearchResult(memory=entry, score=1.0)
                )
            if len(results) >= query.limit:
                break
        return results

    async def get(self, memory_id: UUID) -> Optional[MemoryEntry]:
        """Retrieve a single entry by ID."""
        return self._store.get(memory_id)

    async def update(self, entry: MemoryEntry) -> bool:
        """Replace an existing entry (identified by ``entry.id``)."""
        if entry.id not in self._store:
            return False
        self._store[entry.id] = entry
        return True

    async def delete(self, memory_id: UUID) -> bool:
        """Remove an entry by ID."""
        if memory_id in self._store:
            del self._store[memory_id]
            return True
        return False

    async def clear(self, agent_id: Optional[UUID] = None) -> int:
        """
        Clear entries, optionally filtered by agent.

        Returns the number of entries removed.
        """
        if agent_id is None:
            count = len(self._store)
            self._store.clear()
            return count

        to_remove = [
            eid for eid, entry in self._store.items()
            if entry.agent_id == agent_id
        ]
        for eid in to_remove:
            del self._store[eid]
        return len(to_remove)

    def get_history(self, limit: int = 10) -> List[MemoryEntry]:
        """
        Return the most recent entries (newest first).

        Parameters
        ----------
        limit : int
            Maximum entries to return.
        """
        entries = list(reversed(self._store.values()))
        return entries[:limit]

    def __len__(self) -> int:
        return len(self._store)
