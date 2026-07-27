"""
Episodic Memory

Stores full task experiences — task description, agents involved,
tools used, results, errors, and evaluation scores.  Backed by
PostgreSQL in production; uses an in-memory store for testing and
development.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog

from app.memory.base import BaseMemory
from app.models.memory import (
    EpisodicMemoryEntry,
    MemoryEntry,
    MemorySearchQuery,
    MemorySearchResult,
    MemoryType,
)


class EpisodicMemory(BaseMemory):
    """
    Experience memory for completed task episodes.

    In production this would persist to PostgreSQL via SQLAlchemy.
    The in-memory fallback is always used for correctness and fast
    unit testing.

    Parameters
    ----------
    name : str
        Human-readable name for this memory instance.
    database_url : str | None
        PostgreSQL connection string.  ``None`` for in-memory mode.
    """

    def __init__(
        self,
        name: str = "episodic",
        database_url: Optional[str] = None,
    ) -> None:
        super().__init__(memory_type=MemoryType.EPISODIC, name=name)
        self.database_url = database_url
        self._store: Dict[UUID, EpisodicMemoryEntry] = {}
        self.logger.info(
            "Episodic memory created",
            has_database=bool(database_url),
        )

    async def add(self, entry: MemoryEntry) -> UUID:
        """Add a generic memory entry (wraps into an episodic record)."""
        episode = EpisodicMemoryEntry(
            id=entry.id,
            task_description=entry.content,
            agent_id=entry.agent_id,
            metadata=entry.metadata,
            created_at=entry.created_at,
        )
        self._store[episode.id] = episode
        self.logger.debug("Generic entry added as episodic", entry_id=str(entry.id))
        return episode.id

    async def add_episode(self, episode: EpisodicMemoryEntry) -> UUID:
        """Add a fully-formed episodic memory entry."""
        self._store[episode.id] = episode
        self.logger.debug(
            "Episode added",
            episode_id=str(episode.id),
            task=episode.task_description[:80],
            score=episode.evaluation_score,
        )
        return episode.id

    async def search(self, query: MemorySearchQuery) -> List[MemorySearchResult]:
        """Search episodes by substring match on task description and result."""
        results: List[MemorySearchResult] = []
        for episode in reversed(self._store.values()):
            if query.agent_id and episode.agent_id != query.agent_id:
                continue

            searchable = (
                f"{episode.task_description} {episode.result_summary} "
                f"{' '.join(episode.errors)}"
            )
            if query.query.lower() in searchable.lower():
                entry = MemoryEntry(
                    id=episode.id,
                    type=MemoryType.EPISODIC,
                    content=episode.task_description,
                    agent_id=episode.agent_id,
                    metadata=episode.metadata,
                    created_at=episode.created_at,
                )
                score = min(1.0, 0.5 + episode.evaluation_score * 0.5)
                results.append(MemorySearchResult(memory=entry, score=score))

            if len(results) >= query.limit:
                break

        return results

    async def get(self, memory_id: UUID) -> Optional[MemoryEntry]:
        """Retrieve a generic MemoryEntry for the given episode ID."""
        episode = self._store.get(memory_id)
        if episode is None:
            return None
        return MemoryEntry(
            id=episode.id,
            type=MemoryType.EPISODIC,
            content=episode.task_description,
            agent_id=episode.agent_id,
            metadata=episode.metadata,
            created_at=episode.created_at,
        )

    async def get_episode(self, episode_id: UUID) -> Optional[EpisodicMemoryEntry]:
        """Retrieve the full episodic entry by ID."""
        return self._store.get(episode_id)

    async def update(self, entry: MemoryEntry) -> bool:
        """Update an episode by replacing its generic fields."""
        if entry.id not in self._store:
            return False
        episode = self._store[entry.id]
        episode.task_description = entry.content
        episode.agent_id = entry.agent_id
        episode.metadata = entry.metadata
        return True

    async def update_episode(self, episode: EpisodicMemoryEntry) -> bool:
        """Replace a full episode."""
        if episode.id not in self._store:
            return False
        self._store[episode.id] = episode
        return True

    async def delete(self, memory_id: UUID) -> bool:
        """Delete an episode by ID."""
        if memory_id in self._store:
            del self._store[memory_id]
            return True
        return False

    async def clear(self, agent_id: Optional[UUID] = None) -> int:
        """Clear episodes, optionally filtered by agent."""
        if agent_id is None:
            count = len(self._store)
            self._store.clear()
            return count
        to_remove = [
            eid for eid, ep in self._store.items()
            if ep.agent_id == agent_id
        ]
        for eid in to_remove:
            del self._store[eid]
        return len(to_remove)

    async def get_recent_episodes(
        self,
        limit: int = 10,
        agent_id: Optional[UUID] = None,
    ) -> List[EpisodicMemoryEntry]:
        """Return the most recent episodes (newest first)."""
        episodes = list(reversed(self._store.values()))
        if agent_id is not None:
            episodes = [e for e in episodes if e.agent_id == agent_id]
        return episodes[:limit]

    async def get_successful_episodes(
        self,
        min_score: float = 0.7,
        limit: int = 10,
    ) -> List[EpisodicMemoryEntry]:
        """Return episodes with an evaluation score >= min_score."""
        episodes = [
            e for e in self._store.values()
            if e.evaluation_score >= min_score
        ]
        episodes.sort(key=lambda e: e.evaluation_score, reverse=True)
        return episodes[:limit]

    def __len__(self) -> int:
        return len(self._store)
