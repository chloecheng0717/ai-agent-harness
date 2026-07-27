"""
Unit Tests for MemoryManager

Tests cover:
- Store, retrieve, search, delete across backends
- Store episodic memory
- Summarize memory
- Clear memory
- Health check
- Disabled mode (all ops are no-ops)
"""

import pytest
from uuid import UUID, uuid4

from app.memory.manager import MemoryManager
from app.models.memory import (
    EpisodicMemoryEntry,
    MemoryType,
)


@pytest.fixture
def manager() -> MemoryManager:
    return MemoryManager(enabled=True)


@pytest.fixture
def disabled_manager() -> MemoryManager:
    return MemoryManager(enabled=False)


class TestMemoryManagerStore:
    @pytest.mark.asyncio
    async def test_store_short_term(self, manager: MemoryManager) -> None:
        entry_id = await manager.store_memory(
            content="Hello world",
            memory_type=MemoryType.SHORT_TERM,
        )
        assert entry_id is not None
        assert entry_id != UUID(int=0)

    @pytest.mark.asyncio
    async def test_store_long_term(self, manager: MemoryManager) -> None:
        entry_id = await manager.store_memory(
            content="Knowledge about Python",
            memory_type=MemoryType.LONG_TERM,
        )
        assert entry_id is not None

    @pytest.mark.asyncio
    async def test_store_episodic(self, manager: MemoryManager) -> None:
        entry_id = await manager.store_memory(
            content="Task completed successfully",
            memory_type=MemoryType.EPISODIC,
        )
        assert entry_id is not None

    @pytest.mark.asyncio
    async def test_store_with_metadata(self, manager: MemoryManager) -> None:
        agent_id = uuid4()
        entry_id = await manager.store_memory(
            content="Test",
            memory_type=MemoryType.SHORT_TERM,
            agent_id=agent_id,
            metadata={"key": "value"},
        )
        entry = await manager.retrieve_memory(entry_id, MemoryType.SHORT_TERM)
        assert entry is not None
        assert entry.agent_id == agent_id
        assert entry.metadata == {"key": "value"}


class TestMemoryManagerStoreEpisode:
    @pytest.mark.asyncio
    async def test_store_episode(self, manager: MemoryManager) -> None:
        episode = EpisodicMemoryEntry(
            task_description="Build feature X",
            agents_involved=["coder", "qa"],
            evaluation_score=0.85,
        )
        episode_id = await manager.store_episode(episode)
        assert episode_id is not None

        retrieved = await manager.retrieve_episode(episode_id)
        assert retrieved is not None
        assert retrieved.task_description == "Build feature X"
        assert retrieved.evaluation_score == 0.85


class TestMemoryManagerRetrieve:
    @pytest.mark.asyncio
    async def test_retrieve_from_short_term(self, manager: MemoryManager) -> None:
        entry_id = await manager.store_memory(
            content="Findable",
            memory_type=MemoryType.SHORT_TERM,
        )
        entry = await manager.retrieve_memory(entry_id, MemoryType.SHORT_TERM)
        assert entry is not None
        assert entry.content == "Findable"

    @pytest.mark.asyncio
    async def test_retrieve_from_long_term(self, manager: MemoryManager) -> None:
        entry_id = await manager.store_memory(
            content="Knowledge",
            memory_type=MemoryType.LONG_TERM,
        )
        entry = await manager.retrieve_memory(entry_id, MemoryType.LONG_TERM)
        assert entry is not None
        assert entry.content == "Knowledge"

    @pytest.mark.asyncio
    async def test_retrieve_from_episodic(self, manager: MemoryManager) -> None:
        entry_id = await manager.store_memory(
            content="Episode",
            memory_type=MemoryType.EPISODIC,
        )
        entry = await manager.retrieve_memory(entry_id, MemoryType.EPISODIC)
        assert entry is not None
        assert entry.content == "Episode"

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent(self, manager: MemoryManager) -> None:
        entry = await manager.retrieve_memory(uuid4())
        assert entry is None

    @pytest.mark.asyncio
    async def test_retrieve_cross_backend(self, manager: MemoryManager) -> None:
        entry_id = await manager.store_memory(
            content="Stored in ST",
            memory_type=MemoryType.SHORT_TERM,
        )
        entry = await manager.retrieve_memory(entry_id)
        assert entry is not None
        assert entry.content == "Stored in ST"


class TestMemoryManagerSearch:
    @pytest.mark.asyncio
    async def test_search_short_term(self, manager: MemoryManager) -> None:
        await manager.store_memory(
            content="Python programming tips",
            memory_type=MemoryType.SHORT_TERM,
        )
        results = await manager.search_memory(
            query="Python",
            memory_type=MemoryType.SHORT_TERM,
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_all_backends(self, manager: MemoryManager) -> None:
        await manager.store_memory(
            content="Python basics",
            memory_type=MemoryType.SHORT_TERM,
        )
        await manager.store_memory(
            content="Python advanced",
            memory_type=MemoryType.LONG_TERM,
        )
        results = await manager.search_memory(query="Python")
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, manager: MemoryManager) -> None:
        for i in range(5):
            await manager.store_memory(
                content=f"Item {i} about testing",
                memory_type=MemoryType.SHORT_TERM,
            )
        results = await manager.search_memory(query="testing", limit=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_empty(self, manager: MemoryManager) -> None:
        results = await manager.search_memory(query="nothing")
        assert len(results) == 0


class TestMemoryManagerDelete:
    @pytest.mark.asyncio
    async def test_delete_from_specific_backend(self, manager: MemoryManager) -> None:
        entry_id = await manager.store_memory(
            content="Delete me",
            memory_type=MemoryType.SHORT_TERM,
        )
        success = await manager.delete_memory(entry_id, MemoryType.SHORT_TERM)
        assert success is True

        entry = await manager.retrieve_memory(entry_id, MemoryType.SHORT_TERM)
        assert entry is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, manager: MemoryManager) -> None:
        success = await manager.delete_memory(uuid4())
        assert success is False


class TestMemoryManagerSummarize:
    @pytest.mark.asyncio
    async def test_summarize_empty(self, manager: MemoryManager) -> None:
        summary = await manager.summarize_memory()
        assert summary["total"] == 0
        assert summary["short_term"] == 0
        assert summary["long_term"] == 0
        assert summary["episodic"] == 0

    @pytest.mark.asyncio
    async def test_summarize_with_data(self, manager: MemoryManager) -> None:
        await manager.store_memory("a", memory_type=MemoryType.SHORT_TERM)
        await manager.store_memory("b", memory_type=MemoryType.SHORT_TERM)
        await manager.store_memory("c", memory_type=MemoryType.LONG_TERM)

        summary = await manager.summarize_memory()
        assert summary["short_term"] == 2
        assert summary["long_term"] == 1
        assert summary["total"] == 3


class TestMemoryManagerClear:
    @pytest.mark.asyncio
    async def test_clear_all(self, manager: MemoryManager) -> None:
        await manager.store_memory("a", memory_type=MemoryType.SHORT_TERM)
        await manager.store_memory("b", memory_type=MemoryType.LONG_TERM)
        count = await manager.clear_memory()
        assert count >= 2

    @pytest.mark.asyncio
    async def test_clear_specific_type(self, manager: MemoryManager) -> None:
        await manager.store_memory("a", memory_type=MemoryType.SHORT_TERM)
        await manager.store_memory("b", memory_type=MemoryType.SHORT_TERM)
        count = await manager.clear_memory(memory_type=MemoryType.SHORT_TERM)
        assert count == 2


class TestMemoryManagerHealth:
    @pytest.mark.asyncio
    async def test_health_check(self, manager: MemoryManager) -> None:
        health = await manager.health_check()
        assert health["short_term"] is True
        assert health["long_term"] is True
        assert health["episodic"] is True


class TestMemoryManagerDisabled:
    @pytest.mark.asyncio
    async def test_disabled_store_returns_zero_id(self, disabled_manager: MemoryManager) -> None:
        entry_id = await disabled_manager.store_memory(
            content="test", memory_type=MemoryType.SHORT_TERM
        )
        assert entry_id == UUID(int=0)

    @pytest.mark.asyncio
    async def test_disabled_retrieve_returns_none(self, disabled_manager: MemoryManager) -> None:
        entry = await disabled_manager.retrieve_memory(uuid4())
        assert entry is None

    @pytest.mark.asyncio
    async def test_disabled_search_returns_empty(self, disabled_manager: MemoryManager) -> None:
        results = await disabled_manager.search_memory(query="test")
        assert results == []

    @pytest.mark.asyncio
    async def test_disabled_delete_returns_false(self, disabled_manager: MemoryManager) -> None:
        success = await disabled_manager.delete_memory(uuid4())
        assert success is False

    @pytest.mark.asyncio
    async def test_disabled_clear_returns_zero(self, disabled_manager: MemoryManager) -> None:
        count = await disabled_manager.clear_memory()
        assert count == 0

    @pytest.mark.asyncio
    async def test_disabled_summarize(self, disabled_manager: MemoryManager) -> None:
        summary = await disabled_manager.summarize_memory()
        assert summary["total"] == 0
