"""
Unit Tests for LongTermMemory

Tests cover:
- Add and retrieve entries
- Search by content (fallback mode)
- Search with filters
- Update and delete
- Clear
- Cosine similarity fallback
- Health check
"""

import pytest
from uuid import uuid4

from app.memory.long_term import LongTermMemory, _cosine_similarity
from app.models.memory import MemoryEntry, MemorySearchQuery, MemoryType


@pytest.fixture
def memory() -> LongTermMemory:
    return LongTermMemory(name="test_ltm", collection_name="test_collection")


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_empty_vectors(self) -> None:
        assert _cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self) -> None:
        assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestLongTermMemoryAdd:
    @pytest.mark.asyncio
    async def test_add_returns_uuid(self, memory: LongTermMemory) -> None:
        entry = MemoryEntry(type=MemoryType.LONG_TERM, content="knowledge")
        entry_id = await memory.add(entry)
        assert entry_id == entry.id

    @pytest.mark.asyncio
    async def test_add_stores_in_fallback(self, memory: LongTermMemory) -> None:
        entry = MemoryEntry(type=MemoryType.LONG_TERM, content="stored")
        await memory.add(entry)
        assert entry.id in memory._fallback_store


class TestLongTermMemoryGet:
    @pytest.mark.asyncio
    async def test_get_existing(self, memory: LongTermMemory) -> None:
        entry = MemoryEntry(type=MemoryType.LONG_TERM, content="find me")
        await memory.add(entry)
        result = await memory.get(entry.id)
        assert result is not None
        assert result.content == "find me"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, memory: LongTermMemory) -> None:
        result = await memory.get(uuid4())
        assert result is None


class TestLongTermMemorySearch:
    @pytest.mark.asyncio
    async def test_search_exact_match(self, memory: LongTermMemory) -> None:
        await memory.add(MemoryEntry(type=MemoryType.LONG_TERM, content="Python machine learning"))
        await memory.add(MemoryEntry(type=MemoryType.LONG_TERM, content="Java web development"))

        query = MemorySearchQuery(query="Python machine learning", min_score=0.0)
        results = await memory.search(query)
        assert len(results) >= 1
        assert any("Python" in r.memory.content for r in results)

    @pytest.mark.asyncio
    async def test_search_partial_match(self, memory: LongTermMemory) -> None:
        await memory.add(MemoryEntry(type=MemoryType.LONG_TERM, content="Python is great"))
        await memory.add(MemoryEntry(type=MemoryType.LONG_TERM, content="Java is okay"))

        query = MemorySearchQuery(query="Python", min_score=0.0)
        results = await memory.search(query)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_with_type_filter(self, memory: LongTermMemory) -> None:
        await memory.add(MemoryEntry(type=MemoryType.LONG_TERM, content="knowledge"))
        await memory.add(MemoryEntry(type=MemoryType.EPISODIC, content="knowledge"))

        query = MemorySearchQuery(
            query="knowledge",
            type=MemoryType.LONG_TERM,
            min_score=0.0,
        )
        results = await memory.search(query)
        assert len(results) == 1
        assert results[0].memory.type == MemoryType.LONG_TERM

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, memory: LongTermMemory) -> None:
        for i in range(10):
            await memory.add(
                MemoryEntry(type=MemoryType.LONG_TERM, content=f"item {i} knowledge")
            )

        query = MemorySearchQuery(query="item knowledge", limit=3, min_score=0.0)
        results = await memory.search(query)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_empty(self, memory: LongTermMemory) -> None:
        query = MemorySearchQuery(query="nothing here")
        results = await memory.search(query)
        assert len(results) == 0


class TestLongTermMemoryUpdate:
    @pytest.mark.asyncio
    async def test_update_existing(self, memory: LongTermMemory) -> None:
        entry = MemoryEntry(type=MemoryType.LONG_TERM, content="original")
        await memory.add(entry)

        entry.content = "updated"
        success = await memory.update(entry)
        assert success is True

        retrieved = await memory.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == "updated"

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, memory: LongTermMemory) -> None:
        entry = MemoryEntry(type=MemoryType.LONG_TERM, content="ghost")
        success = await memory.update(entry)
        assert success is False


class TestLongTermMemoryDelete:
    @pytest.mark.asyncio
    async def test_delete_existing(self, memory: LongTermMemory) -> None:
        entry = MemoryEntry(type=MemoryType.LONG_TERM, content="bye")
        await memory.add(entry)
        success = await memory.delete(entry.id)
        assert success is True
        assert await memory.get(entry.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory: LongTermMemory) -> None:
        success = await memory.delete(uuid4())
        assert success is False


class TestLongTermMemoryClear:
    @pytest.mark.asyncio
    async def test_clear_all(self, memory: LongTermMemory) -> None:
        for i in range(3):
            await memory.add(
                MemoryEntry(type=MemoryType.LONG_TERM, content=f"knowledge {i}")
            )
        count = await memory.clear()
        assert count == 3

    @pytest.mark.asyncio
    async def test_clear_by_agent(self, memory: LongTermMemory) -> None:
        agent_id = uuid4()
        await memory.add(
            MemoryEntry(type=MemoryType.LONG_TERM, content="a", agent_id=agent_id)
        )
        await memory.add(MemoryEntry(type=MemoryType.LONG_TERM, content="b"))

        count = await memory.clear(agent_id=agent_id)
        assert count == 1


class TestLongTermMemoryHealth:
    @pytest.mark.asyncio
    async def test_health_check(self, memory: LongTermMemory) -> None:
        assert await memory.health_check() is True
