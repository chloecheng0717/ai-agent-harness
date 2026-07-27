"""
Unit Tests for ShortTermMemory

Tests cover:
- Add and retrieve entries
- Sliding window eviction
- Search by content
- Search with filters (agent_id, task_id, type)
- Update and delete
- Clear with and without agent filter
- get_history
- Health check
"""

import pytest
from uuid import uuid4

from app.memory.short_term import ShortTermMemory
from app.models.memory import MemoryEntry, MemorySearchQuery, MemoryType


@pytest.fixture
def memory() -> ShortTermMemory:
    return ShortTermMemory(name="test_stm", max_size=5)


class TestShortTermMemoryAdd:
    @pytest.mark.asyncio
    async def test_add_returns_uuid(self, memory: ShortTermMemory) -> None:
        entry = MemoryEntry(type=MemoryType.SHORT_TERM, content="hello")
        entry_id = await memory.add(entry)
        assert entry_id == entry.id

    @pytest.mark.asyncio
    async def test_add_increases_size(self, memory: ShortTermMemory) -> None:
        assert len(memory) == 0
        await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content="a"))
        assert len(memory) == 1
        await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content="b"))
        assert len(memory) == 2

    @pytest.mark.asyncio
    async def test_eviction_when_full(self, memory: ShortTermMemory) -> None:
        for i in range(5):
            await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content=f"msg {i}"))
        assert len(memory) == 5

        await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content="msg 5"))
        assert len(memory) == 5

        oldest = await memory.get(
            (await memory.add.__wrapped__(None)) if False else uuid4()
        )
        entries = memory.get_history(limit=10)
        contents = [e.content for e in entries]
        assert "msg 0" not in contents
        assert "msg 5" in contents


class TestShortTermMemoryGet:
    @pytest.mark.asyncio
    async def test_get_existing(self, memory: ShortTermMemory) -> None:
        entry = MemoryEntry(type=MemoryType.SHORT_TERM, content="find me")
        await memory.add(entry)
        result = await memory.get(entry.id)
        assert result is not None
        assert result.content == "find me"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, memory: ShortTermMemory) -> None:
        result = await memory.get(uuid4())
        assert result is None


class TestShortTermMemorySearch:
    @pytest.mark.asyncio
    async def test_search_by_content(self, memory: ShortTermMemory) -> None:
        await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content="Python programming"))
        await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content="Java development"))
        await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content="Python testing"))

        query = MemorySearchQuery(query="Python", limit=10)
        results = await memory.search(query)
        assert len(results) == 2
        for r in results:
            assert "Python" in r.memory.content

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, memory: ShortTermMemory) -> None:
        await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content="HELLO world"))
        query = MemorySearchQuery(query="hello", limit=10)
        results = await memory.search(query)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_with_type_filter(self, memory: ShortTermMemory) -> None:
        await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content="test"))
        await memory.add(MemoryEntry(type=MemoryType.LONG_TERM, content="test knowledge"))

        query = MemorySearchQuery(query="test", type=MemoryType.SHORT_TERM)
        results = await memory.search(query)
        assert len(results) == 1
        assert results[0].memory.type == MemoryType.SHORT_TERM

    @pytest.mark.asyncio
    async def test_search_with_agent_filter(self, memory: ShortTermMemory) -> None:
        agent_id = uuid4()
        await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content="a", agent_id=agent_id))
        await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content="b"))

        query = MemorySearchQuery(query="a", agent_id=agent_id)
        results = await memory.search(query)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_with_task_filter(self, memory: ShortTermMemory) -> None:
        task_id = uuid4()
        await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content="task result", task_id=task_id))
        await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content="other"))

        query = MemorySearchQuery(query="task", task_id=task_id)
        results = await memory.search(query)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, memory: ShortTermMemory) -> None:
        for i in range(10):
            await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content=f"item {i}"))

        query = MemorySearchQuery(query="item", limit=3)
        results = await memory.search(query)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_no_match(self, memory: ShortTermMemory) -> None:
        await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content="hello"))
        query = MemorySearchQuery(query="xyz")
        results = await memory.search(query)
        assert len(results) == 0


class TestShortTermMemoryUpdate:
    @pytest.mark.asyncio
    async def test_update_existing(self, memory: ShortTermMemory) -> None:
        entry = MemoryEntry(type=MemoryType.SHORT_TERM, content="original")
        await memory.add(entry)

        entry.content = "updated"
        success = await memory.update(entry)
        assert success is True

        retrieved = await memory.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == "updated"

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, memory: ShortTermMemory) -> None:
        entry = MemoryEntry(type=MemoryType.SHORT_TERM, content="ghost")
        success = await memory.update(entry)
        assert success is False


class TestShortTermMemoryDelete:
    @pytest.mark.asyncio
    async def test_delete_existing(self, memory: ShortTermMemory) -> None:
        entry = MemoryEntry(type=MemoryType.SHORT_TERM, content="bye")
        await memory.add(entry)
        success = await memory.delete(entry.id)
        assert success is True
        assert await memory.get(entry.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory: ShortTermMemory) -> None:
        success = await memory.delete(uuid4())
        assert success is False


class TestShortTermMemoryClear:
    @pytest.mark.asyncio
    async def test_clear_all(self, memory: ShortTermMemory) -> None:
        for i in range(3):
            await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content=f"msg {i}"))
        count = await memory.clear()
        assert count == 3
        assert len(memory) == 0

    @pytest.mark.asyncio
    async def test_clear_by_agent(self, memory: ShortTermMemory) -> None:
        agent_id = uuid4()
        await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content="a", agent_id=agent_id))
        await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content="b"))
        await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content="c", agent_id=agent_id))

        count = await memory.clear(agent_id=agent_id)
        assert count == 2
        assert len(memory) == 1


class TestShortTermMemoryHistory:
    @pytest.mark.asyncio
    async def test_get_history_newest_first(self, memory: ShortTermMemory) -> None:
        for i in range(3):
            await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content=f"msg {i}"))
        history = memory.get_history(limit=10)
        assert len(history) == 3
        assert history[0].content == "msg 2"
        assert history[2].content == "msg 0"

    @pytest.mark.asyncio
    async def test_get_history_respects_limit(self, memory: ShortTermMemory) -> None:
        for i in range(5):
            await memory.add(MemoryEntry(type=MemoryType.SHORT_TERM, content=f"msg {i}"))
        history = memory.get_history(limit=2)
        assert len(history) == 2


class TestShortTermMemoryHealth:
    @pytest.mark.asyncio
    async def test_health_check(self, memory: ShortTermMemory) -> None:
        assert await memory.health_check() is True
