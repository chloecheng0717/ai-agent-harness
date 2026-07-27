"""
Unit Tests for EpisodicMemory

Tests cover:
- Add generic and episodic entries
- Retrieve by ID
- Search by content
- Update and delete
- Clear with and without agent filter
- get_recent_episodes
- get_successful_episodes
- Health check
"""

import pytest
from uuid import uuid4

from app.memory.episodic import EpisodicMemory
from app.models.memory import (
    EpisodicMemoryEntry,
    MemoryEntry,
    MemorySearchQuery,
    MemoryType,
)


@pytest.fixture
def memory() -> EpisodicMemory:
    return EpisodicMemory(name="test_episodic")


class TestEpisodicMemoryAdd:
    @pytest.mark.asyncio
    async def test_add_generic_entry(self, memory: EpisodicMemory) -> None:
        entry = MemoryEntry(type=MemoryType.EPISODIC, content="task description")
        entry_id = await memory.add(entry)
        assert entry_id == entry.id

    @pytest.mark.asyncio
    async def test_add_episode(self, memory: EpisodicMemory) -> None:
        episode = EpisodicMemoryEntry(
            task_description="Build API",
            agents_involved=["coder"],
            evaluation_score=0.9,
        )
        episode_id = await memory.add_episode(episode)
        assert episode_id == episode.id
        assert len(memory) == 1

    @pytest.mark.asyncio
    async def test_add_multiple_episodes(self, memory: EpisodicMemory) -> None:
        for i in range(3):
            await memory.add_episode(
                EpisodicMemoryEntry(task_description=f"Task {i}")
            )
        assert len(memory) == 3


class TestEpisodicMemoryGet:
    @pytest.mark.asyncio
    async def test_get_generic_entry(self, memory: EpisodicMemory) -> None:
        episode = EpisodicMemoryEntry(
            task_description="Test task",
            evaluation_score=0.8,
        )
        await memory.add_episode(episode)

        entry = await memory.get(episode.id)
        assert entry is not None
        assert entry.content == "Test task"
        assert entry.type == MemoryType.EPISODIC

    @pytest.mark.asyncio
    async def test_get_full_episode(self, memory: EpisodicMemory) -> None:
        episode = EpisodicMemoryEntry(
            task_description="Full episode",
            agents_involved=["researcher", "coder"],
            tools_used=["nmap"],
            result_summary="Found issues",
            errors=["timeout"],
            evaluation_score=0.6,
        )
        await memory.add_episode(episode)

        retrieved = await memory.get_episode(episode.id)
        assert retrieved is not None
        assert retrieved.task_description == "Full episode"
        assert len(retrieved.agents_involved) == 2
        assert retrieved.tools_used == ["nmap"]
        assert len(retrieved.errors) == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, memory: EpisodicMemory) -> None:
        result = await memory.get(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_episode_nonexistent(self, memory: EpisodicMemory) -> None:
        result = await memory.get_episode(uuid4())
        assert result is None


class TestEpisodicMemorySearch:
    @pytest.mark.asyncio
    async def test_search_by_task_description(self, memory: EpisodicMemory) -> None:
        await memory.add_episode(
            EpisodicMemoryEntry(task_description="Security analysis of web app")
        )
        await memory.add_episode(
            EpisodicMemoryEntry(task_description="Code review for API")
        )

        query = MemorySearchQuery(query="Security analysis")
        results = await memory.search(query)
        assert len(results) == 1
        assert "Security" in results[0].memory.content

    @pytest.mark.asyncio
    async def test_search_by_result_summary(self, memory: EpisodicMemory) -> None:
        await memory.add_episode(
            EpisodicMemoryEntry(
                task_description="Task",
                result_summary="Found critical vulnerability",
            )
        )

        query = MemorySearchQuery(query="critical vulnerability")
        results = await memory.search(query)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_by_errors(self, memory: EpisodicMemory) -> None:
        await memory.add_episode(
            EpisodicMemoryEntry(
                task_description="Task",
                errors=["Connection timeout on port 443"],
            )
        )

        query = MemorySearchQuery(query="Connection timeout")
        results = await memory.search(query)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_with_agent_filter(self, memory: EpisodicMemory) -> None:
        agent_id = uuid4()
        await memory.add_episode(
            EpisodicMemoryEntry(
                task_description="Task A",
                agent_id=agent_id,
            )
        )
        await memory.add_episode(
            EpisodicMemoryEntry(task_description="Task B")
        )

        query = MemorySearchQuery(query="Task", agent_id=agent_id)
        results = await memory.search(query)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_score_includes_eval(self, memory: EpisodicMemory) -> None:
        await memory.add_episode(
            EpisodicMemoryEntry(
                task_description="Good task",
                evaluation_score=1.0,
            )
        )

        query = MemorySearchQuery(query="Good task", min_score=0.0)
        results = await memory.search(query)
        assert len(results) == 1
        assert results[0].score > 0.5

    @pytest.mark.asyncio
    async def test_search_no_match(self, memory: EpisodicMemory) -> None:
        await memory.add_episode(
            EpisodicMemoryEntry(task_description="Something unrelated")
        )
        query = MemorySearchQuery(query="xyz non existent")
        results = await memory.search(query)
        assert len(results) == 0


class TestEpisodicMemoryUpdate:
    @pytest.mark.asyncio
    async def test_update_generic_entry(self, memory: EpisodicMemory) -> None:
        episode = EpisodicMemoryEntry(task_description="original")
        await memory.add_episode(episode)

        entry = MemoryEntry(
            id=episode.id,
            type=MemoryType.EPISODIC,
            content="updated",
        )
        success = await memory.update(entry)
        assert success is True

        retrieved = await memory.get_episode(episode.id)
        assert retrieved is not None
        assert retrieved.task_description == "updated"

    @pytest.mark.asyncio
    async def test_update_full_episode(self, memory: EpisodicMemory) -> None:
        episode = EpisodicMemoryEntry(task_description="v1")
        await memory.add_episode(episode)

        episode.task_description = "v2"
        episode.evaluation_score = 0.95
        success = await memory.update_episode(episode)
        assert success is True

        retrieved = await memory.get_episode(episode.id)
        assert retrieved.task_description == "v2"
        assert retrieved.evaluation_score == 0.95

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, memory: EpisodicMemory) -> None:
        entry = MemoryEntry(
            id=uuid4(), type=MemoryType.EPISODIC, content="ghost"
        )
        success = await memory.update(entry)
        assert success is False


class TestEpisodicMemoryDelete:
    @pytest.mark.asyncio
    async def test_delete_existing(self, memory: EpisodicMemory) -> None:
        episode = EpisodicMemoryEntry(task_description="delete me")
        await memory.add_episode(episode)
        success = await memory.delete(episode.id)
        assert success is True
        assert await memory.get_episode(episode.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory: EpisodicMemory) -> None:
        success = await memory.delete(uuid4())
        assert success is False


class TestEpisodicMemoryClear:
    @pytest.mark.asyncio
    async def test_clear_all(self, memory: EpisodicMemory) -> None:
        for i in range(3):
            await memory.add_episode(
                EpisodicMemoryEntry(task_description=f"Task {i}")
            )
        count = await memory.clear()
        assert count == 3
        assert len(memory) == 0

    @pytest.mark.asyncio
    async def test_clear_by_agent(self, memory: EpisodicMemory) -> None:
        agent_id = uuid4()
        await memory.add_episode(
            EpisodicMemoryEntry(task_description="A", agent_id=agent_id)
        )
        await memory.add_episode(
            EpisodicMemoryEntry(task_description="B")
        )
        count = await memory.clear(agent_id=agent_id)
        assert count == 1
        assert len(memory) == 1


class TestEpisodicMemoryRecent:
    @pytest.mark.asyncio
    async def test_get_recent_episodes(self, memory: EpisodicMemory) -> None:
        for i in range(5):
            await memory.add_episode(
                EpisodicMemoryEntry(task_description=f"Task {i}")
            )
        recent = await memory.get_recent_episodes(limit=3)
        assert len(recent) == 3

    @pytest.mark.asyncio
    async def test_get_recent_with_agent_filter(self, memory: EpisodicMemory) -> None:
        agent_id = uuid4()
        await memory.add_episode(
            EpisodicMemoryEntry(task_description="A", agent_id=agent_id)
        )
        await memory.add_episode(EpisodicMemoryEntry(task_description="B"))

        recent = await memory.get_recent_episodes(limit=10, agent_id=agent_id)
        assert len(recent) == 1


class TestEpisodicMemorySuccessful:
    @pytest.mark.asyncio
    async def test_get_successful_episodes(self, memory: EpisodicMemory) -> None:
        await memory.add_episode(
            EpisodicMemoryEntry(task_description="Good", evaluation_score=0.9)
        )
        await memory.add_episode(
            EpisodicMemoryEntry(task_description="Bad", evaluation_score=0.3)
        )
        await memory.add_episode(
            EpisodicMemoryEntry(task_description="OK", evaluation_score=0.75)
        )

        successful = await memory.get_successful_episodes(min_score=0.7)
        assert len(successful) == 2
        scores = [e.evaluation_score for e in successful]
        assert all(s >= 0.7 for s in scores)

    @pytest.mark.asyncio
    async def test_get_successful_sorted_by_score(self, memory: EpisodicMemory) -> None:
        await memory.add_episode(
            EpisodicMemoryEntry(task_description="A", evaluation_score=0.8)
        )
        await memory.add_episode(
            EpisodicMemoryEntry(task_description="B", evaluation_score=1.0)
        )

        successful = await memory.get_successful_episodes(min_score=0.0)
        assert successful[0].evaluation_score >= successful[1].evaluation_score
