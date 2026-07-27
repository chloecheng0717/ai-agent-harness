"""
Unit Tests for Memory Models

Tests cover:
- MemoryType enum
- MemoryEntry creation and defaults
- MemorySearchResult
- MemorySearchQuery
- EpisodicMemoryEntry
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from app.models.memory import (
    EpisodicMemoryEntry,
    MemoryEntry,
    MemorySearchQuery,
    MemorySearchResult,
    MemoryType,
)


class TestMemoryType:
    def test_enum_values(self) -> None:
        assert MemoryType.SHORT_TERM == "short_term"
        assert MemoryType.LONG_TERM == "long_term"
        assert MemoryType.EPISODIC == "episodic"

    def test_enum_members(self) -> None:
        assert len(MemoryType) == 3


class TestMemoryEntry:
    def test_default_creation(self) -> None:
        entry = MemoryEntry(
            type=MemoryType.SHORT_TERM,
            content="Hello world",
        )
        assert entry.id is not None
        assert entry.type == MemoryType.SHORT_TERM
        assert entry.content == "Hello world"
        assert entry.metadata == {}
        assert entry.embedding is None
        assert entry.agent_id is None
        assert entry.task_id is None
        assert entry.created_at is not None
        assert entry.expires_at is None

    def test_full_creation(self) -> None:
        agent_id = uuid4()
        task_id = uuid4()
        entry = MemoryEntry(
            type=MemoryType.LONG_TERM,
            content="Knowledge entry",
            metadata={"source": "test"},
            embedding=[0.1, 0.2, 0.3],
            agent_id=agent_id,
            task_id=task_id,
        )
        assert entry.agent_id == agent_id
        assert entry.task_id == task_id
        assert entry.embedding == [0.1, 0.2, 0.3]
        assert entry.metadata == {"source": "test"}

    def test_serialization(self) -> None:
        entry = MemoryEntry(
            type=MemoryType.EPISODIC,
            content="Task completed",
        )
        data = entry.model_dump()
        assert data["type"] == "episodic"
        assert data["content"] == "Task completed"
        assert "id" in data
        assert "created_at" in data


class TestMemorySearchResult:
    def test_creation(self) -> None:
        entry = MemoryEntry(type=MemoryType.SHORT_TERM, content="test")
        result = MemorySearchResult(memory=entry, score=0.85)
        assert result.memory == entry
        assert result.score == 0.85
        assert result.distance is None

    def test_with_distance(self) -> None:
        entry = MemoryEntry(type=MemoryType.LONG_TERM, content="test")
        result = MemorySearchResult(memory=entry, score=0.9, distance=0.1)
        assert result.distance == 0.1

    def test_score_bounds(self) -> None:
        entry = MemoryEntry(type=MemoryType.SHORT_TERM, content="test")
        with pytest.raises(Exception):
            MemorySearchResult(memory=entry, score=1.5)
        with pytest.raises(Exception):
            MemorySearchResult(memory=entry, score=-0.1)


class TestMemorySearchQuery:
    def test_default_query(self) -> None:
        query = MemorySearchQuery(query="find this")
        assert query.query == "find this"
        assert query.type is None
        assert query.agent_id is None
        assert query.task_id is None
        assert query.limit == 10
        assert query.min_score == 0.5

    def test_filtered_query(self) -> None:
        agent_id = uuid4()
        task_id = uuid4()
        query = MemorySearchQuery(
            query="specific search",
            type=MemoryType.EPISODIC,
            agent_id=agent_id,
            task_id=task_id,
            limit=5,
            min_score=0.8,
        )
        assert query.type == MemoryType.EPISODIC
        assert query.agent_id == agent_id
        assert query.task_id == task_id
        assert query.limit == 5
        assert query.min_score == 0.8


class TestEpisodicMemoryEntry:
    def test_default_creation(self) -> None:
        episode = EpisodicMemoryEntry(
            task_description="Build a web app",
        )
        assert episode.id is not None
        assert episode.task_description == "Build a web app"
        assert episode.agents_involved == []
        assert episode.tools_used == []
        assert episode.result_summary == ""
        assert episode.errors == []
        assert episode.evaluation_score == 0.0
        assert episode.metadata == {}
        assert episode.agent_id is None
        assert episode.created_at is not None

    def test_full_creation(self) -> None:
        agent_id = uuid4()
        episode = EpisodicMemoryEntry(
            task_description="Analyze security",
            agents_involved=["researcher", "security"],
            tools_used=["nmap", "nikto"],
            result_summary="Found 3 critical vulnerabilities",
            errors=["Timeout on port scan"],
            evaluation_score=0.75,
            metadata={"severity": "high"},
            agent_id=agent_id,
        )
        assert len(episode.agents_involved) == 2
        assert len(episode.tools_used) == 2
        assert episode.evaluation_score == 0.75
        assert episode.agent_id == agent_id

    def test_evaluation_score_bounds(self) -> None:
        with pytest.raises(Exception):
            EpisodicMemoryEntry(
                task_description="test",
                evaluation_score=1.5,
            )
        with pytest.raises(Exception):
            EpisodicMemoryEntry(
                task_description="test",
                evaluation_score=-0.1,
            )

    def test_serialization(self) -> None:
        episode = EpisodicMemoryEntry(
            task_description="Test task",
            evaluation_score=0.9,
        )
        data = episode.model_dump()
        assert data["task_description"] == "Test task"
        assert data["evaluation_score"] == 0.9
        assert "id" in data
