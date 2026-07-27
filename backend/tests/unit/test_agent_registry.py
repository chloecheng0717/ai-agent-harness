"""
Unit Tests for AgentRegistry

Tests cover:
- Agent registration
- Agent retrieval
- Agent listing
- Agent validation
- Duplicate prevention
- Unregistration
- Clear
- Error cases
"""

import pytest
from typing import Any, Dict

from app.agents.base import AgentConfig, BaseAgent
from app.agents.specialized.registry import (
    AgentNotFoundError,
    AgentRegistry,
    DuplicateAgentError,
)
from app.agents.specialized.research import ResearchAgent
from app.agents.specialized.coding import CodingAgent
from app.agents.specialized.security import SecurityAgent
from app.agents.specialized.qa import QAAgent
from app.models.agent import AgentType


# ── Helpers ───────────────────────────────────────────────────


class NotAnAgent:
    """Not a BaseAgent subclass — used to test type checking."""
    pass


class MockAgent(BaseAgent):
    """Minimal agent for testing."""
    def __init__(self, config=None):
        if config is None:
            config = AgentConfig(name="MockAgent", type=AgentType.RESEARCHER)
        super().__init__(config)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True}

    async def reflect(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {}


# ── Registration Tests ────────────────────────────────────────


class TestAgentRegistration:
    def test_register_agent(self) -> None:
        registry = AgentRegistry()
        registry.register("researcher", ResearchAgent)
        assert registry.exists("researcher")

    def test_register_multiple(self) -> None:
        registry = AgentRegistry()
        registry.register("researcher", ResearchAgent)
        registry.register("coder", CodingAgent)
        registry.register("security", SecurityAgent)
        registry.register("qa", QAAgent)
        assert registry.count() == 4

    def test_register_duplicate_raises(self) -> None:
        registry = AgentRegistry()
        registry.register("researcher", ResearchAgent)
        with pytest.raises(DuplicateAgentError, match="already registered"):
            registry.register("researcher", ResearchAgent)

    def test_register_overwrite(self) -> None:
        registry = AgentRegistry()
        registry.register("researcher", ResearchAgent)
        registry.register("researcher", CodingAgent, overwrite=True)
        assert registry.get("researcher") is CodingAgent

    def test_register_non_agent_raises(self) -> None:
        registry = AgentRegistry()
        with pytest.raises(TypeError, match="BaseAgent subclass"):
            registry.register("bad", NotAnAgent)


# ── Retrieval Tests ───────────────────────────────────────────


class TestAgentRetrieval:
    def test_get_existing(self) -> None:
        registry = AgentRegistry()
        registry.register("researcher", ResearchAgent)
        assert registry.get("researcher") is ResearchAgent

    def test_get_nonexistent_returns_none(self) -> None:
        registry = AgentRegistry()
        assert registry.get("nonexistent") is None

    def test_get_or_raise_existing(self) -> None:
        registry = AgentRegistry()
        registry.register("researcher", ResearchAgent)
        assert registry.get_or_raise("researcher") is ResearchAgent

    def test_get_or_raise_nonexistent_raises(self) -> None:
        registry = AgentRegistry()
        with pytest.raises(AgentNotFoundError, match="not registered"):
            registry.get_or_raise("nonexistent")


# ── Listing Tests ─────────────────────────────────────────────


class TestAgentListing:
    def test_list_agents_empty(self) -> None:
        registry = AgentRegistry()
        assert registry.list_agents() == []

    def test_list_agents(self) -> None:
        registry = AgentRegistry()
        registry.register("researcher", ResearchAgent)
        registry.register("coder", CodingAgent)
        agents = registry.list_agents()
        assert len(agents) == 2
        keys = [a["key"] for a in agents]
        assert "researcher" in keys
        assert "coder" in keys

    def test_list_keys(self) -> None:
        registry = AgentRegistry()
        registry.register("qa", QAAgent)
        registry.register("coder", CodingAgent)
        keys = registry.list_keys()
        assert keys == ["coder", "qa"]

    def test_count(self) -> None:
        registry = AgentRegistry()
        assert registry.count() == 0
        registry.register("r", ResearchAgent)
        assert registry.count() == 1


# ── Validation Tests ──────────────────────────────────────────


class TestAgentValidation:
    def test_validate_existing(self) -> None:
        registry = AgentRegistry()
        registry.register("researcher", ResearchAgent)
        assert registry.validate_agent("researcher") is True

    def test_validate_nonexistent(self) -> None:
        registry = AgentRegistry()
        assert registry.validate_agent("missing") is False


# ── Unregistration Tests ─────────────────────────────────────


class TestAgentUnregistration:
    def test_unregister_existing(self) -> None:
        registry = AgentRegistry()
        registry.register("researcher", ResearchAgent)
        assert registry.unregister("researcher") is True
        assert registry.exists("researcher") is False

    def test_unregister_nonexistent(self) -> None:
        registry = AgentRegistry()
        assert registry.unregister("missing") is False


# ── Clear Tests ───────────────────────────────────────────────


class TestAgentClear:
    def test_clear(self) -> None:
        registry = AgentRegistry()
        registry.register("r", ResearchAgent)
        registry.register("c", CodingAgent)
        registry.clear()
        assert registry.count() == 0
        assert registry.list_agents() == []
