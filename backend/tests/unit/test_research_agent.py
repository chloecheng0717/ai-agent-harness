"""
Unit Tests for ResearchAgent

Tests cover:
- Agent creation and configuration
- Process with valid task
- Process with missing task
- Process with context and sources
- Reflect method
- Memory updates
- BaseAgent interface compliance
"""

import pytest

from app.agents.base import AgentConfig, BaseAgent
from app.agents.specialized.research import ResearchAgent
from app.models.agent import AgentStatus, AgentType


# ── Agent Init Tests ──────────────────────────────────────────


class TestResearchAgentInit:
    def test_creates_with_defaults(self) -> None:
        agent = ResearchAgent()
        assert agent.config.name == "ResearchAgent"
        assert agent.config.type == AgentType.RESEARCHER
        assert agent.config.description != ""

    def test_creates_with_custom_config(self) -> None:
        config = AgentConfig(
            name="CustomResearcher",
            type=AgentType.RESEARCHER,
            description="Custom research agent",
        )
        agent = ResearchAgent(config=config)
        assert agent.config.name == "CustomResearcher"

    def test_is_base_agent(self) -> None:
        agent = ResearchAgent()
        assert isinstance(agent, BaseAgent)

    def test_initial_status_idle(self) -> None:
        agent = ResearchAgent()
        assert agent.status == AgentStatus.IDLE


# ── Process Tests ─────────────────────────────────────────────


class TestResearchAgentProcess:
    @pytest.mark.asyncio
    async def test_no_task_returns_error(self) -> None:
        agent = ResearchAgent()
        result = await agent.process({})
        assert result["success"] is False
        assert "No task provided" in result["errors"][0]
        assert result["findings"] == []
        assert result["summary"] == ""

    @pytest.mark.asyncio
    async def test_basic_task(self) -> None:
        agent = ResearchAgent()
        result = await agent.process({"task": "analyse security posture"})
        assert result["success"] is True
        assert len(result["findings"]) >= 1
        assert result["summary"] != ""
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_task_with_context(self) -> None:
        agent = ResearchAgent()
        result = await agent.process({
            "task": "research topic",
            "context": "additional info about the topic",
        })
        assert result["success"] is True
        assert len(result["findings"]) >= 2

    @pytest.mark.asyncio
    async def test_task_with_sources(self) -> None:
        agent = ResearchAgent()
        result = await agent.process({
            "task": "gather intelligence",
            "sources": ["database_a", "api_b"],
        })
        assert result["success"] is True
        assert len(result["findings"]) >= 3

    @pytest.mark.asyncio
    async def test_returns_to_idle_after_process(self) -> None:
        agent = ResearchAgent()
        await agent.process({"task": "test"})
        assert agent.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_stores_memory(self) -> None:
        agent = ResearchAgent()
        await agent.process({"task": "memorise this"})
        assert agent.get_memory("last_task") == "memorise this"
        assert agent.get_memory("last_findings") is not None
        assert agent.get_memory("last_summary") is not None

    @pytest.mark.asyncio
    async def test_summary_content(self) -> None:
        agent = ResearchAgent()
        result = await agent.process({"task": "analyse system"})
        assert "Research Summary for:" in result["summary"]
        assert "analyse system" in result["summary"]


# ── Reflect Tests ─────────────────────────────────────────────


class TestResearchAgentReflect:
    @pytest.mark.asyncio
    async def test_reflect_on_success(self) -> None:
        agent = ResearchAgent()
        result = {
            "success": True,
            "findings": [{"category": "task_analysis"}, {"category": "context"}],
            "summary": "test",
            "errors": [],
        }
        insights = await agent.reflect(result)
        assert insights["total_findings"] == 2
        assert insights["had_errors"] is False
        assert insights["overall_success"] is True

    @pytest.mark.asyncio
    async def test_reflect_on_failure(self) -> None:
        agent = ResearchAgent()
        result = {
            "success": False,
            "findings": [],
            "summary": "",
            "errors": ["something broke"],
        }
        insights = await agent.reflect(result)
        assert insights["total_findings"] == 0
        assert insights["had_errors"] is True

    @pytest.mark.asyncio
    async def test_reflect_stores_memory(self) -> None:
        agent = ResearchAgent()
        result = {"success": True, "findings": [], "summary": "", "errors": []}
        await agent.reflect(result)
        assert agent.get_memory("last_reflection") is not None
