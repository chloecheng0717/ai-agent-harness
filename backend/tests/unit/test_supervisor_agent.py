"""
Unit Tests for SupervisorAgent

Tests cover:
- Agent creation and configuration
- Process with valid task
- Process with missing task
- Agent selection (keyword-based and explicit)
- Message creation and dispatch
- Result synthesis
- Reflect method
- Workflow integration (Supervisor -> Research -> QA)
"""

import pytest
from typing import Any, Dict

from app.agents.base import AgentConfig, BaseAgent
from app.agents.specialized.supervisor import SupervisorAgent
from app.agents.specialized.registry import AgentRegistry
from app.models.agent import AgentStatus, AgentType


# ── Agent Init Tests ──────────────────────────────────────────


class TestSupervisorAgentInit:
    def test_creates_with_defaults(self) -> None:
        agent = SupervisorAgent()
        assert agent.config.name == "SupervisorAgent"
        assert agent.config.type == AgentType.SUPERVISOR
        assert agent.registry.count() == 4

    def test_creates_with_custom_config(self) -> None:
        config = AgentConfig(name="CustomSupervisor", type=AgentType.SUPERVISOR)
        agent = SupervisorAgent(config=config)
        assert agent.config.name == "CustomSupervisor"

    def test_creates_with_custom_registry(self) -> None:
        registry = AgentRegistry()
        registry.register("researcher", __import__(
            "app.agents.specialized.research", fromlist=["ResearchAgent"]
        ).ResearchAgent)
        agent = SupervisorAgent(registry=registry)
        assert agent.registry.count() == 1

    def test_is_base_agent(self) -> None:
        agent = SupervisorAgent()
        assert isinstance(agent, BaseAgent)

    def test_default_agents_registered(self) -> None:
        agent = SupervisorAgent()
        assert agent.registry.exists("researcher")
        assert agent.registry.exists("coder")
        assert agent.registry.exists("security")
        assert agent.registry.exists("qa")


# ── Agent Selection Tests ─────────────────────────────────────


class TestAgentSelection:
    def test_select_security_agent(self) -> None:
        agent = SupervisorAgent()
        selected = agent._select_agents("scan for vulnerabilities")
        assert "security" in selected

    def test_select_coder_agent(self) -> None:
        agent = SupervisorAgent()
        selected = agent._select_agents("write a sorting algorithm")
        assert "coder" in selected

    def test_select_researcher_agent(self) -> None:
        agent = SupervisorAgent()
        selected = agent._select_agents("research machine learning")
        assert "researcher" in selected

    def test_select_qa_agent(self) -> None:
        agent = SupervisorAgent()
        selected = agent._select_agents("test the login page")
        assert "qa" in selected

    def test_explicit_agents_override(self) -> None:
        agent = SupervisorAgent()
        selected = agent._select_agents(
            "scan for vulnerabilities",
            explicit_agents=["coder"],
        )
        assert selected == ["coder"]

    def test_explicit_agents_filtered_by_registry(self) -> None:
        agent = SupervisorAgent()
        selected = agent._select_agents(
            "task",
            explicit_agents=["nonexistent", "researcher"],
        )
        assert selected == ["researcher"]

    def test_fallback_to_researcher(self) -> None:
        agent = SupervisorAgent()
        selected = agent._select_agents("do something vague")
        assert "researcher" in selected

    def test_multi_agent_selection(self) -> None:
        agent = SupervisorAgent()
        selected = agent._select_agents(
            "test the security of this code"
        )
        assert "security" in selected
        assert "qa" in selected


# ── Process Tests ─────────────────────────────────────────────


class TestSupervisorAgentProcess:
    @pytest.mark.asyncio
    async def test_no_task_returns_error(self) -> None:
        agent = SupervisorAgent()
        result = await agent.process({})
        assert result["success"] is False
        assert "No task provided" in result["errors"][0]
        assert result["agent_results"] == {}

    @pytest.mark.asyncio
    async def test_basic_task(self) -> None:
        agent = SupervisorAgent()
        result = await agent.process({"task": "research AI trends"})
        assert result["success"] is True
        assert result["final_result"] != ""
        assert len(result["agent_results"]) >= 1
        assert len(result["messages"]) >= 1

    @pytest.mark.asyncio
    async def test_explicit_agent_dispatch(self) -> None:
        agent = SupervisorAgent()
        result = await agent.process({
            "task": "write code",
            "agents": ["coder"],
        })
        assert result["success"] is True
        assert "coder" in result["agent_results"]

    @pytest.mark.asyncio
    async def test_returns_to_idle(self) -> None:
        agent = SupervisorAgent()
        await agent.process({"task": "do something"})
        assert agent.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_message_log_populated(self) -> None:
        agent = SupervisorAgent()
        await agent.process({"task": "security scan", "agents": ["security"]})
        log = agent.get_message_log()
        assert len(log) >= 1
        assert log[0]["sender"] == "SupervisorAgent"
        assert log[0]["receiver"] == "security"

    @pytest.mark.asyncio
    async def test_stores_memory(self) -> None:
        agent = SupervisorAgent()
        await agent.process({"task": "research topic"})
        assert agent.get_memory("last_task") == "research topic"
        assert agent.get_memory("last_final_result") is not None

    @pytest.mark.asyncio
    async def test_final_result_contains_agent_output(self) -> None:
        agent = SupervisorAgent()
        result = await agent.process({
            "task": "write a function",
            "agents": ["coder"],
        })
        final_upper = result["final_result"].upper()
        assert "CODER" in final_upper or "SUPERVISOR" in final_upper


# ── Reflect Tests ─────────────────────────────────────────────


class TestSupervisorAgentReflect:
    @pytest.mark.asyncio
    async def test_reflect_on_success(self) -> None:
        agent = SupervisorAgent()
        result = {
            "success": True,
            "final_result": "completed",
            "agent_results": {
                "researcher": {"success": True},
                "coder": {"success": True},
            },
            "messages": [],
            "errors": [],
        }
        insights = await agent.reflect(result)
        assert insights["agents_dispatched"] == 2
        assert insights["successful_agents"] == 2
        assert insights["failed_agents"] == 0

    @pytest.mark.asyncio
    async def test_reflect_on_partial_failure(self) -> None:
        agent = SupervisorAgent()
        result = {
            "success": False,
            "final_result": "",
            "agent_results": {
                "researcher": {"success": True},
                "security": {"success": False, "errors": ["failed"]},
            },
            "messages": [],
            "errors": ["Agent 'security' failed: failed"],
        }
        insights = await agent.reflect(result)
        assert insights["successful_agents"] == 1
        assert insights["failed_agents"] == 1
        assert insights["had_errors"] is True


# ── Workflow Integration Tests ────────────────────────────────


class TestSupervisorWorkflowIntegration:
    @pytest.mark.asyncio
    async def test_security_analysis_workflow(self) -> None:
        """Simulate the example workflow: 'Analyse my web application security'"""
        agent = SupervisorAgent()
        result = await agent.process({
            "task": "Analyze my web application security",
            "agents": ["researcher", "security", "qa"],
        })
        assert result["success"] is True
        assert "researcher" in result["agent_results"]
        assert "security" in result["agent_results"]
        assert "qa" in result["agent_results"]
        assert result["final_result"] != ""
        # Should have 3 messages
        assert len(result["messages"]) == 3

    @pytest.mark.asyncio
    async def test_coding_with_qa_workflow(self) -> None:
        """Code something and verify with QA."""
        agent = SupervisorAgent()
        result = await agent.process({
            "task": "write and test a sorting function",
            "agents": ["coder", "qa"],
        })
        assert result["success"] is True
        assert "coder" in result["agent_results"]
        assert "qa" in result["agent_results"]
