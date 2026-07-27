"""
Unit Tests for CodingAgent

Tests cover:
- Agent creation and configuration
- Process for write, debug, and explain task types
- Process with missing task
- Reflect method
- Task classification
- Memory updates
"""

import pytest

from app.agents.base import AgentConfig, BaseAgent
from app.agents.specialized.coding import CodingAgent
from app.models.agent import AgentStatus, AgentType


# ── Agent Init Tests ──────────────────────────────────────────


class TestCodingAgentInit:
    def test_creates_with_defaults(self) -> None:
        agent = CodingAgent()
        assert agent.config.name == "CodingAgent"
        assert agent.config.type == AgentType.CODER
        assert agent.config.description != ""

    def test_creates_with_custom_config(self) -> None:
        config = AgentConfig(
            name="CustomCoder",
            type=AgentType.CODER,
        )
        agent = CodingAgent(config=config)
        assert agent.config.name == "CustomCoder"

    def test_is_base_agent(self) -> None:
        agent = CodingAgent()
        assert isinstance(agent, BaseAgent)


# ── Task Classification Tests ──────────────────────────────────


class TestTaskClassification:
    def test_debug_classification(self) -> None:
        assert CodingAgent._classify_task("debug this code", "") == "debug"
        assert CodingAgent._classify_task("fix the bug", "") == "debug"
        assert CodingAgent._classify_task("there is an error", "") == "debug"

    def test_explain_classification(self) -> None:
        assert CodingAgent._classify_task("explain this function", "") == "explain"
        assert CodingAgent._classify_task("what does this code do", "") == "explain"

    def test_write_classification(self) -> None:
        assert CodingAgent._classify_task("write a function", "") == "write"
        assert CodingAgent._classify_task("create a class", "") == "write"
        assert CodingAgent._classify_task("implement sorting", "") == "write"

    def test_defaults_to_debug_with_code(self) -> None:
        assert CodingAgent._classify_task("review this", "def foo(): pass") == "debug"

    def test_defaults_to_write(self) -> None:
        assert CodingAgent._classify_task("make something", "") == "write"


# ── Process Tests ─────────────────────────────────────────────


class TestCodingAgentProcess:
    @pytest.mark.asyncio
    async def test_no_task_returns_error(self) -> None:
        agent = CodingAgent()
        result = await agent.process({})
        assert result["success"] is False
        assert "No task provided" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_write_task(self) -> None:
        agent = CodingAgent()
        result = await agent.process({
            "task": "write a sorting algorithm",
            "language": "python",
        })
        assert result["success"] is True
        assert result["task_type"] == "write"
        assert "python" in result["output"].lower()

    @pytest.mark.asyncio
    async def test_debug_task(self) -> None:
        agent = CodingAgent()
        result = await agent.process({
            "task": "debug this function",
            "code": "def add(a, b): return a - b",
        })
        assert result["success"] is True
        assert result["task_type"] == "debug"

    @pytest.mark.asyncio
    async def test_explain_task(self) -> None:
        agent = CodingAgent()
        result = await agent.process({
            "task": "explain this code",
            "code": "x = [i**2 for i in range(10)]",
            "language": "python",
        })
        assert result["success"] is True
        assert result["task_type"] == "explain"

    @pytest.mark.asyncio
    async def test_returns_to_idle(self) -> None:
        agent = CodingAgent()
        await agent.process({"task": "write code"})
        assert agent.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_stores_memory(self) -> None:
        agent = CodingAgent()
        await agent.process({"task": "write code"})
        assert agent.get_memory("last_task") == "write code"
        assert agent.get_memory("last_task_type") is not None


# ── Reflect Tests ─────────────────────────────────────────────


class TestCodingAgentReflect:
    @pytest.mark.asyncio
    async def test_reflect_on_success(self) -> None:
        agent = CodingAgent()
        result = {
            "success": True,
            "output": "some code",
            "task_type": "write",
            "errors": [],
        }
        insights = await agent.reflect(result)
        assert insights["task_type"] == "write"
        assert insights["overall_success"] is True
        assert insights["output_length"] == 9

    @pytest.mark.asyncio
    async def test_reflect_on_failure(self) -> None:
        agent = CodingAgent()
        result = {
            "success": False,
            "output": "",
            "task_type": "unknown",
            "errors": ["failed"],
        }
        insights = await agent.reflect(result)
        assert insights["had_errors"] is True
