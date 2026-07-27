"""
Unit Tests for QAAgent

Tests cover:
- Agent creation and configuration
- Process with valid task
- Process with missing task
- Test result generation
- Verdict determination
- Reflect method
"""

import pytest

from app.agents.base import AgentConfig, BaseAgent
from app.agents.specialized.qa import QAAgent
from app.models.agent import AgentStatus, AgentType


# ── Agent Init Tests ──────────────────────────────────────────


class TestQAAgentInit:
    def test_creates_with_defaults(self) -> None:
        agent = QAAgent()
        assert agent.config.name == "QAAgent"
        assert agent.config.type == AgentType.QA
        assert agent.config.description != ""

    def test_creates_with_custom_config(self) -> None:
        config = AgentConfig(name="CustomQA", type=AgentType.QA)
        agent = QAAgent(config=config)
        assert agent.config.name == "CustomQA"

    def test_is_base_agent(self) -> None:
        agent = QAAgent()
        assert isinstance(agent, BaseAgent)


# ── Verdict Tests ─────────────────────────────────────────────


class TestVerdictDetermination:
    def test_pass(self) -> None:
        assert QAAgent._determine_verdict(3, 0) == "pass"

    def test_partial(self) -> None:
        assert QAAgent._determine_verdict(2, 1) == "partial"

    def test_fail(self) -> None:
        assert QAAgent._determine_verdict(0, 3) == "fail"

    def test_no_tests(self) -> None:
        assert QAAgent._determine_verdict(0, 0) == "no_tests"


# ── Process Tests ─────────────────────────────────────────────


class TestQAAgentProcess:
    @pytest.mark.asyncio
    async def test_no_task_returns_error(self) -> None:
        agent = QAAgent()
        result = await agent.process({})
        assert result["success"] is False
        assert "No task provided" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_basic_task(self) -> None:
        agent = QAAgent()
        result = await agent.process({"task": "verify login functionality"})
        assert result["success"] is True
        assert len(result["test_results"]) >= 1
        assert result["verdict"] in ("pass", "partial", "fail", "no_tests", "error")

    @pytest.mark.asyncio
    async def test_with_expected_actual_output(self) -> None:
        agent = QAAgent()
        result = await agent.process({
            "task": "verify output",
            "expected_output": "hello world",
            "actual_output": "hello world",
        })
        assert result["success"] is True
        assert result["passed"] >= 1

    @pytest.mark.asyncio
    async def test_mismatched_output(self) -> None:
        agent = QAAgent()
        result = await agent.process({
            "task": "verify output",
            "expected_output": "expected",
            "actual_output": "actual",
        })
        assert result["success"] is True
        # Should have at least one failure from the comparison
        assert result["failed"] >= 1

    @pytest.mark.asyncio
    async def test_with_criteria(self) -> None:
        agent = QAAgent()
        result = await agent.process({
            "task": "acceptance test",
            "criteria": ["must be fast", "must be secure"],
        })
        assert result["success"] is True
        assert len(result["test_results"]) >= 3  # base + 2 criteria

    @pytest.mark.asyncio
    async def test_returns_to_idle(self) -> None:
        agent = QAAgent()
        await agent.process({"task": "run tests"})
        assert agent.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_stores_memory(self) -> None:
        agent = QAAgent()
        await agent.process({"task": "qa check"})
        assert agent.get_memory("last_task") == "qa check"
        assert agent.get_memory("last_verdict") is not None


# ── Reflect Tests ─────────────────────────────────────────────


class TestQAAgentReflect:
    @pytest.mark.asyncio
    async def test_reflect_on_success(self) -> None:
        agent = QAAgent()
        result = {
            "success": True,
            "test_results": [
                {"category": "completeness"},
                {"category": "correctness"},
            ],
            "passed": 2,
            "failed": 0,
            "verdict": "pass",
            "errors": [],
        }
        insights = await agent.reflect(result)
        assert insights["total_tests"] == 2
        assert insights["passed"] == 2
        assert insights["verdict"] == "pass"

    @pytest.mark.asyncio
    async def test_reflect_on_failure(self) -> None:
        agent = QAAgent()
        result = {
            "success": False,
            "test_results": [],
            "passed": 0,
            "failed": 1,
            "verdict": "fail",
            "errors": ["test failed"],
        }
        insights = await agent.reflect(result)
        assert insights["had_errors"] is True
        assert insights["overall_success"] is False
