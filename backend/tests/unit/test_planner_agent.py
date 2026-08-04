"""
Unit Tests for PlannerAgent

Tests cover:
- Agent creation and configuration
- Plan generation from template (no LLM)
- Plan generation from LLM (mocked)
- Plan validation (structure, dependencies, cycles)
- Plan revision with error feedback
- Reflect method
- Edge cases (empty goal, malformed JSON, etc.)
"""

import json
import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.base import AgentConfig
from app.agents.specialized.planner import (
    MAX_PLAN_STEPS,
    MAX_REVISION_ATTEMPTS,
    VALID_PRIORITIES,
    PlannerAgent,
)
from app.models.agent import AgentType
from app.models.llm import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatMessage,
    LLMProviderConfig,
    LLMProviderError,
    TokenUsageInfo,
)
from app.models.providers.openai_provider import OpenAIProvider
from app.models.state import PlanStep


# ── Helpers ──────────────────────────────────────────────────────


def _make_llm_response(content: str, model: str = "test-model") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="cmpl-test",
        model=model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=TokenUsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


def _make_provider() -> OpenAIProvider:
    return OpenAIProvider(
        LLMProviderConfig(
            api_key="sk-test",
            base_url="https://api.example.com/v1",
            model="test-model",
        )
    )


def _sample_plan_json() -> str:
    return json.dumps([
        {
            "index": 0,
            "description": "Scan target ports",
            "tool_name": "nmap",
            "tool_parameters": {"target": "192.168.1.1"},
            "depends_on": [],
            "rationale": "Discover open ports",
            "priority": "high",
            "estimated_tools": ["nmap"],
            "expected_output": "List of open ports",
        },
        {
            "index": 1,
            "description": "Analyse results",
            "tool_name": None,
            "tool_parameters": {},
            "depends_on": [0],
            "rationale": "Summarise findings",
            "priority": "medium",
            "estimated_tools": [],
            "expected_output": "Summary report",
        },
    ])


# ── Agent creation ───────────────────────────────────────────────


class TestPlannerAgentInit:
    def test_creates_with_defaults(self) -> None:
        agent = PlannerAgent()
        assert agent.config.name == "PlannerAgent"
        assert agent.config.type == AgentType.PLANNER
        assert agent.llm_provider is None
        assert agent.max_plan_steps == MAX_PLAN_STEPS

    def test_creates_with_custom_config(self) -> None:
        config = AgentConfig(
            name="CustomPlanner",
            type=AgentType.PLANNER,
            model="gpt-4o",
        )
        provider = _make_provider()
        agent = PlannerAgent(config=config, llm_provider=provider, max_plan_steps=10)
        assert agent.config.name == "CustomPlanner"
        assert agent.llm_provider is provider
        assert agent.max_plan_steps == 10

    def test_is_base_agent(self) -> None:
        from app.agents.base import BaseAgent
        agent = PlannerAgent()
        assert isinstance(agent, BaseAgent)


# ── Template fallback ────────────────────────────────────────────


class TestTemplatePlan:
    def test_returns_single_step(self) -> None:
        steps = PlannerAgent._plan_from_template("scan the network")
        assert len(steps) == 1
        assert steps[0].description == "scan the network"
        assert steps[0].index == 0
        assert steps[0].priority == "medium"
        assert steps[0].rationale != ""
        assert steps[0].expected_output is not None

    def test_no_tool_name(self) -> None:
        steps = PlannerAgent._plan_from_template("do something")
        assert steps[0].tool_name is None

    def test_step_has_all_new_fields(self) -> None:
        steps = PlannerAgent._plan_from_template("test")
        step = steps[0]
        assert hasattr(step, "priority")
        assert hasattr(step, "estimated_tools")
        assert hasattr(step, "expected_output")


# ── process() without LLM ────────────────────────────────────────


class TestProcessWithoutLLM:
    @pytest.mark.asyncio
    async def test_returns_template_plan(self) -> None:
        agent = PlannerAgent()
        result = await agent.process({"goal": "scan network"})
        assert result["success"] is True
        assert len(result["plan"]) == 1
        assert result["plan"][0]["description"] == "scan network"

    @pytest.mark.asyncio
    async def test_empty_goal_fails(self) -> None:
        agent = PlannerAgent()
        result = await agent.process({"goal": ""})
        assert result["success"] is False
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_missing_goal_fails(self) -> None:
        agent = PlannerAgent()
        result = await agent.process({})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_plan_stored_in_memory(self) -> None:
        agent = PlannerAgent()
        await agent.process({"goal": "test"})
        assert agent.get_memory("last_goal") == "test"
        assert agent.get_memory("last_plan") is not None


# ── process() with LLM ───────────────────────────────────────────


class TestProcessWithLLM:
    @pytest.mark.asyncio
    async def test_generates_plan_from_llm(self) -> None:
        provider = _make_provider()
        agent = PlannerAgent(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response(_sample_plan_json())
            result = await agent.process({"goal": "scan the network"})

        assert result["success"] is True
        assert len(result["plan"]) == 2
        assert result["plan"][0]["tool_name"] == "nmap"
        assert result["plan"][0]["priority"] == "high"
        assert result["plan"][0]["expected_output"] == "List of open ports"
        assert result["plan"][1]["depends_on"] == [0]

    @pytest.mark.asyncio
    async def test_passes_tools_available(self) -> None:
        provider = _make_provider()
        agent = PlannerAgent(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response(_sample_plan_json())
            await agent.process({
                "goal": "scan",
                "tools_available": ["nmap", "curl"],
            })
            call_args = mock.call_args
            user_msg = call_args[0][0].messages[-1].content
            assert "nmap" in user_msg
            assert "curl" in user_msg

    @pytest.mark.asyncio
    async def test_passes_tool_descriptions(self) -> None:
        provider = _make_provider()
        agent = PlannerAgent(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response(_sample_plan_json())
            tool_desc = "  - nmap: Network port scanner\n  - curl: HTTP client"
            await agent.process({
                "goal": "scan",
                "tools_available": ["nmap", "curl"],
                "tool_descriptions": tool_desc,
            })
            call_args = mock.call_args
            user_msg = call_args[0][0].messages[-1].content
            assert "Network port scanner" in user_msg
            assert "HTTP client" in user_msg
            assert "Available tools:" in user_msg

    @pytest.mark.asyncio
    async def test_passes_context(self) -> None:
        provider = _make_provider()
        agent = PlannerAgent(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response(_sample_plan_json())
            await agent.process({
                "goal": "scan",
                "context": "Target is a Linux server",
            })
            call_args = mock.call_args
            user_msg = call_args[0][0].messages[-1].content
            assert "Linux server" in user_msg

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_error(self) -> None:
        provider = _make_provider()
        agent = PlannerAgent(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.side_effect = LLMProviderError("server error")
            result = await agent.process({"goal": "test"})

        assert result["success"] is True
        assert len(result["plan"]) == 1
        assert result["plan"][0]["description"] == "test"

    @pytest.mark.asyncio
    async def test_falls_back_on_invalid_json(self) -> None:
        provider = _make_provider()
        agent = PlannerAgent(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response("not valid json at all")
            result = await agent.process({"goal": "test"})

        assert result["success"] is True
        assert len(result["plan"]) == 1

    @pytest.mark.asyncio
    async def test_falls_back_on_non_array_json(self) -> None:
        provider = _make_provider()
        agent = PlannerAgent(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response(json.dumps({"steps": []}))
            result = await agent.process({"goal": "test"})

        assert result["success"] is True
        assert len(result["plan"]) == 1

    @pytest.mark.asyncio
    async def test_strips_code_fences(self) -> None:
        provider = _make_provider()
        agent = PlannerAgent(llm_provider=provider)
        fenced = f"```json\n{_sample_plan_json()}\n```"

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response(fenced)
            result = await agent.process({"goal": "scan"})

        assert result["success"] is True
        assert len(result["plan"]) == 2

    @pytest.mark.asyncio
    async def test_preserves_tool_args_alias_keys(self) -> None:
        """LLM `tool`/`args` keys must be preserved as PlanStep fields.

        Regression test: a model may author plan steps with tool-call
        style keys (`tool`, `args`) instead of the documented
        `tool_name`/`tool_parameters`.  The arguments were previously
        dropped, leaving an empty parameters dict and a python step that
        could not run.
        """
        provider = _make_provider()
        agent = PlannerAgent(llm_provider=provider)
        args_plan = json.dumps([
            {
                "index": 0,
                "description": "Decode the flag from the message file",
                "tool": "python",
                "args": {
                    "code": "print(open('/workspace/message.txt').read())"
                },
                "depends_on": [],
                "rationale": "The file holds encoded text.",
                "priority": "high",
            }
        ])

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response(args_plan)
            result = await agent.process({
                "goal": "decode the flag",
                "tools_available": ["python"],
            })

        assert result["success"] is True
        assert result["plan"][0]["tool_name"] == "python"
        assert result["plan"][0]["tool_parameters"] == {
            "code": "print(open('/workspace/message.txt').read())"
        }


# ── Plan validation ──────────────────────────────────────────────


class TestValidatePlan:
    def test_empty_plan_fails(self) -> None:
        agent = PlannerAgent()
        errors = agent.validate_plan([])
        assert len(errors) == 1
        assert "empty" in errors[0].lower()

    def test_valid_plan_passes(self) -> None:
        agent = PlannerAgent()
        steps = [
            PlanStep(index=0, description="Step one", priority="high"),
            PlanStep(index=1, description="Step two", depends_on=[0], priority="medium"),
        ]
        errors = agent.validate_plan(steps)
        assert errors == []

    def test_duplicate_index_fails(self) -> None:
        agent = PlannerAgent()
        steps = [
            PlanStep(index=0, description="A"),
            PlanStep(index=0, description="B"),
        ]
        errors = agent.validate_plan(steps)
        assert any("Duplicate" in e for e in errors)

    def test_empty_description_fails(self) -> None:
        agent = PlannerAgent()
        steps = [PlanStep(index=0, description="")]
        errors = agent.validate_plan(steps)
        assert any("empty description" in e.lower() for e in errors)

    def test_invalid_priority_fails(self) -> None:
        agent = PlannerAgent()
        steps = [PlanStep(index=0, description="Step", priority="urgent")]
        errors = agent.validate_plan(steps)
        assert any("invalid priority" in e.lower() for e in errors)

    def test_dependency_on_nonexistent_fails(self) -> None:
        agent = PlannerAgent()
        steps = [
            PlanStep(index=0, description="A"),
            PlanStep(index=1, description="B", depends_on=[5]),
        ]
        errors = agent.validate_plan(steps)
        assert any("non-existent" in e for e in errors)

    def test_dependency_on_later_step_fails(self) -> None:
        agent = PlannerAgent()
        steps = [
            PlanStep(index=0, description="A", depends_on=[1]),
            PlanStep(index=1, description="B"),
        ]
        errors = agent.validate_plan(steps)
        assert any("not earlier" in e for e in errors)

    def test_cycle_detected(self) -> None:
        agent = PlannerAgent()
        steps = [
            PlanStep(index=0, description="A", depends_on=[1]),
            PlanStep(index=1, description="B", depends_on=[0]),
        ]
        errors = agent.validate_plan(steps)
        # The forward-dependency check catches this as "not earlier",
        # which effectively prevents the cycle from executing.
        assert len(errors) > 0
        assert any("step" in e.lower() for e in errors)

    def test_too_many_steps_fails(self) -> None:
        agent = PlannerAgent(max_plan_steps=3)
        steps = [
            PlanStep(index=i, description=f"Step {i}")
            for i in range(5)
        ]
        errors = agent.validate_plan(steps)
        assert any("exceeding limit" in e for e in errors)

    def test_valid_priorities(self) -> None:
        assert VALID_PRIORITIES == {"low", "medium", "high", "critical"}


# ── reflect() ────────────────────────────────────────────────────


class TestReflect:
    @pytest.mark.asyncio
    async def test_reflect_on_success(self) -> None:
        agent = PlannerAgent()
        plan = [
            {"index": 0, "description": "A", "tool_name": "nmap"},
            {"index": 1, "description": "B", "tool_name": None},
        ]
        insights = await agent.reflect({"success": True, "plan": plan, "errors": []})
        assert insights["plan_generated"] is True
        assert insights["step_count"] == 2
        assert insights["tool_steps"] == 1
        assert insights["reasoning_steps"] == 1

    @pytest.mark.asyncio
    async def test_reflect_on_failure(self) -> None:
        agent = PlannerAgent()
        insights = await agent.reflect({
            "success": False,
            "plan": [],
            "errors": ["bad plan"],
        })
        assert insights["plan_generated"] is False
        assert insights["failure_reasons"] == ["bad plan"]

    @pytest.mark.asyncio
    async def test_reflect_stores_memory(self) -> None:
        agent = PlannerAgent()
        await agent.reflect({"success": True, "plan": [], "errors": []})
        assert agent.get_memory("last_reflection") is not None


# ── revise_plan() ────────────────────────────────────────────────


class TestRevisePlan:
    @pytest.mark.asyncio
    async def test_revision_with_llm(self) -> None:
        provider = _make_provider()
        agent = PlannerAgent(llm_provider=provider)
        agent.update_memory("last_goal", "scan network")

        revised_json = json.dumps([
            {
                "index": 0,
                "description": "Revised scan with retry",
                "tool_name": "nmap",
                "depends_on": [],
                "rationale": "Address timeout error",
                "priority": "high",
                "estimated_tools": ["nmap"],
                "expected_output": "Port list",
            }
        ])

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response(revised_json)
            steps = await agent.revise_plan(
                previous_plan=[{"index": 0, "description": "old scan"}],
                errors=["nmap timed out"],
                feedback="Use shorter timeout",
            )

        assert len(steps) == 1
        assert "Revised" in steps[0].description

    @pytest.mark.asyncio
    async def test_revision_fallback_without_llm(self) -> None:
        agent = PlannerAgent()
        agent.update_memory("last_goal", "test goal")
        steps = await agent.revise_plan(
            previous_plan=[{"index": 0, "description": "old"}],
            errors=["failed"],
        )
        assert len(steps) == 1
        assert steps[0].description == "test goal"

    @pytest.mark.asyncio
    async def test_revision_fallback_on_llm_error(self) -> None:
        provider = _make_provider()
        agent = PlannerAgent(llm_provider=provider)
        agent.update_memory("last_goal", "fallback goal")

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.side_effect = LLMProviderError("error")
            steps = await agent.revise_plan(
                previous_plan=[{"index": 0, "description": "old"}],
                errors=["failed"],
            )

        assert len(steps) == 1
        assert steps[0].description == "fallback goal"

    @pytest.mark.asyncio
    async def test_revision_tries_multiple_times(self) -> None:
        provider = _make_provider()
        agent = PlannerAgent(llm_provider=provider)
        agent.update_memory("last_goal", "test")

        call_count = 0

        async def side_effect(request: Any) -> ChatCompletionResponse:
            nonlocal call_count
            call_count += 1
            if call_count < MAX_REVISION_ATTEMPTS:
                return _make_llm_response("not json")
            return _make_llm_response(_sample_plan_json())

        with patch.object(provider, "chat_completion", side_effect=side_effect):
            steps = await agent.revise_plan(
                previous_plan=[{"index": 0, "description": "old"}],
                errors=["failed"],
            )

        assert len(steps) == 2
        assert call_count == MAX_REVISION_ATTEMPTS


# ── JSON parsing ─────────────────────────────────────────────────


class TestParsePlanJson:
    def test_valid_json(self) -> None:
        agent = PlannerAgent()
        steps = agent._parse_plan_json(_sample_plan_json())
        assert len(steps) == 2
        assert steps[0].tool_name == "nmap"
        assert steps[1].depends_on == [0]

    def test_empty_array(self) -> None:
        agent = PlannerAgent()
        steps = agent._parse_plan_json("[]")
        assert steps == []

    def test_invalid_json(self) -> None:
        agent = PlannerAgent()
        steps = agent._parse_plan_json("not json")
        assert steps == []

    def test_non_array_json(self) -> None:
        agent = PlannerAgent()
        steps = agent._parse_plan_json('{"key": "value"}')
        assert steps == []

    def test_malformed_step_skipped(self) -> None:
        agent = PlannerAgent()
        data = json.dumps([
            {"index": 0, "description": "valid step"},
            "not a dict",
            {"index": 1, "description": "also valid"},
        ])
        steps = agent._parse_plan_json(data)
        assert len(steps) == 2

    def test_strips_markdown_fences(self) -> None:
        agent = PlannerAgent()
        fenced = f"```json\n{_sample_plan_json()}\n```"
        steps = agent._parse_plan_json(fenced)
        assert len(steps) == 2

    def test_handles_missing_optional_fields(self) -> None:
        agent = PlannerAgent()
        data = json.dumps([{"index": 0, "description": "minimal step"}])
        steps = agent._parse_plan_json(data)
        assert len(steps) == 1
        assert steps[0].tool_name is None
        assert steps[0].depends_on == []
        assert steps[0].priority == "medium"
        assert steps[0].estimated_tools == []
        assert steps[0].expected_output is None


# ── Cycle detection ──────────────────────────────────────────────


class TestCycleDetection:
    def test_no_cycle(self) -> None:
        steps = [
            PlanStep(index=0, description="A"),
            PlanStep(index=1, description="B", depends_on=[0]),
            PlanStep(index=2, description="C", depends_on=[0, 1]),
        ]
        assert PlannerAgent._detect_cycle(steps) is None

    def test_direct_cycle(self) -> None:
        steps = [
            PlanStep(index=0, description="A", depends_on=[1]),
            PlanStep(index=1, description="B", depends_on=[0]),
        ]
        result = PlannerAgent._detect_cycle(steps)
        assert result is not None
        assert "cycle" in result.lower()

    def test_indirect_cycle(self) -> None:
        steps = [
            PlanStep(index=0, description="A", depends_on=[2]),
            PlanStep(index=1, description="B", depends_on=[0]),
            PlanStep(index=2, description="C", depends_on=[1]),
        ]
        result = PlannerAgent._detect_cycle(steps)
        assert result is not None
