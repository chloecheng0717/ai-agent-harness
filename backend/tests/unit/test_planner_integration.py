"""
Integration Tests for PlannerAgent + Orchestrator

Tests cover:
- Orchestrator with PlannerAgent for create_plan
- Full lifecycle with PlannerAgent
- Replanning on execution errors
- Plan metadata propagation
- Fallback behaviour without planner
- Plan revision flow through orchestrator
"""

import json
import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.agents.core.orchestrator import Orchestrator
from app.agents.specialized.planner import PlannerAgent
from app.agents.base import AgentConfig
from app.models.agent import AgentType
from app.models.llm import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    LLMProviderConfig,
    LLMProviderError,
    TokenUsageInfo,
)
from app.models.providers.openai_provider import OpenAIProvider
from app.models.state import (
    AgentState,
    LifecycleStep,
    PlanStep,
    StepStatus,
)


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


def _sample_plan_json(tools: bool = True) -> str:
    steps = [
        {
            "index": 0,
            "description": "Scan target ports",
            "tool_name": "nmap" if tools else None,
            "tool_parameters": {"target": "192.168.1.1"} if tools else {},
            "depends_on": [],
            "rationale": "Discover open ports",
            "priority": "high",
            "estimated_tools": ["nmap"] if tools else [],
            "expected_output": "List of open ports",
        },
        {
            "index": 1,
            "description": "Analyse scan results",
            "tool_name": None,
            "tool_parameters": {},
            "depends_on": [0],
            "rationale": "Summarise findings",
            "priority": "medium",
            "estimated_tools": [],
            "expected_output": "Summary report",
        },
    ]
    return json.dumps(steps)


# ── Orchestrator create_plan with PlannerAgent ───────────────────


class TestOrchestratorCreatePlanWithPlanner:
    @pytest.mark.asyncio
    async def test_planner_agent_used_for_plan(self) -> None:
        planner = PlannerAgent()
        orch = Orchestrator(planner=planner)

        state = AgentState()
        state.goal = "scan the network"
        state.tools_available = ["nmap"]
        state = await orch.create_plan(state)

        assert len(state.plan) == 1
        assert state.plan[0].description == "scan the network"
        assert state.metadata.get("plan_source") == "planner_agent"

    @pytest.mark.asyncio
    async def test_preserves_existing_plan(self) -> None:
        planner = PlannerAgent()
        orch = Orchestrator(planner=planner)
        state = AgentState()
        existing = [PlanStep(index=0, description="existing")]
        state.plan = existing
        state = await orch.create_plan(state)
        assert state.plan == existing

    @pytest.mark.asyncio
    async def test_falls_back_on_planner_failure(self) -> None:
        planner = PlannerAgent()
        orch = Orchestrator(planner=planner)

        with patch.object(planner, "process", new_callable=AsyncMock) as mock:
            mock.return_value = {"success": False, "plan": [], "errors": ["bad"]}
            state = AgentState()
            state.goal = "test"
            state = await orch.create_plan(state)

        assert len(state.plan) == 1
        assert "fallback" in state.plan[0].rationale.lower()

    @pytest.mark.asyncio
    async def test_falls_back_on_planner_exception(self) -> None:
        planner = PlannerAgent()
        orch = Orchestrator(planner=planner)

        with patch.object(planner, "process", new_callable=AsyncMock) as mock:
            mock.side_effect = RuntimeError("boom")
            state = AgentState()
            state.goal = "test"
            state = await orch.create_plan(state)

        assert len(state.plan) == 1
        assert "fallback" in state.plan[0].rationale.lower()

    @pytest.mark.asyncio
    async def test_plan_step_has_extended_fields(self) -> None:
        planner = PlannerAgent()
        orch = Orchestrator(planner=planner)

        state = AgentState()
        state.goal = "scan network"
        state = await orch.create_plan(state)

        step = state.plan[0]
        assert step.priority == "medium"
        assert step.estimated_tools == []
        assert step.expected_output is not None


# ── Orchestrator with LLM + Planner ──────────────────────────────


class TestOrchestratorWithLLMAndPlanner:
    @pytest.mark.asyncio
    async def test_planner_agent_takes_precedence_over_llm(self) -> None:
        provider = _make_provider()
        planner = PlannerAgent(llm_provider=provider)
        orch = Orchestrator(llm_provider=provider, planner=planner)

        with patch.object(planner, "process", new_callable=AsyncMock) as mock_planner:
            mock_planner.return_value = {
                "success": True,
                "plan": [
                    {
                        "index": 0,
                        "description": "Planner agent step",
                        "tool_name": "nmap",
                        "depends_on": [],
                        "rationale": "from planner",
                        "priority": "high",
                        "estimated_tools": ["nmap"],
                        "expected_output": "ports",
                    }
                ],
                "errors": [],
            }
            state = AgentState()
            state.goal = "scan"
            state = await orch.create_plan(state)

        assert state.plan[0].description == "Planner agent step"
        assert state.plan[0].priority == "high"
        assert state.metadata.get("plan_source") == "planner_agent"

    @pytest.mark.asyncio
    async def test_llm_fallback_when_planner_absent(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response(_sample_plan_json(tools=False))
            state = AgentState()
            state.goal = "scan"
            state = await orch.create_plan(state)

        assert len(state.plan) == 2
        assert state.metadata.get("plan_source") == "orchestrator_llm"


# ── Full lifecycle with PlannerAgent ─────────────────────────────


class TestFullLifecycleWithPlanner:
    @pytest.mark.asyncio
    async def test_full_lifecycle_no_llm(self) -> None:
        planner = PlannerAgent()
        orch = Orchestrator(planner=planner)
        result = await orch.run(
            task_id=uuid4(),
            task_data={"description": "scan the network"},
        )
        assert result["status"] == "completed"
        assert result["final_answer"] is not None

    @pytest.mark.asyncio
    async def test_full_lifecycle_with_llm_and_planner(self) -> None:
        provider = _make_provider()
        planner = PlannerAgent(llm_provider=provider)
        orch = Orchestrator(llm_provider=provider, planner=planner)

        responses = [
            _make_llm_response("Scan the target"),        # understand_task
            _make_llm_response(_sample_plan_json()),      # create_plan (via planner)
            _make_llm_response("Execution successful"),    # evaluate_result
            _make_llm_response("All steps completed"),     # reflect
            _make_llm_response("Scan complete"),           # respond
        ]

        call_idx = 0

        async def mock_chat(request: ChatCompletionRequest) -> ChatCompletionResponse:
            nonlocal call_idx
            resp = responses[call_idx]
            call_idx += 1
            return resp

        with patch.object(provider, "chat_completion", side_effect=mock_chat):
            result = await orch.run(
                task_id=uuid4(),
                task_data={"description": "scan target 192.168.1.1"},
            )

        assert result["status"] == "completed"
        assert result["token_usage"]["total_tokens"] > 0


# ── Replanning integration ───────────────────────────────────────


class TestReplanningIntegration:
    @pytest.mark.asyncio
    async def test_replan_triggered_on_errors(self) -> None:
        planner = PlannerAgent()
        orch = Orchestrator(planner=planner)

        state = AgentState()
        state.goal = "scan network"
        state.plan = [PlanStep(index=0, description="original step")]
        state.results = [{"step_index": 0, "output": None, "success": False}]
        state.add_error("nmap timed out")
        state.current_step = LifecycleStep.EVALUATE_RESULT

        with patch.object(planner, "process", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "success": True,
                "plan": [
                    {
                        "index": 0,
                        "description": "Revised scan with shorter timeout",
                        "tool_name": "nmap",
                        "depends_on": [],
                        "rationale": "Address timeout",
                        "priority": "high",
                        "estimated_tools": ["nmap"],
                        "expected_output": "Port list",
                    }
                ],
                "errors": [],
            }
            state = await orch.evaluate_result(state)

        assert len(state.plan) == 1
        assert "Revised" in state.plan[0].description
        assert state.metadata.get("replan_count") == 1
        assert state.metadata.get("plan_source") == "planner_agent_replan"
        assert len(state.results) == 0
        assert len(state.errors) == 0

    @pytest.mark.asyncio
    async def test_no_replan_without_errors(self) -> None:
        planner = PlannerAgent()
        orch = Orchestrator(planner=planner)

        state = AgentState()
        state.goal = "test"
        state.plan = [PlanStep(index=0, description="step")]
        state.results = [{"step_index": 0, "output": "done", "success": True}]
        state.current_step = LifecycleStep.EVALUATE_RESULT

        with patch.object(planner, "process", new_callable=AsyncMock) as mock:
            state = await orch.evaluate_result(state)
            mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_replan_without_planner(self) -> None:
        orch = Orchestrator()

        state = AgentState()
        state.goal = "test"
        state.add_error("something failed")
        state.current_step = LifecycleStep.EVALUATE_RESULT

        state = await orch.evaluate_result(state)
        assert state.metadata.get("replan_count") is None

    @pytest.mark.asyncio
    async def test_replan_respects_max_attempts(self) -> None:
        planner = PlannerAgent()
        orch = Orchestrator(planner=planner)

        state = AgentState()
        state.goal = "test"
        state.plan = [PlanStep(index=0, description="step")]
        state.results = []
        state.add_error("error 1")
        state.metadata["replan_count"] = 2
        state.current_step = LifecycleStep.EVALUATE_RESULT

        with patch.object(planner, "process", new_callable=AsyncMock) as mock:
            state = await orch.evaluate_result(state)
            mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_replan_clears_old_results_and_errors(self) -> None:
        planner = PlannerAgent()
        orch = Orchestrator(planner=planner)

        state = AgentState()
        state.goal = "test"
        state.plan = [PlanStep(index=0, description="old")]
        state.results = [{"step_index": 0, "output": "failed"}]
        state.add_error("error 1")
        state.add_error("error 2")
        state.current_step = LifecycleStep.EVALUATE_RESULT

        with patch.object(planner, "process", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "success": True,
                "plan": [{"index": 0, "description": "new step", "depends_on": [],
                          "rationale": "retry", "priority": "medium",
                          "estimated_tools": [], "expected_output": None}],
                "errors": [],
            }
            state = await orch.evaluate_result(state)

        assert len(state.results) == 0
        assert len(state.errors) == 0
        assert any("revised" in m.content.lower() for m in state.messages)

    @pytest.mark.asyncio
    async def test_replan_failure_does_not_crash(self) -> None:
        planner = PlannerAgent()
        orch = Orchestrator(planner=planner)

        state = AgentState()
        state.goal = "test"
        state.plan = [PlanStep(index=0, description="step")]
        state.results = []
        state.add_error("error")
        state.current_step = LifecycleStep.EVALUATE_RESULT

        with patch.object(planner, "process", new_callable=AsyncMock) as mock:
            mock.side_effect = RuntimeError("planner crashed")
            state = await orch.evaluate_result(state)

        assert len(state.plan) == 1
        assert state.plan[0].description == "step"


# ── _plan_context ────────────────────────────────────────────────


class TestPlanContext:
    def test_empty_context(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        ctx = orch._plan_context(state)
        assert ctx == ""

    def test_replan_errors_in_context(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.metadata["replan_errors"] = ["timeout", "connection refused"]
        ctx = orch._plan_context(state)
        assert "timeout" in ctx
        assert "connection refused" in ctx

    def test_replan_feedback_in_context(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.metadata["replan_feedback"] = "Use faster tool"
        ctx = orch._plan_context(state)
        assert "Use faster tool" in ctx
