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


# ── evaluate_result task_complete detection ───────────────────────


class TestEvaluateResultCompletion:
    """Tests that evaluate_result correctly sets the task_complete flag."""

    @pytest.mark.asyncio
    async def test_sets_complete_false_when_errors_exist(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.goal = "scan network"
        state.add_error("nmap timed out")
        state = await orch.evaluate_result(state)
        assert state.metadata.get("task_complete") is False

    @pytest.mark.asyncio
    async def test_sets_complete_true_when_no_errors(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.goal = "test"
        state.results = [{"step_index": 0, "output": "done", "success": True}]
        state = await orch.evaluate_result(state)
        assert state.metadata.get("task_complete") is True

    @pytest.mark.asyncio
    async def test_planner_not_called_in_evaluate_result(self) -> None:
        """The planner is no longer called inside evaluate_result;
        replanning is handled by the outer loop in run()."""
        planner = PlannerAgent()
        orch = Orchestrator(planner=planner)

        state = AgentState()
        state.goal = "test"
        state.add_error("error")
        with patch.object(planner, "process", new_callable=AsyncMock) as mock:
            state = await orch.evaluate_result(state)
            mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_parses_llm_complete_yes(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response("COMPLETE: YES\nAll goals met.")
            state = AgentState()
            state.goal = "test"
            state.results = [{"step_index": 0, "output": "ok"}]
            state = await orch.evaluate_result(state)
            assert state.metadata.get("task_complete") is True

    @pytest.mark.asyncio
    async def test_parses_llm_complete_no(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response("COMPLETE: NO\nResult does not satisfy goal.")
            state = AgentState()
            state.goal = "test"
            state.results = [{"step_index": 0, "output": "partial"}]
            state = await orch.evaluate_result(state)
            assert state.metadata.get("task_complete") is False

    @pytest.mark.asyncio
    async def test_unclear_llm_verdict_is_not_complete(self) -> None:
        """A verdict without an explicit COMPLETE: YES prefix must not stop
        the loop, even when no errors were recorded."""
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response("Some ambiguous evaluation without verdict.")
            state = AgentState()
            state.goal = "test"
            state.results = [{"step_index": 0, "output": "ok"}]
            state = await orch.evaluate_result(state)
            assert state.metadata.get("task_complete") is False


# ── Iterative loop (run) ─────────────────────────────────────────


class TestIterativeLoop:
    """Tests that run() loops back when evaluate_result returns incomplete."""

    @pytest.mark.asyncio
    async def test_completes_in_one_iteration_when_successful(self) -> None:
        orch = Orchestrator(max_iterations=5)
        result = await orch.run(
            task_id=uuid4(),
            task_data={"description": "simple task"},
        )
        assert result["status"] == "completed"
        assert result["final_answer"] is not None

    @pytest.mark.asyncio
    async def test_replans_when_task_incomplete(self) -> None:
        """Simulate a failure on the first execution that triggers a
        second plan-execute-evaluate cycle."""
        orch = Orchestrator(max_iterations=3)
        exec_count = 0

        async def counting_execute(state: AgentState) -> AgentState:
            nonlocal exec_count
            exec_count += 1
            # Fallback: create a single-step plan if none exists
            if not state.plan:
                state.plan = [PlanStep(index=0, description="auto step")]
            # Execute plan steps (minimal — no tool registry)
            for ps in state.plan:
                state.results.append({
                    "step_index": ps.index,
                    "description": ps.description,
                    "success": True,
                    "output": f"result from iteration {exec_count}",
                })
            return state

        original_execute = orch._step_registry[LifecycleStep.EXECUTE_PLAN]
        orch._step_registry[LifecycleStep.EXECUTE_PLAN] = counting_execute

        # Override evaluate_result to report incomplete on first call,
        # complete afterwards.
        eval_count = 0

        async def conditional_eval(state: AgentState) -> AgentState:
            nonlocal eval_count
            eval_count += 1
            if eval_count == 1:
                state.add_error("transient error")
                state.metadata["task_complete"] = False
            else:
                state.metadata["task_complete"] = True
            return state

        orch._step_registry[LifecycleStep.EVALUATE_RESULT] = conditional_eval

        result = await orch.run(
            task_id=uuid4(),
            task_data={"description": "retry task"},
        )

        assert result["status"] == "completed"
        assert exec_count == 2
        assert eval_count == 2

        # Restore original handlers for other tests
        orch._step_registry[LifecycleStep.EXECUTE_PLAN] = original_execute
        orch._step_registry[LifecycleStep.EVALUATE_RESULT] = orch.evaluate_result

    @pytest.mark.asyncio
    async def test_stops_after_max_iterations(self) -> None:
        """When the task never completes, the loop stops after max_iterations."""
        orch = Orchestrator(max_iterations=3)

        async def always_incomplete(state: AgentState) -> AgentState:
            state.add_error("persistent error")
            state.metadata["task_complete"] = False
            return state

        original = orch._step_registry[LifecycleStep.EVALUATE_RESULT]
        orch._step_registry[LifecycleStep.EVALUATE_RESULT] = always_incomplete

        result = await orch.run(
            task_id=uuid4(),
            task_data={"description": "failing task"},
        )

        assert result["status"] == "completed"
        assert "persistent error" in result["errors"]
        # 4 inner steps × 3 iterations + 2 init + 2 final = 16 step records
        assert len(result["step_history"]) == 16

        orch._step_registry[LifecycleStep.EVALUATE_RESULT] = original


# ── _plan_context ────────────────────────────────────────────────


class TestPlanContext:
    def test_empty_context(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        ctx = orch._plan_context(state)
        assert ctx == "Iteration: 1"

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
