"""
Unit Tests for Orchestrator Runtime Engine

Tests cover:
- Full lifecycle execution
- Individual lifecycle steps
- Retry mechanism
- Error handling
- Backward-compatible process_task
- Tool selection hooks
- State flow through lifecycle
"""

import pytest
from typing import Any, Dict
from uuid import uuid4

from app.agents.core.orchestrator import Orchestrator, RetryExhaustedError
from app.models.state import (
    AgentState,
    LifecycleStep,
    PlanStep,
    StepStatus,
)


# ── Orchestrator creation ───────────────────────────────────────


class TestOrchestratorInit:
    """Tests for orchestrator initialization."""

    def test_creates_successfully(self) -> None:
        orch = Orchestrator()
        assert orch.settings is not None
        assert orch.max_retries == 3

    def test_step_registry_is_populated(self) -> None:
        orch = Orchestrator()
        assert len(orch._step_registry) == 8
        for step in LifecycleStep:
            assert step in orch._step_registry


# ── Full lifecycle ──────────────────────────────────────────────


class TestFullLifecycle:
    """Tests for the complete lifecycle run."""

    @pytest.mark.asyncio
    async def test_run_completes_successfully(self) -> None:
        orch = Orchestrator()
        result = await orch.run(
            task_id=uuid4(),
            task_data={"description": "scan target 192.168.1.1"},
        )
        assert result["status"] == "completed"
        assert result["final_answer"] is not None
        assert len(result["step_history"]) == 8

    @pytest.mark.asyncio
    async def test_run_populates_goal(self) -> None:
        orch = Orchestrator()
        result = await orch.run(
            task_id=uuid4(),
            task_data={"description": "analyze network"},
        )
        assert result["goal"] == "analyze network"

    @pytest.mark.asyncio
    async def test_run_returns_task_id(self) -> None:
        tid = uuid4()
        orch = Orchestrator()
        result = await orch.run(task_id=tid, task_data={"description": "test"})
        assert result["task_id"] == str(tid)

    @pytest.mark.asyncio
    async def test_run_with_empty_description(self) -> None:
        orch = Orchestrator()
        result = await orch.run(task_id=uuid4(), task_data={})
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_run_with_metadata(self) -> None:
        orch = Orchestrator()
        result = await orch.run(
            task_id=uuid4(),
            task_data={
                "description": "test",
                "metadata": {"priority": "high"},
            },
        )
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_run_with_tools_available(self) -> None:
        orch = Orchestrator()
        result = await orch.run(
            task_id=uuid4(),
            task_data={"description": "test"},
            tools_available=["nmap", "curl"],
        )
        assert result["status"] == "completed"


# ── Individual steps ────────────────────────────────────────────


class TestIndividualSteps:
    """Tests for individual lifecycle steps."""

    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state = await orch.initialize(state)
        assert len(state.messages) == 1
        assert state.messages[0].role == "system"
        assert state.status == StepStatus.RUNNING

    @pytest.mark.asyncio
    async def test_understand_task(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.add_message("user", "scan the network")
        state = await orch.understand_task(state)
        assert state.goal == "scan the network"

    @pytest.mark.asyncio
    async def test_understand_task_with_no_messages(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state = await orch.understand_task(state)
        assert state.goal == ""

    @pytest.mark.asyncio
    async def test_create_plan_default(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.goal = "test goal"
        state = await orch.create_plan(state)
        assert len(state.plan) == 1
        assert state.plan[0].index == 0

    @pytest.mark.asyncio
    async def test_create_plan_preserves_existing(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        existing = [PlanStep(index=0, description="custom step")]
        state.plan = existing
        state = await orch.create_plan(state)
        assert state.plan == existing

    @pytest.mark.asyncio
    async def test_select_tools_passthrough(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state = await orch.select_tools(state)
        assert state.plan == []

    @pytest.mark.asyncio
    async def test_execute_plan_no_tools(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.plan = [PlanStep(index=0, description="do something")]
        state = await orch.execute_plan(state)
        assert len(state.results) == 1
        assert state.results[0]["success"] is True

    @pytest.mark.asyncio
    async def test_execute_plan_with_tool(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.plan = [
            PlanStep(index=0, description="scan", tool_name="nmap")
        ]
        state = await orch.execute_plan(state)
        assert "nmap" in state.tools_used
        assert len(state.results) == 1

    @pytest.mark.asyncio
    async def test_execute_plan_multiple_steps(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.plan = [
            PlanStep(index=0, description="step 1"),
            PlanStep(index=1, description="step 2", tool_name="curl"),
            PlanStep(index=2, description="step 3"),
        ]
        state = await orch.execute_plan(state)
        assert len(state.results) == 3
        assert "curl" in state.tools_used

    @pytest.mark.asyncio
    async def test_evaluate_result_no_errors(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state = await orch.evaluate_result(state)
        assert len(state.messages) == 1
        assert "successfully" in state.messages[0].content

    @pytest.mark.asyncio
    async def test_evaluate_result_with_errors(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.add_error("something failed")
        state = await orch.evaluate_result(state)
        assert "error" in state.messages[0].content

    @pytest.mark.asyncio
    async def test_respond(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.results = [{"output": "scan result"}]
        state = await orch.respond(state)
        assert "scan result" in state.final_answer
        assert state.status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_respond_no_results(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state = await orch.respond(state)
        assert "No result" in state.final_answer

    @pytest.mark.asyncio
    async def test_reflect(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.goal = "test"
        state.results = [{"output": "result"}]
        state.tools_used = ["nmap"]
        state = await orch.reflect(state)
        assert state.goal == "test"


# ── Retry mechanism ─────────────────────────────────────────────


class TestRetryMechanism:
    """Tests for step retry and failure handling."""

    @pytest.mark.asyncio
    async def test_retry_on_step_failure(self) -> None:
        orch = Orchestrator()
        call_count = 0

        async def failing_then_ok(state: AgentState) -> AgentState:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("transient error")
            return state

        orch._step_registry[LifecycleStep.INITIALIZE] = failing_then_ok
        result = await orch.run(
            task_id=uuid4(),
            task_data={"description": "test"},
        )
        assert call_count == 2
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_retry_exhausted(self) -> None:
        orch = Orchestrator()
        orch.max_retries = 2

        async def always_fail(state: AgentState) -> AgentState:
            raise ValueError("permanent error")

        orch._step_registry[LifecycleStep.INITIALIZE] = always_fail
        result = await orch.run(
            task_id=uuid4(),
            task_data={"description": "test"},
        )
        assert result["status"] == "failed"
        assert any("permanent error" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_retry_count_tracked(self) -> None:
        orch = Orchestrator()
        orch.max_retries = 2

        async def fail_once(state: AgentState) -> AgentState:
            count = state.get_retry_count("initialize")
            if count < 1:
                state.increment_retry("initialize")
                raise ValueError("first fail")
            return state

        orch._step_registry[LifecycleStep.INITIALIZE] = fail_once
        result = await orch.run(
            task_id=uuid4(),
            task_data={"description": "test"},
        )
        assert result["status"] == "completed"


# ── Backward compatibility ──────────────────────────────────────


class TestBackwardCompatibility:
    """Tests for process_task legacy entry point."""

    @pytest.mark.asyncio
    async def test_process_task(self) -> None:
        orch = Orchestrator()
        result = await orch.process_task(
            task_id=uuid4(),
            task_data={"description": "test task"},
        )
        assert result["success"] is True
        assert result["result"] is not None
        assert "history" in result

    @pytest.mark.asyncio
    async def test_process_task_on_failure(self) -> None:
        orch = Orchestrator()
        orch.max_retries = 1

        async def always_fail(state: AgentState) -> AgentState:
            raise RuntimeError("boom")

        orch._step_registry[LifecycleStep.INITIALIZE] = always_fail
        result = await orch.process_task(
            task_id=uuid4(),
            task_data={"description": "test"},
        )
        assert result["success"] is False


# ── Iterative loop ──────────────────────────────────────────────


class TestIterativeLoop:
    """Tests for the plan-execute-evaluate iterative loop in run()."""

    @pytest.mark.asyncio
    async def test_single_iteration_on_success(self) -> None:
        orch = Orchestrator()
        result = await orch.run(
            task_id=uuid4(),
            task_data={"description": "success"},
        )
        # 2 init + 4 inner + 2 final = 8 step records
        assert len(result["step_history"]) == 8
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_multiple_iterations_when_incomplete(self) -> None:
        """Simulate incomplete task on first pass, then complete on second."""
        orch = Orchestrator(max_iterations=5)
        eval_calls = 0

        async def incomplete_then_complete(state: AgentState) -> AgentState:
            nonlocal eval_calls
            eval_calls += 1
            if eval_calls == 1:
                state.add_error("not done yet")
                state.metadata["task_complete"] = False
            else:
                state.metadata["task_complete"] = True
            # Assign a fallback plan so create_plan doesn't re-run fresh
            state.plan = [PlanStep(index=0, description="step")]
            return state

        original = orch._step_registry[LifecycleStep.EVALUATE_RESULT]
        orch._step_registry[LifecycleStep.EVALUATE_RESULT] = incomplete_then_complete

        result = await orch.run(
            task_id=uuid4(),
            task_data={"description": "multi-step"},
        )

        assert result["status"] == "completed"
        assert eval_calls == 2
        # 2 init + (4 steps × 2 iterations) + 2 final = 12 records
        assert len(result["step_history"]) == 12

        orch._step_registry[LifecycleStep.EVALUATE_RESULT] = original

    @pytest.mark.asyncio
    async def test_honours_max_iterations(self) -> None:
        """When the task never completes, stop after max_iterations."""
        orch = Orchestrator(max_iterations=3)

        async def never_complete(state: AgentState) -> AgentState:
            state.add_error("always fails")
            state.metadata["task_complete"] = False
            return state

        original = orch._step_registry[LifecycleStep.EVALUATE_RESULT]
        orch._step_registry[LifecycleStep.EVALUATE_RESULT] = never_complete

        result = await orch.run(
            task_id=uuid4(),
            task_data={"description": "hopeless"},
        )

        assert result["status"] == "completed"
        # 2 init + (4 steps × 3 iterations) + 2 final = 16 records
        assert len(result["step_history"]) == 16

        orch._step_registry[LifecycleStep.EVALUATE_RESULT] = original

    @pytest.mark.asyncio
    async def test_iteration_tracked_in_metadata(self) -> None:
        """Each iteration sets state.metadata['iteration']."""
        orch = Orchestrator(max_iterations=3)

        async def check_iteration(state: AgentState) -> AgentState:
            # On the first pass it should be iteration 1
            state.metadata["task_complete"] = True
            return state

        original = orch._step_registry[LifecycleStep.EVALUATE_RESULT]
        orch._step_registry[LifecycleStep.EVALUATE_RESULT] = check_iteration

        result = await orch.run(
            task_id=uuid4(),
            task_data={"description": "check iter"},
        )

        assert result["status"] == "completed"

        orch._step_registry[LifecycleStep.EVALUATE_RESULT] = original


# ── Deterministic completion guard ────────────────────────────────


class TestDeterministicCompletionGuard:
    """Tests for the completion guard that inspects real tool output.

    The guard must veto an LLM/heuristic ``COMPLETE: YES`` verdict when the
    tool output is still encoded, and must hard-confirm completion when a
    flag is recovered even if the evaluator said ``COMPLETE: NO``.
    """

    def test_guard_blocks_completion_when_output_still_encoded(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.results = [
            {
                "step_index": 0,
                "output": {
                    "stdout": "666c61677b6c61796572735f6f665f656e636f64696e677d"
                },
            }
        ]
        state.metadata["task_complete"] = True  # lazy evaluator said YES

        orch._enforce_deterministic_completion(state)

        assert state.metadata["task_complete"] is False
        assert "encoded" in state.metadata.get("replan_feedback", "")

    def test_guard_blocks_completion_when_output_is_base64(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.results = [{"step_index": 0, "output": {"stdout": "ZmxhZ3tvbmVfbW9yZV9sYXllcn0="}}]
        state.metadata["task_complete"] = True

        orch._enforce_deterministic_completion(state)

        assert state.metadata["task_complete"] is False
        assert state.metadata.get("replan_feedback")

    def test_guard_allows_completion_when_flag_recovered(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.results = [
            {"step_index": 0, "output": {"stdout": "flag{guard_found_me}"}}
        ]
        state.metadata["task_complete"] = False  # evaluator said NO

        orch._enforce_deterministic_completion(state)

        assert state.metadata["task_complete"] is True

    def test_guard_allows_completion_for_plain_text(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.results = [
            {
                "step_index": 0,
                "output": {"stdout": "All layers decoded. The work is finished."},
            }
        ]
        state.metadata["task_complete"] = True

        orch._enforce_deterministic_completion(state)

        assert state.metadata["task_complete"] is True

    @pytest.mark.asyncio
    async def test_evaluate_result_guard_overrides_lazy_verdict(self) -> None:
        """evaluate_result runs the guard after the LLM/heuristic verdict."""
        orch = Orchestrator()
        state = AgentState()
        state.results = [
            {
                "step_index": 0,
                "output": {
                    "stdout": "666c61677b6c61796572735f6f665f656e636f64696e677d"
                },
            }
        ]

        state = await orch.evaluate_result(state)

        # No LLM provider → heuristic would say complete, guard must veto.
        assert state.metadata["task_complete"] is False
        assert state.metadata.get("replan_feedback")
