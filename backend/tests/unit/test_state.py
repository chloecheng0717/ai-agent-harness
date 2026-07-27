"""
Unit Tests for AgentState Model

Tests cover:
- State creation and defaults
- Message handling
- Error tracking
- Step recording
- Retry counting
- Token usage tracking
- Plan step handling
- Result serialization
"""

import pytest
from uuid import uuid4

from app.models.state import (
    AgentState,
    LifecycleStep,
    Message,
    PlanStep,
    StepRecord,
    StepStatus,
    TokenUsage,
)


# ── TokenUsage ──────────────────────────────────────────────────


class TestTokenUsage:
    """Tests for the TokenUsage model."""

    def test_default_values(self) -> None:
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.requests == 0
        assert usage.model == ""

    def test_add_single(self) -> None:
        usage = TokenUsage()
        usage.add(prompt=100, completion=50, model="gpt-4")
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.requests == 1
        assert usage.model == "gpt-4"

    def test_add_multiple(self) -> None:
        usage = TokenUsage()
        usage.add(prompt=100, completion=50, model="gpt-4")
        usage.add(prompt=200, completion=100, model="gpt-4")
        assert usage.prompt_tokens == 300
        assert usage.completion_tokens == 150
        assert usage.total_tokens == 450
        assert usage.requests == 2

    def test_add_without_model(self) -> None:
        usage = TokenUsage()
        usage.add(prompt=10, completion=5)
        assert usage.model == ""
        usage.add(prompt=10, completion=5, model="gpt-3.5")
        assert usage.model == "gpt-3.5"


# ── Message ─────────────────────────────────────────────────────


class TestMessage:
    """Tests for the Message model."""

    def test_creation(self) -> None:
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.name is None
        assert msg.timestamp is not None
        assert msg.metadata == {}

    def test_with_metadata(self) -> None:
        msg = Message(role="assistant", content="hi", metadata={"source": "llm"})
        assert msg.metadata == {"source": "llm"}


# ── PlanStep ────────────────────────────────────────────────────


class TestPlanStep:
    """Tests for the PlanStep model."""

    def test_minimal(self) -> None:
        step = PlanStep(index=0, description="do something")
        assert step.index == 0
        assert step.tool_name is None
        assert step.tool_parameters == {}
        assert step.depends_on == []
        assert step.rationale == ""

    def test_full(self) -> None:
        step = PlanStep(
            index=1,
            description="scan target",
            tool_name="nmap",
            tool_parameters={"target": "192.168.1.1"},
            depends_on=[0],
            rationale="need to discover open ports",
        )
        assert step.tool_name == "nmap"
        assert step.depends_on == [0]


# ── AgentState ──────────────────────────────────────────────────


class TestAgentState:
    """Tests for the AgentState model."""

    def test_default_creation(self) -> None:
        state = AgentState()
        assert state.task_id is not None
        assert state.goal == ""
        assert state.messages == []
        assert state.plan == []
        assert state.tools_used == []
        assert state.results == []
        assert state.errors == []
        assert state.final_answer is None
        assert state.current_step == LifecycleStep.INITIALIZE
        assert state.status == StepStatus.PENDING
        assert state.metadata == {}
        assert state.token_usage.total_tokens == 0
        assert state.step_history == []
        assert state.retry_counts == {}

    def test_add_message(self) -> None:
        state = AgentState()
        msg = state.add_message("user", "hello")
        assert len(state.messages) == 1
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_add_message_with_metadata(self) -> None:
        state = AgentState()
        state.add_message("tool", "result", tool_name="nmap")
        assert state.messages[0].metadata == {"tool_name": "nmap"}

    def test_add_error(self) -> None:
        state = AgentState()
        state.add_error("something went wrong")
        assert len(state.errors) == 1
        assert state.errors[0] == "something went wrong"

    def test_add_multiple_errors(self) -> None:
        state = AgentState()
        state.add_error("error 1")
        state.add_error("error 2")
        assert len(state.errors) == 2

    def test_retry_count(self) -> None:
        state = AgentState()
        assert state.get_retry_count("my_step") == 0
        assert state.increment_retry("my_step") == 1
        assert state.increment_retry("my_step") == 2
        assert state.get_retry_count("my_step") == 2

    def test_to_result(self) -> None:
        state = AgentState()
        state.final_answer = "done"
        state.status = StepStatus.COMPLETED
        state.tools_used = ["nmap"]
        result = state.to_result()
        assert result["status"] == "completed"
        assert result["final_answer"] == "done"
        assert result["tools_used"] == ["nmap"]
        assert "token_usage" in result
        assert "step_history" in result


# ── StepRecord ──────────────────────────────────────────────────


class TestStepRecord:
    """Tests for the StepRecord model."""

    def test_defaults(self) -> None:
        record = StepRecord(step=LifecycleStep.INITIALIZE, status=StepStatus.RUNNING)
        assert record.step == LifecycleStep.INITIALIZE
        assert record.status == StepStatus.RUNNING
        assert record.started_at is not None
        assert record.completed_at is None
        assert record.duration_ms is None
        assert record.output is None
        assert record.error is None
        assert record.retry_count == 0


# ── record_step helper ──────────────────────────────────────────


class TestRecordStep:
    """Tests for AgentState.record_step helper."""

    def test_record_step_creates_running_record(self) -> None:
        state = AgentState()
        record = state.record_step(
            LifecycleStep.INITIALIZE, StepStatus.RUNNING
        )
        assert record.status == StepStatus.RUNNING
        assert len(state.step_history) == 1

    def test_record_step_completes_running_record(self) -> None:
        state = AgentState()
        state.record_step(LifecycleStep.INITIALIZE, StepStatus.RUNNING)
        record = state.record_step(
            LifecycleStep.INITIALIZE, StepStatus.COMPLETED
        )
        assert record.status == StepStatus.COMPLETED
        assert record.completed_at is not None
        assert record.duration_ms is not None
        assert record.duration_ms >= 0
        assert len(state.step_history) == 1

    def test_record_step_with_error(self) -> None:
        state = AgentState()
        state.record_step(LifecycleStep.CREATE_PLAN, StepStatus.RUNNING)
        record = state.record_step(
            LifecycleStep.CREATE_PLAN, StepStatus.FAILED, error="boom"
        )
        assert record.status == StepStatus.FAILED
        assert record.error == "boom"

    def test_record_step_with_output(self) -> None:
        state = AgentState()
        state.record_step(LifecycleStep.EXECUTE_PLAN, StepStatus.RUNNING)
        record = state.record_step(
            LifecycleStep.EXECUTE_PLAN,
            StepStatus.COMPLETED,
            output={"key": "value"},
        )
        assert record.output == {"key": "value"}
