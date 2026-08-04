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
    extract_llm_tool_name,
    extract_llm_tool_params,
    parse_llm_plan_json,
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


# ── LLM plan key normalization ───────────────────────────────────


class TestExtractLLMToolParams:
    """Regression tests for LLM plan key normalization.

    Models sometimes author plan steps with tool-call style keys
    (``args``, ``tool``) instead of the documented ``tool_parameters`` /
    ``tool_name``.  The arguments must survive parsing, otherwise a step
    that was supposed to run ``python`` with real ``code`` ends up with an
    empty parameter dict and fails with "Either 'code' or 'script_path'
    must be provided".
    """

    def test_canonical_tool_parameters_dict(self) -> None:
        params = extract_llm_tool_params({"tool_parameters": {"code": "x"}})
        assert params == {"code": "x"}

    def test_args_alias_used_when_tool_parameters_missing(self) -> None:
        params = extract_llm_tool_params({"args": {"code": "print(1)"}})
        assert params == {"code": "print(1)"}

    def test_args_json_string_is_parsed(self) -> None:
        params = extract_llm_tool_params({"args": '{"code": "print(1)"}'})
        assert params == {"code": "print(1)"}

    def test_tool_parameters_preferred_over_args(self) -> None:
        params = extract_llm_tool_params(
            {"tool_parameters": {"a": 1}, "args": {"b": 2}}
        )
        assert params == {"a": 1}

    def test_missing_or_invalid_params_returns_empty(self) -> None:
        assert extract_llm_tool_params({}) == {}
        assert extract_llm_tool_params({"tool_parameters": None}) == {}
        assert extract_llm_tool_params({"args": 42}) == {}
        assert extract_llm_tool_params({"args": "not json"}) == {}


class TestExtractLLMToolName:
    """Regression tests for tool-name key normalization."""

    def test_canonical_tool_name(self) -> None:
        assert extract_llm_tool_name({"tool_name": "python"}) == "python"

    def test_tool_alias_used_when_tool_name_missing(self) -> None:
        assert extract_llm_tool_name({"tool": "python"}) == "python"

    def test_tool_name_preferred_over_tool(self) -> None:
        assert (
            extract_llm_tool_name({"tool_name": "python", "tool": "nmap"})
            == "python"
        )

    def test_missing_or_non_string_returns_none(self) -> None:
        assert extract_llm_tool_name({}) is None
        assert extract_llm_tool_name({"tool": None}) is None
        assert extract_llm_tool_name({"tool": 42}) is None


# ── LLM plan JSON parsing ────────────────────────────────────────


class TestParseLLMPlanJson:
    """Regression tests for robust parsing of LLM plan output.

    LLMs wrap plans in markdown fences and prose, or truncate them
    mid-string.  ``parse_llm_plan_json`` must recover usable arrays and
    return ``None`` (never raise) for unusable output so the caller's
    fallback planner runs instead of crashing.
    """

    def test_valid_array(self) -> None:
        data = parse_llm_plan_json(
            '[{"index": 0, "description": "run", "tool_name": "strings"}]'
        )
        assert data == [{"index": 0, "description": "run", "tool_name": "strings"}]

    def test_markdown_fenced_array(self) -> None:
        raw = "```json\n[{\"index\": 0, \"description\": \"run\"}]\n```"
        data = parse_llm_plan_json(raw)
        assert data == [{"index": 0, "description": "run"}]

    def test_prose_wrapped_array(self) -> None:
        """Prose before/after the array used to break json.loads."""
        raw = (
            "Here is the plan:\n"
            '[{"index": 0, "description": "extract strings"}]\n'
            "Good luck!"
        )
        data = parse_llm_plan_json(raw)
        assert data == [{"index": 0, "description": "extract strings"}]

    def test_prose_and_fences_wrapped_array(self) -> None:
        raw = (
            "Sure thing:\n```json\n"
            '[{"index": 0, "description": "decode"}, {"index": 1, "description": "done"}]\n'
            "```\nHope this helps."
        )
        data = parse_llm_plan_json(raw)
        assert len(data) == 2

    def test_unterminated_string_returns_none(self) -> None:
        """A truncated reply must not crash the run."""
        raw = '[{"index": 0, "description": "unterminated'
        assert parse_llm_plan_json(raw) is None

    def test_valid_but_not_array_returns_none(self) -> None:
        assert parse_llm_plan_json('{"key": "value"}') is None

    def test_invalid_text_returns_none(self) -> None:
        assert parse_llm_plan_json("not json at all") is None

    def test_empty_and_non_string_returns_none(self) -> None:
        assert parse_llm_plan_json("") is None
        assert parse_llm_plan_json(None) is None  # type: ignore[arg-type]


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
