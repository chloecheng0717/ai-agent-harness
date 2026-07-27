"""
Unit Tests for ExecutorAgent

Tests cover:
- Agent creation and configuration
- Execution model validation
- Dependency resolution (topological sort, cycles, missing deps)
- Process without tools (reasoning steps)
- Process with tools via registry
- Retry mechanism
- Timeout handling
- Error propagation and step skipping
- Reflect method
- Execution history and stats
"""

import asyncio
import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.base import AgentConfig, BaseAgent
from app.agents.specialized.executor import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF,
    DEFAULT_RETRY_DELAY,
    DEFAULT_STEP_TIMEOUT,
    DependencyResolutionError,
    ExecutionTimeoutError,
    StepExecutionError,
    ExecutorAgent,
)
from app.models.agent import AgentStatus, AgentType
from app.models.execution import ExecutionRecord, ExecutionResult, ExecutionStatus
from app.models.state import AgentState, PlanStep
from app.models.tool import (
    ConfirmationRequirement,
    RiskLevel,
    ToolDefinition,
    ToolExecutionRequest,
    ToolPermission,
    ToolResult,
)
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry


# ── Helpers ──────────────────────────────────────────────────────


class StubTool(BaseTool):
    """A stub tool that echoes its input for testing."""

    def __init__(
        self,
        name: str = "stub_tool",
        should_succeed: bool = True,
        output_value: Any = None,
        delay: float = 0,
    ) -> None:
        definition = ToolDefinition(
            name=name,
            description="Stub test tool",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                },
            },
            returns={
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                },
            },
            permissions=[ToolPermission.EXECUTE],
        )
        super().__init__(definition)
        self._should_succeed = should_succeed
        self._output_value = output_value
        self._delay = delay

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if not self._should_succeed:
            return ToolResult(success=False, error="Stub tool failure")
        return ToolResult(
            success=True,
            output=self._output_value or {"result": parameters.get("input", "")},
        )


class FailingTool(BaseTool):
    """A tool that always raises an exception."""

    def __init__(self) -> None:
        definition = ToolDefinition(
            name="failing_tool",
            description="Always fails",
            parameters={"type": "object", "properties": {}},
            returns={"type": "object", "properties": {}},
            permissions=[ToolPermission.EXECUTE],
        )
        super().__init__(definition)

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        raise RuntimeError("Intentional tool failure")


def _make_state_with_plan(plan: List[PlanStep]) -> AgentState:
    state = AgentState()
    state.plan = plan
    return state


def _simple_plan() -> List[PlanStep]:
    return [
        PlanStep(index=0, description="Step one", priority="high"),
        PlanStep(index=1, description="Step two", depends_on=[0]),
    ]


def _tool_plan() -> List[PlanStep]:
    return [
        PlanStep(
            index=0,
            description="Run stub tool",
            tool_name="stub_tool",
            tool_parameters={"input": "hello"},
            priority="high",
        ),
    ]


def _complex_plan() -> List[PlanStep]:
    """Plan with diamond dependency: 0 -> 1, 0 -> 2, 1+2 -> 3."""
    return [
        PlanStep(index=0, description="Init"),
        PlanStep(index=1, description="Branch A", depends_on=[0]),
        PlanStep(index=2, description="Branch B", depends_on=[0]),
        PlanStep(index=3, description="Merge", depends_on=[1, 2]),
    ]


# ── Execution Model Tests ────────────────────────────────────────


class TestExecutionStatus:
    def test_enum_values(self) -> None:
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.SKIPPED.value == "skipped"
        assert ExecutionStatus.TIMED_OUT.value == "timed_out"
        assert ExecutionStatus.RETRYING.value == "retrying"


class TestExecutionResult:
    def test_creation(self) -> None:
        result = ExecutionResult(
            step_index=0,
            tool_name="calc",
            success=True,
            output={"result": 42},
        )
        assert result.step_index == 0
        assert result.tool_name == "calc"
        assert result.success is True
        assert result.output == {"result": 42}
        assert result.error is None
        assert result.execution_time_ms == 0
        assert result.retry_count == 0
        assert result.timeout is False
        assert result.metadata == {}

    def test_failure_result(self) -> None:
        result = ExecutionResult(
            step_index=1,
            tool_name="broken",
            success=False,
            error="something broke",
        )
        assert result.success is False
        assert result.error == "something broke"

    def test_defaults(self) -> None:
        result = ExecutionResult(step_index=0, success=True)
        assert result.tool_name is None
        assert result.output is None
        assert result.metadata == {}


class TestExecutionRecord:
    def test_creation(self) -> None:
        record = ExecutionRecord(step_index=0, description="test step")
        assert record.step_index == 0
        assert record.description == "test step"
        assert record.status == ExecutionStatus.PENDING
        assert record.started_at is None
        assert record.completed_at is None
        assert record.result is None
        assert record.attempts == []

    def test_mark_started(self) -> None:
        record = ExecutionRecord(step_index=0)
        record.mark_started()
        assert record.status == ExecutionStatus.RUNNING
        assert record.started_at is not None

    def test_mark_completed(self) -> None:
        record = ExecutionRecord(step_index=0)
        record.mark_started()
        result = ExecutionResult(step_index=0, success=True, output={"ok": True})
        record.mark_completed(result)
        assert record.status == ExecutionStatus.COMPLETED
        assert record.result is result
        assert len(record.attempts) == 1
        assert record.completed_at is not None
        assert record.duration_ms is not None

    def test_mark_failed(self) -> None:
        record = ExecutionRecord(step_index=0)
        record.mark_started()
        result = ExecutionResult(step_index=0, success=False, error="fail")
        record.mark_failed(result, "fail")
        assert record.status == ExecutionStatus.FAILED
        assert record.error_message == "fail"

    def test_mark_timed_out(self) -> None:
        record = ExecutionRecord(step_index=0)
        record.mark_started()
        record.mark_timed_out("timeout after 30s")
        assert record.status == ExecutionStatus.TIMED_OUT
        assert record.error_message == "timeout after 30s"

    def test_mark_retrying(self) -> None:
        record = ExecutionRecord(step_index=0)
        record.mark_started()
        record.mark_retrying()
        assert record.status == ExecutionStatus.RETRYING

    def test_mark_skipped(self) -> None:
        record = ExecutionRecord(step_index=0)
        record.mark_skipped("dep failed")
        assert record.status == ExecutionStatus.SKIPPED
        assert record.error_message == "dep failed"

    def test_mark_skipped_no_reason(self) -> None:
        record = ExecutionRecord(step_index=0)
        record.mark_skipped()
        assert record.status == ExecutionStatus.SKIPPED
        assert record.error_message is None

    def test_to_dict(self) -> None:
        record = ExecutionRecord(
            step_index=1, description="test", tool_name="calc"
        )
        record.mark_started()
        result = ExecutionResult(step_index=1, success=True)
        record.mark_completed(result)
        d = record.to_dict()
        assert d["step_index"] == 1
        assert d["description"] == "test"
        assert d["tool_name"] == "calc"
        assert d["status"] == "completed"
        assert d["success"] is True
        assert d["started_at"] is not None
        assert d["completed_at"] is not None
        assert d["record_id"] is not None


# ── Agent Init Tests ─────────────────────────────────────────────


class TestExecutorAgentInit:
    def test_creates_with_defaults(self) -> None:
        agent = ExecutorAgent()
        assert agent.config.name == "ExecutorAgent"
        assert agent.config.type == AgentType.EXECUTOR
        assert agent.tool_registry is None
        assert agent.max_retries == DEFAULT_MAX_RETRIES
        assert agent.default_timeout == DEFAULT_STEP_TIMEOUT
        assert agent.execution_history == []

    def test_creates_with_custom_config(self) -> None:
        config = AgentConfig(
            name="CustomExecutor",
            type=AgentType.EXECUTOR,
        )
        registry = ToolRegistry()
        agent = ExecutorAgent(
            config=config,
            tool_registry=registry,
            max_retries=5,
            default_timeout=60,
        )
        assert agent.config.name == "CustomExecutor"
        assert agent.tool_registry is registry
        assert agent.max_retries == 5
        assert agent.default_timeout == 60

    def test_is_base_agent(self) -> None:
        agent = ExecutorAgent()
        assert isinstance(agent, BaseAgent)


# ── Dependency Resolution Tests ──────────────────────────────────


class TestDependencyResolution:
    def test_empty_plan(self) -> None:
        assert ExecutorAgent._resolve_execution_order([]) == []

    def test_single_step(self) -> None:
        steps = [PlanStep(index=0, description="Solo")]
        order = ExecutorAgent._resolve_execution_order(steps)
        assert len(order) == 1
        assert order[0].index == 0

    def test_linear_chain(self) -> None:
        steps = [
            PlanStep(index=0, description="A"),
            PlanStep(index=1, description="B", depends_on=[0]),
            PlanStep(index=2, description="C", depends_on=[1]),
        ]
        order = ExecutorAgent._resolve_execution_order(steps)
        indices = [s.index for s in order]
        assert indices == [0, 1, 2]

    def test_diamond_dependency(self) -> None:
        steps = _complex_plan()
        order = ExecutorAgent._resolve_execution_order(steps)
        indices = [s.index for s in order]
        assert indices[0] == 0  # root first
        assert indices[-1] == 3  # merge last
        assert 1 in indices and 2 in indices

    def test_independent_steps(self) -> None:
        steps = [
            PlanStep(index=0, description="A"),
            PlanStep(index=1, description="B"),
            PlanStep(index=2, description="C"),
        ]
        order = ExecutorAgent._resolve_execution_order(steps)
        indices = [s.index for s in order]
        assert set(indices) == {0, 1, 2}

    def test_missing_dependency_raises(self) -> None:
        steps = [
            PlanStep(index=0, description="A"),
            PlanStep(index=1, description="B", depends_on=[5]),
        ]
        with pytest.raises(DependencyResolutionError, match="non-existent"):
            ExecutorAgent._resolve_execution_order(steps)

    def test_cycle_detection(self) -> None:
        steps = [
            PlanStep(index=0, description="A", depends_on=[1]),
            PlanStep(index=1, description="B", depends_on=[0]),
        ]
        with pytest.raises(DependencyResolutionError, match="cycle"):
            ExecutorAgent._resolve_execution_order(steps)


# ── Process Tests (no tools) ─────────────────────────────────────


class TestProcessWithoutTools:
    @pytest.mark.asyncio
    async def test_no_state_returns_error(self) -> None:
        agent = ExecutorAgent()
        result = await agent.process({})
        assert result["success"] is False
        assert "No AgentState" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_empty_plan_succeeds(self) -> None:
        agent = ExecutorAgent()
        state = AgentState()
        result = await agent.process({"state": state})
        assert result["success"] is True
        assert result["results"] == []
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_reasoning_steps_succeed(self) -> None:
        agent = ExecutorAgent()
        state = _make_state_with_plan(_simple_plan())
        result = await agent.process({"state": state})
        assert result["success"] is True
        assert len(result["results"]) == 2
        assert all(r["success"] for r in result["results"])

    @pytest.mark.asyncio
    async def test_results_stored_in_state(self) -> None:
        agent = ExecutorAgent()
        state = _make_state_with_plan(_simple_plan())
        await agent.process({"state": state})
        assert len(state.results) == 2

    @pytest.mark.asyncio
    async def test_execution_history_populated(self) -> None:
        agent = ExecutorAgent()
        state = _make_state_with_plan(_simple_plan())
        await agent.process({"state": state})
        assert len(agent.execution_history) == 2

    @pytest.mark.asyncio
    async def test_complex_plan_executes_in_order(self) -> None:
        agent = ExecutorAgent()
        state = _make_state_with_plan(_complex_plan())
        result = await agent.process({"state": state})
        assert result["success"] is True
        assert len(result["results"]) == 4


# ── Process Tests (with tools) ───────────────────────────────────


class TestProcessWithTools:
    @pytest.mark.asyncio
    async def test_tool_execution_success(self) -> None:
        registry = ToolRegistry()
        registry.register_instance(StubTool())
        agent = ExecutorAgent(tool_registry=registry)
        state = _make_state_with_plan(_tool_plan())
        result = await agent.process({"state": state})
        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["tool_name"] == "stub_tool"
        assert "stub_tool" in result["tools_used"]

    @pytest.mark.asyncio
    async def test_tool_not_found_records_error(self) -> None:
        registry = ToolRegistry()
        agent = ExecutorAgent(tool_registry=registry)
        state = _make_state_with_plan([
            PlanStep(
                index=0,
                description="Use missing tool",
                tool_name="nonexistent",
                tool_parameters={},
            )
        ])
        result = await agent.process({"state": state})
        assert result["success"] is False
        assert len(result["errors"]) > 0
        assert "not found" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_no_registry_records_error(self) -> None:
        agent = ExecutorAgent()  # no registry
        state = _make_state_with_plan([
            PlanStep(
                index=0,
                description="Use tool",
                tool_name="some_tool",
                tool_parameters={},
            )
        ])
        result = await agent.process({"state": state})
        assert result["success"] is False
        assert "No tool registry" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_tool_failure_records_error(self) -> None:
        registry = ToolRegistry()
        registry.register_instance(StubTool(should_succeed=False))
        agent = ExecutorAgent(tool_registry=registry)
        state = _make_state_with_plan([
            PlanStep(
                index=0,
                description="Failing step",
                tool_name="stub_tool",
                tool_parameters={"input": "test"},
            )
        ])
        result = await agent.process({"state": state})
        assert result["success"] is False
        assert "Stub tool failure" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_tool_exception_records_error(self) -> None:
        registry = ToolRegistry()
        registry.register_instance(FailingTool())
        agent = ExecutorAgent(tool_registry=registry, max_retries=1)
        state = _make_state_with_plan([
            PlanStep(
                index=0,
                description="Exception step",
                tool_name="failing_tool",
                tool_parameters={},
            )
        ])
        result = await agent.process({"state": state})
        assert result["success"] is False
        assert "Intentional tool failure" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_override_registry_from_input(self) -> None:
        registry = ToolRegistry()
        registry.register_instance(StubTool(name="override_tool"))
        agent = ExecutorAgent()  # no default registry
        state = _make_state_with_plan([
            PlanStep(
                index=0,
                description="Use override",
                tool_name="override_tool",
                tool_parameters={"input": "hi"},
            )
        ])
        result = await agent.process({"state": state, "tool_registry": registry})
        assert result["success"] is True
        assert "override_tool" in result["tools_used"]

    @pytest.mark.asyncio
    async def test_no_duplicate_tools_used(self) -> None:
        registry = ToolRegistry()
        registry.register_instance(StubTool())
        agent = ExecutorAgent(tool_registry=registry)
        state = _make_state_with_plan([
            PlanStep(
                index=0,
                description="First use",
                tool_name="stub_tool",
                tool_parameters={"input": "a"},
            ),
            PlanStep(
                index=1,
                description="Second use",
                tool_name="stub_tool",
                tool_parameters={"input": "b"},
            ),
        ])
        result = await agent.process({"state": state})
        assert result["tools_used"] == ["stub_tool"]


# ── Retry Tests ──────────────────────────────────────────────────


class TestRetryMechanism:
    @pytest.mark.asyncio
    async def test_succeeds_after_retry(self) -> None:
        call_count = 0

        class RetryStubTool(BaseTool):
            def __init__(self) -> None:
                super().__init__(ToolDefinition(
                    name="retry_tool",
                    description="Fails first",
                    parameters={"type": "object", "properties": {}},
                    returns={"type": "object", "properties": {}},
                    permissions=[ToolPermission.EXECUTE],
                ))

            async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    return ToolResult(success=False, error="transient failure")
                return ToolResult(success=True, output={"attempt": call_count})

        registry = ToolRegistry()
        registry.register_instance(RetryStubTool())
        agent = ExecutorAgent(
            tool_registry=registry,
            max_retries=3,
        )
        # Patch sleep to avoid real delays
        agent._async_sleep = AsyncMock()

        state = _make_state_with_plan([
            PlanStep(
                index=0,
                description="Retry step",
                tool_name="retry_tool",
                tool_parameters={},
            )
        ])
        result = await agent.process({"state": state})
        assert result["success"] is True
        assert call_count == 2
        assert result["execution_history"][0]["retry_count"] >= 1

    @pytest.mark.asyncio
    async def test_fails_after_max_retries(self) -> None:
        registry = ToolRegistry()
        registry.register_instance(StubTool(should_succeed=False))
        agent = ExecutorAgent(
            tool_registry=registry,
            max_retries=2,
        )
        agent._async_sleep = AsyncMock()

        state = _make_state_with_plan([
            PlanStep(
                index=0,
                description="Always fail",
                tool_name="stub_tool",
                tool_parameters={},
            )
        ])
        result = await agent.process({"state": state})
        assert result["success"] is False
        assert "Failed after 2 attempts" in result["errors"][0]


# ── Timeout Tests ────────────────────────────────────────────────


class TestTimeoutHandling:
    @pytest.mark.asyncio
    async def test_timeout_records_error(self) -> None:
        registry = ToolRegistry()
        registry.register_instance(StubTool(delay=10))
        agent = ExecutorAgent(
            tool_registry=registry,
            default_timeout=1,
            max_retries=1,
        )

        state = _make_state_with_plan([
            PlanStep(
                index=0,
                description="Slow step",
                tool_name="stub_tool",
                tool_parameters={},
            )
        ])
        result = await agent.process({"state": state})
        assert result["success"] is False
        assert any("timed out" in e.lower() for e in result["errors"])
        assert result["execution_history"][0]["status"] == "timed_out"


# ── Dependency Failure Skip Tests ────────────────────────────────


class TestDependencyFailureSkipping:
    @pytest.mark.asyncio
    async def test_dependent_step_skipped_on_failure(self) -> None:
        registry = ToolRegistry()
        registry.register_instance(StubTool(should_succeed=False))
        agent = ExecutorAgent(
            tool_registry=registry,
            max_retries=1,
        )
        agent._async_sleep = AsyncMock()

        state = _make_state_with_plan([
            PlanStep(
                index=0,
                description="Fail first",
                tool_name="stub_tool",
                tool_parameters={},
            ),
            PlanStep(
                index=1,
                description="Depend on first",
                depends_on=[0],
            ),
        ])
        result = await agent.process({"state": state})
        assert result["success"] is False
        # Step 1 should be skipped
        step1_record = [
            r for r in result["execution_history"]
            if r["step_index"] == 1
        ][0]
        assert step1_record["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_independent_steps_not_affected(self) -> None:
        registry = ToolRegistry()
        registry.register_instance(StubTool(should_succeed=False))
        agent = ExecutorAgent(
            tool_registry=registry,
            max_retries=1,
        )
        agent._async_sleep = AsyncMock()

        state = _make_state_with_plan([
            PlanStep(
                index=0,
                description="Fail first",
                tool_name="stub_tool",
                tool_parameters={},
            ),
            PlanStep(
                index=1,
                description="Independent step",
            ),
        ])
        result = await agent.process({"state": state})
        # Step 1 should still execute (not dependent on step 0)
        step1_record = [
            r for r in result["execution_history"]
            if r["step_index"] == 1
        ][0]
        assert step1_record["status"] == "completed"


# ── Reflect Tests ────────────────────────────────────────────────


class TestReflect:
    @pytest.mark.asyncio
    async def test_reflect_on_success(self) -> None:
        agent = ExecutorAgent()
        result = {
            "success": True,
            "results": [
                {"step_index": 0, "success": True, "execution_time_ms": 10},
                {"step_index": 1, "success": True, "execution_time_ms": 20},
            ],
            "errors": [],
            "tools_used": ["stub_tool"],
            "execution_history": [
                {"status": "completed", "retry_count": 0},
                {"status": "completed", "retry_count": 0},
            ],
        }
        insights = await agent.reflect(result)
        assert insights["total_steps"] == 2
        assert insights["successful_steps"] == 2
        assert insights["failed_steps"] == 0
        assert insights["overall_success"] is True
        assert "stub_tool" in insights["unique_tools_used"]

    @pytest.mark.asyncio
    async def test_reflect_on_failure(self) -> None:
        agent = ExecutorAgent()
        result = {
            "success": False,
            "results": [
                {"step_index": 0, "success": True, "execution_time_ms": 10},
                {"step_index": 1, "success": False, "execution_time_ms": 5},
            ],
            "errors": ["Step 1 failed"],
            "tools_used": ["stub_tool"],
            "execution_history": [
                {"status": "completed", "retry_count": 0},
                {"status": "failed", "retry_count": 2},
            ],
        }
        insights = await agent.reflect(result)
        assert insights["failed_steps"] == 1
        assert insights["total_retries"] == 2
        assert insights["overall_success"] is False

    @pytest.mark.asyncio
    async def test_reflect_stores_memory(self) -> None:
        agent = ExecutorAgent()
        result = {
            "success": True,
            "results": [],
            "errors": [],
            "tools_used": [],
            "execution_history": [],
        }
        await agent.reflect(result)
        assert agent.get_memory("last_reflection") is not None


# ── Execution History & Stats Tests ──────────────────────────────


class TestExecutionHistory:
    def test_get_history_empty(self) -> None:
        agent = ExecutorAgent()
        assert agent.get_execution_history() == []

    def test_get_history_all(self) -> None:
        agent = ExecutorAgent()
        agent.execution_history = [
            ExecutionRecord(step_index=0),
            ExecutionRecord(step_index=1),
        ]
        assert len(agent.get_execution_history()) == 2

    def test_get_history_by_step(self) -> None:
        agent = ExecutorAgent()
        agent.execution_history = [
            ExecutionRecord(step_index=0),
            ExecutionRecord(step_index=1),
            ExecutionRecord(step_index=0),
        ]
        assert len(agent.get_execution_history(step_index=0)) == 2
        assert len(agent.get_execution_history(step_index=1)) == 1

    def test_get_execution_stats(self) -> None:
        agent = ExecutorAgent()
        r1 = ExecutionRecord(step_index=0)
        r1.mark_started()
        r1.mark_completed(ExecutionResult(step_index=0, success=True))

        r2 = ExecutionRecord(step_index=1)
        r2.mark_started()
        r2.mark_failed(
            ExecutionResult(step_index=1, success=False, error="fail"),
            "fail",
        )

        r3 = ExecutionRecord(step_index=2)
        r3.mark_skipped("dep failed")

        agent.execution_history = [r1, r2, r3]
        stats = agent.get_execution_stats()
        assert stats["total_steps"] == 3
        assert stats["completed"] == 1
        assert stats["failed"] == 1
        assert stats["skipped"] == 1
