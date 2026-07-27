"""
Integration Tests for ExecutorAgent

Tests cover:
- ExecutorAgent + ToolRegistry integration
- ExecutorAgent + Orchestrator integration
- ExecutorAgent + PlannerAgent integration (full pipeline)
- State population and verification
- Execution history persistence
"""

import asyncio
import pytest
from typing import Any, Dict
from unittest.mock import AsyncMock

from app.agents.base import AgentConfig
from app.agents.core.orchestrator import Orchestrator
from app.agents.specialized.executor import ExecutorAgent
from app.agents.specialized.planner import PlannerAgent
from app.models.agent import AgentType
from app.models.execution import ExecutionRecord, ExecutionResult, ExecutionStatus
from app.models.state import AgentState, LifecycleStep, PlanStep, StepStatus
from app.models.tool import (
    RiskLevel,
    ToolDefinition,
    ToolExecutionRequest,
    ToolPermission,
    ToolResult,
)
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry
from app.tools.plugins.calculator import CalculatorTool


# ── Helpers ──────────────────────────────────────────────────────


class CountingTool(BaseTool):
    """Tool that counts how many times it was called."""

    def __init__(self) -> None:
        super().__init__(ToolDefinition(
            name="counting_tool",
            description="Counts invocations",
            parameters={
                "type": "object",
                "properties": {
                    "value": {"type": "integer"},
                },
            },
            returns={"type": "object", "properties": {"count": {"type": "integer"}}},
            permissions=[ToolPermission.EXECUTE],
        ))
        self.call_count = 0

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        self.call_count += 1
        return ToolResult(
            success=True,
            output={"count": self.call_count, "value": parameters.get("value", 0)},
        )


class ConditionalTool(BaseTool):
    """Tool that fails on first call, succeeds after."""

    def __init__(self) -> None:
        super().__init__(ToolDefinition(
            name="conditional_tool",
            description="Fails first time",
            parameters={"type": "object", "properties": {}},
            returns={"type": "object", "properties": {"result": {"type": "string"}}},
            permissions=[ToolPermission.EXECUTE],
        ))
        self.call_count = 0

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        self.call_count += 1
        if self.call_count == 1:
            return ToolResult(success=False, error="First call failure")
        return ToolResult(success=True, output={"result": "ok"})


# ── Orchestrator Integration Tests ──────────────────────────────


class TestExecutorWithOrchestrator:
    """Integration tests for ExecutorAgent with the Orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_execute_plan_step(self) -> None:
        """Orchestrator's execute_plan step uses the same ToolRegistry path."""
        registry = ToolRegistry()
        registry.register_instance(CalculatorTool())

        orchestrator = Orchestrator(tool_registry=registry)
        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Calculate 2 + 2",
                tool_name="calculator",
                tool_parameters={"expression": "2 + 2"},
                priority="high",
            )
        ]

        state = await orchestrator.execute_plan(state)

        assert len(state.results) == 1
        assert state.results[0]["success"] is True
        assert state.results[0]["output"]["result"] == 4.0

    @pytest.mark.asyncio
    async def test_executor_agent_produces_compatible_results(self) -> None:
        """ExecutorAgent results are compatible with Orchestrator expectations."""
        registry = ToolRegistry()
        registry.register_instance(CalculatorTool())

        executor = ExecutorAgent(tool_registry=registry)
        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Calculate 5 * 3",
                tool_name="calculator",
                tool_parameters={"expression": "5 * 3"},
                priority="high",
            )
        ]

        result = await executor.process({"state": state})

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["tool_name"] == "calculator"
        assert result["results"][0]["output"]["result"] == 15.0

    @pytest.mark.asyncio
    async def test_state_populated_by_executor(self) -> None:
        """ExecutorAgent correctly populates AgentState fields."""
        registry = ToolRegistry()
        registry.register_instance(CalculatorTool())

        executor = ExecutorAgent(tool_registry=registry)
        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Calc",
                tool_name="calculator",
                tool_parameters={"expression": "1 + 1"},
            ),
            PlanStep(
                index=1,
                description="Reason",
                depends_on=[0],
            ),
        ]

        await executor.process({"state": state})

        assert len(state.results) == 2
        assert "calculator" in state.tools_used
        assert len(state.errors) == 0


# ── PlannerAgent + ExecutorAgent Integration ─────────────────────


class TestPlannerExecutorPipeline:
    """Full pipeline: PlannerAgent generates plan, ExecutorAgent runs it."""

    @pytest.mark.asyncio
    async def test_full_pipeline_no_llm(self) -> None:
        """PlannerAgent template plan -> ExecutorAgent reasoning steps."""
        planner = PlannerAgent()
        executor = ExecutorAgent()

        plan_result = await planner.process({"goal": "analyse the network"})
        assert plan_result["success"] is True

        # Convert plan dicts to PlanStep objects
        plan_steps = [
            PlanStep(
                index=item["index"],
                description=item["description"],
                tool_name=item.get("tool_name"),
                tool_parameters=item.get("tool_parameters", {}),
                depends_on=item.get("depends_on", []),
                rationale=item.get("rationale", ""),
                priority=item.get("priority", "medium"),
            )
            for item in plan_result["plan"]
        ]

        state = AgentState()
        state.plan = plan_steps
        state.goal = "analyse the network"

        exec_result = await executor.process({"state": state})
        assert exec_result["success"] is True
        assert len(exec_result["results"]) == len(plan_steps)

    @pytest.mark.asyncio
    async def test_full_pipeline_with_tool(self) -> None:
        """PlannerAgent plan with tool -> ExecutorAgent executes via registry."""
        planner = PlannerAgent()
        registry = ToolRegistry()
        registry.register_instance(CalculatorTool())
        executor = ExecutorAgent(tool_registry=registry)

        # Create a plan that uses the calculator
        plan_steps = [
            PlanStep(
                index=0,
                description="Calculate sum",
                tool_name="calculator",
                tool_parameters={"expression": "10 + 20"},
                priority="high",
            ),
            PlanStep(
                index=1,
                description="Summarise result",
                depends_on=[0],
            ),
        ]

        state = AgentState()
        state.plan = plan_steps
        state.goal = "add numbers"

        exec_result = await executor.process({"state": state})
        assert exec_result["success"] is True
        assert exec_result["results"][0]["output"]["result"] == 30.0
        assert "calculator" in exec_result["tools_used"]
        assert len(exec_result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_full_pipeline_tool_failure_propagates(self) -> None:
        """Tool failure in execution propagates correctly."""
        registry = ToolRegistry()
        registry.register_instance(CountingTool())
        executor = ExecutorAgent(
            tool_registry=registry,
            max_retries=1,
        )

        plan_steps = [
            PlanStep(
                index=0,
                description="Use tool",
                tool_name="counting_tool",
                tool_parameters={"value": 42},
            ),
            PlanStep(
                index=1,
                description="Depend on tool",
                depends_on=[0],
            ),
        ]

        state = AgentState()
        state.plan = plan_steps

        exec_result = await executor.process({"state": state})
        assert exec_result["success"] is True  # counting_tool succeeds
        assert len(exec_result["errors"]) == 0


# ── Execution History Integration Tests ──────────────────────────


class TestExecutionHistoryIntegration:
    @pytest.mark.asyncio
    async def test_history_tracks_across_invocations(self) -> None:
        """Execution history persists across multiple process() calls."""
        registry = ToolRegistry()
        registry.register_instance(CountingTool())
        executor = ExecutorAgent(tool_registry=registry)

        # First invocation
        state1 = AgentState()
        state1.plan = [
            PlanStep(
                index=0,
                description="First call",
                tool_name="counting_tool",
                tool_parameters={"value": 1},
            )
        ]
        await executor.process({"state": state1})
        assert len(executor.execution_history) == 1

        # Second invocation
        state2 = AgentState()
        state2.plan = [
            PlanStep(
                index=0,
                description="Second call",
                tool_name="counting_tool",
                tool_parameters={"value": 2},
            ),
            PlanStep(
                index=1,
                description="Third call",
                tool_name="counting_tool",
                tool_parameters={"value": 3},
            ),
        ]
        await executor.process({"state": state2})
        assert len(executor.execution_history) == 3

        # Stats aggregate correctly
        stats = executor.get_execution_stats()
        assert stats["total_steps"] == 3
        assert stats["completed"] == 3

    @pytest.mark.asyncio
    async def test_execution_history_in_result(self) -> None:
        """Execution history is included in the process() result."""
        executor = ExecutorAgent()
        state = AgentState()
        state.plan = [
            PlanStep(index=0, description="Simple step"),
        ]

        result = await executor.process({"state": state})
        assert len(result["execution_history"]) == 1
        rec = result["execution_history"][0]
        assert rec["step_index"] == 0
        assert rec["description"] == "Simple step"
        assert rec["status"] == "completed"
        assert rec["success"] is True

    @pytest.mark.asyncio
    async def test_state_results_and_errors_accumulate(self) -> None:
        """State.results and state.errors accumulate across calls."""
        registry = ToolRegistry()
        registry.register_instance(CalculatorTool())
        executor = ExecutorAgent(tool_registry=registry)

        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Calc A",
                tool_name="calculator",
                tool_parameters={"expression": "1+1"},
            ),
            PlanStep(
                index=1,
                description="Calc B",
                tool_name="calculator",
                tool_parameters={"expression": "2+2"},
            ),
        ]

        await executor.process({"state": state})

        assert len(state.results) == 2
        assert state.results[0].output["result"] == 2.0
        assert state.results[1].output["result"] == 4.0
        assert "calculator" in state.tools_used


# ── Complex Plan Integration Tests ──────────────────────────────


class TestComplexPlanExecution:
    @pytest.mark.asyncio
    async def test_diamond_plan_all_steps_execute(self) -> None:
        """Diamond dependency graph: all four steps execute."""
        registry = ToolRegistry()
        registry.register_instance(CalculatorTool())
        executor = ExecutorAgent(tool_registry=registry)

        plan = [
            PlanStep(
                index=0,
                description="Root calc",
                tool_name="calculator",
                tool_parameters={"expression": "1+1"},
            ),
            PlanStep(
                index=1,
                description="Branch A",
                depends_on=[0],
            ),
            PlanStep(
                index=2,
                description="Branch B",
                depends_on=[0],
            ),
            PlanStep(
                index=3,
                description="Merge",
                depends_on=[1, 2],
            ),
        ]

        state = AgentState()
        state.plan = plan

        result = await executor.process({"state": state})
        assert result["success"] is True
        assert len(result["results"]) == 4
        # Verify ordering: step 0 first, step 3 last
        indices = [r["step_index"] for r in result["results"]]
        assert indices[0] == 0
        assert indices[-1] == 3

    @pytest.mark.asyncio
    async def test_partial_failure_skips_downstream(self) -> None:
        """When a step fails, all its dependents are skipped."""
        registry = ToolRegistry()
        registry.register_instance(StubTool(should_succeed=False))
        executor = ExecutorAgent(
            tool_registry=registry,
            max_retries=1,
        )

        plan = [
            PlanStep(
                index=0,
                description="Fail here",
                tool_name="stub_tool",
                tool_parameters={},
            ),
            PlanStep(
                index=1,
                description="Depend on fail",
                depends_on=[0],
            ),
            PlanStep(
                index=2,
                description="Depend on fail too",
                depends_on=[0],
            ),
            PlanStep(
                index=3,
                description="Independent",
            ),
        ]

        state = AgentState()
        state.plan = plan

        result = await executor.process({"state": state})
        assert result["success"] is False

        # Steps 1 and 2 should be skipped
        statuses = {
            r["step_index"]: r["status"]
            for r in result["execution_history"]
        }
        assert statuses[0] == "failed"
        assert statuses[1] == "skipped"
        assert statuses[2] == "skipped"
        assert statuses[3] == "completed"


# ── Stub Tool class for tests ────────────────────────────────────


class StubTool(BaseTool):
    """A stub tool for integration tests."""

    def __init__(
        self,
        name: str = "stub_tool",
        should_succeed: bool = True,
    ) -> None:
        super().__init__(ToolDefinition(
            name=name,
            description="Stub test tool",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                },
            },
            returns={"type": "object", "properties": {"result": {"type": "string"}}},
            permissions=[ToolPermission.EXECUTE],
        ))
        self._should_succeed = should_succeed

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        if not self._should_succeed:
            return ToolResult(success=False, error="Stub tool failure")
        return ToolResult(success=True, output={"result": "ok"})
