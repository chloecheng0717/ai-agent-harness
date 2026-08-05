"""
Integration Tests for Planner Tool Selection & Reasoning Loop

Covers the six agreed scenarios:
1. Planner selects ArchiveTool for archive extraction.
2. Planner selects EncodingTool for encoded data.
3. Planner selects StringsTool for binary/string extraction.
4. Recovery after a failed tool execution (no repeated failed actions,
   alternative tools allowed).
5. Avoids unnecessary repeated tool calls (deduplication).
6. Multi-step execution that reuses previous observations.

Tools run through the real ToolRegistry with a mocked Docker sandbox, the
same pattern used by the existing CTF integration tests.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from app.agents.core.orchestrator import Orchestrator
from app.agents.core.reasoning_loop import ReasoningLoop
from app.agents.specialized.executor import ExecutorAgent
from app.agents.specialized.planner import PlannerAgent
from app.models.state import AgentState, PlanStep
from app.models.tool import (
    ToolDefinition,
    ToolPermission,
    ToolResult,
)
from app.sandbox.docker_sandbox import SandboxExecutionResult
from app.tools.base import BaseTool
from app.tools.plugins.archive_tool import ArchiveTool
from app.tools.plugins.encoding_tool import EncodingTool
from app.tools.plugins.file_tool import FileTool
from app.tools.plugins.strings_tool import StringsTool
from app.tools.registry import ToolRegistry

# ── Helpers ──────────────────────────────────────────────────────


def _ok_result(stdout: str = "") -> SandboxExecutionResult:
    return SandboxExecutionResult(
        success=True,
        exit_code=0,
        stdout=stdout,
        stderr="",
        execution_time=5.0,
        container_id="mock1234",
        image="python:3.12-slim",
        timed_out=False,
    )


def _mock_sandbox(stdout: str = "") -> MagicMock:
    sandbox = MagicMock()
    sandbox.execute = AsyncMock(return_value=_ok_result(stdout))
    return sandbox


_ARCHIVE_STDOUT = json.dumps(
    {
        "success": True,
        "files": ["flag.txt"],
        "destination": "/workspace/extracted",
        "archive_type": "zip",
        "file_count": 1,
    }
)


def _build_registry() -> ToolRegistry:
    """Register the real CTF tools backed by a mocked sandbox."""
    registry = ToolRegistry()
    registry.register_instance(ArchiveTool(_mock_sandbox(_ARCHIVE_STDOUT)))
    registry.register_instance(EncodingTool(_mock_sandbox("decoded_flag\n")))
    registry.register_instance(StringsTool(_mock_sandbox("flag{from_strings}\n")))
    registry.register_instance(FileTool(_mock_sandbox("application/zip\n")))
    return registry


class CountingTool(BaseTool):
    """Tool that counts how many times it was invoked."""

    def __init__(self) -> None:
        super().__init__(
            ToolDefinition(
                name="counting_tool",
                description="Counts invocations",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                },
                returns={"type": "object", "properties": {"count": {"type": "integer"}}},
                permissions=[ToolPermission.EXECUTE],
            )
        )
        self.call_count = 0

    async def execute(
        self,
        parameters: dict[str, Any],
        workspace: Any = None,
    ) -> ToolResult:
        self.call_count += 1
        return ToolResult(
            success=True,
            output={"count": self.call_count, "value": parameters.get("value", 0)},
        )


# ═══════════════════════════════════════════════════════════════════
# 1. Dedicated tool selection
# ═══════════════════════════════════════════════════════════════════


class TestPlannerToolSelection:
    @pytest.mark.asyncio
    async def test_planner_selects_archive_tool_for_extraction(self) -> None:
        planner = PlannerAgent()
        result = await planner.process({
            "goal": "Extract the archive.zip and find the flag",
            "tools_available": ["archive", "strings", "file", "encoding", "python"],
        })

        assert result["success"] is True
        tools = [step.get("tool_name") for step in result["plan"]]
        assert tools[0] == "archive"
        assert "archive" in tools

    @pytest.mark.asyncio
    async def test_planner_selects_encoding_tool_for_encoded_data(self) -> None:
        planner = PlannerAgent()
        result = await planner.process({
            "goal": "Decode the base64 data SGVJUEk=",
            "tools_available": ["encoding", "python", "archive"],
        })

        assert result["success"] is True
        step = result["plan"][0]
        assert step["tool_name"] == "encoding"
        assert step["tool_parameters"]["encoding"] == "base64"
        assert step["tool_parameters"]["input"] == "SGVJUEk="

    @pytest.mark.asyncio
    async def test_planner_selects_strings_tool_for_binary(self) -> None:
        planner = PlannerAgent()
        result = await planner.process({
            "goal": "Extract printable strings from the binary file",
            "tools_available": ["archive", "strings", "file"],
        })

        assert result["success"] is True
        tools = [step.get("tool_name") for step in result["plan"]]
        assert tools == ["strings"]

    @pytest.mark.asyncio
    async def test_planner_keeps_unavailable_tool_out_of_plan(self) -> None:
        """A hallucinated/unavailable tool is replaced by a real match."""
        planner = PlannerAgent()
        result = await planner.process({
            "goal": "Decode the base64 data SGVJUEk=",
            "tools_available": ["encoding"],
        })

        assert result["success"] is True
        step = result["plan"][0]
        assert step["tool_name"] == "encoding"

    def test_refine_plan_dedups_identical_tool_calls(self) -> None:
        agent = PlannerAgent()
        steps = [
            PlanStep(
                index=0,
                description="Extract the archive",
                tool_name="archive",
                tool_parameters={"archive_path": "data.zip"},
            ),
            PlanStep(
                index=1,
                description="Extract the archive again",
                tool_name="archive",
                tool_parameters={"archive_path": "data.zip"},
            ),
        ]
        refined = agent._refine_plan(steps, ["archive"])

        assert refined[0].tool_name == "archive"
        assert refined[1].tool_name is None
        assert 0 in refined[1].depends_on


# ═══════════════════════════════════════════════════════════════════
# Orchestrator integration
# ═══════════════════════════════════════════════════════════════════


class TestOrchestratorToolSelection:
    @pytest.mark.asyncio
    async def test_create_plan_and_select_tools_assign_archive(self) -> None:
        registry = _build_registry()
        orch = Orchestrator(planner=PlannerAgent(), tool_registry=registry)

        state = AgentState()
        state.goal = "Extract archive.zip"
        state.tools_available = ["archive", "strings", "file", "encoding"]

        state = await orch.create_plan(state)
        assert any(step.tool_name == "archive" for step in state.plan)

        state = await orch.select_tools(state)
        assert "archive" in state.metadata["selected_tools"]

    @pytest.mark.asyncio
    async def test_select_tools_swaps_unavailable_tool(self) -> None:
        registry = _build_registry()
        orch = Orchestrator(tool_registry=registry)

        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Decode the base64 data",
                tool_name="nmap",
                tool_parameters={"target": "x"},
            )
        ]

        state = await orch.select_tools(state)

        assert state.plan[0].tool_name == "encoding"
        assert state.metadata["selected_tools"] == ["encoding"]

    @pytest.mark.asyncio
    async def test_execute_plan_runs_archive_tool(self) -> None:
        registry = _build_registry()
        orch = Orchestrator(tool_registry=registry)

        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Extract the archive",
                tool_name="archive",
                tool_parameters={"archive_path": "/workspace/data.zip"},
                priority="high",
            )
        ]

        state = await orch.execute_plan(state)

        assert state.results[0]["success"] is True
        assert state.results[0]["output"]["files"] == ["flag.txt"]
        assert "archive" in state.tools_used


# ═══════════════════════════════════════════════════════════════════
# 4. Recovery after a failed tool execution
# ═══════════════════════════════════════════════════════════════════


class TestFailureRecovery:
    @pytest.mark.asyncio
    async def test_revision_does_not_repeat_failed_tool(self) -> None:
        planner = PlannerAgent()
        previous_plan = [
            {
                "index": 0,
                "description": "Extract archive",
                "tool_name": "archive",
                "tool_parameters": {"archive_path": "data.zip"},
                "depends_on": [],
            },
            {"index": 1, "description": "Summarise", "depends_on": [0]},
        ]
        errors = ["Step 0: Tool 'archive' failed: corrupt archive"]

        steps = await planner.revise_plan(
            previous_plan,
            errors,
            tools_available=["archive", "python", "file"],
        )

        tools = [step.tool_name for step in steps]
        assert "archive" not in tools
        assert "python" in tools  # alternative approach, not a repeat

    @pytest.mark.asyncio
    async def test_revision_keeps_successful_steps(self) -> None:
        planner = PlannerAgent()
        previous_plan = [
            {
                "index": 0,
                "description": "Extract archive",
                "tool_name": "archive",
                "tool_parameters": {"archive_path": "data.zip"},
                "depends_on": [],
            },
            {
                "index": 1,
                "description": "Decode the base64 data",
                "tool_name": "encoding",
                "tool_parameters": {"input": "aGk=", "encoding": "base64"},
                "depends_on": [0],
            },
        ]
        errors = ["Step 1: Tool 'encoding' failed: bad input"]

        steps = await planner.revise_plan(
            previous_plan,
            errors,
            tools_available=["archive", "encoding", "python"],
        )

        descriptions = [step.description for step in steps]
        assert any("archive" in d.lower() for d in descriptions)


# ═══════════════════════════════════════════════════════════════════
# 5. Avoiding unnecessary repeated tool calls
# ═══════════════════════════════════════════════════════════════════


class TestDeduplication:
    @pytest.mark.asyncio
    async def test_executor_avoids_duplicate_tool_calls(self) -> None:
        registry = ToolRegistry()
        counting = CountingTool()
        registry.register_instance(counting)
        executor = ExecutorAgent(tool_registry=registry)

        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="First scan",
                tool_name="counting_tool",
                tool_parameters={"value": 7},
            ),
            PlanStep(
                index=1,
                description="Repeat scan",
                tool_name="counting_tool",
                tool_parameters={"value": 7},
            ),
        ]

        result = await executor.process({"state": state})

        assert counting.call_count == 1
        assert result["results"][1]["output"]["count"] == 1
        assert result["results"][1]["metadata"]["reused_from"] == 0

    @pytest.mark.asyncio
    async def test_orchestrator_execute_plan_avoids_duplicates(self) -> None:
        registry = ToolRegistry()
        counting = CountingTool()
        registry.register_instance(counting)
        orch = Orchestrator(tool_registry=registry)

        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Compute",
                tool_name="counting_tool",
                tool_parameters={"value": 3},
            ),
            PlanStep(
                index=1,
                description="Compute again",
                tool_name="counting_tool",
                tool_parameters={"value": 3},
            ),
        ]

        state = await orch.execute_plan(state)

        assert counting.call_count == 1
        assert state.results[1]["reused_output_of"] == 0

    @pytest.mark.asyncio
    async def test_orchestrator_cache_persists_across_iterations(self) -> None:
        """The tool-result cache must survive plan-execute cycle boundaries:
        a repeated identical call in a later iteration reuses the cached
        output instead of invoking the tool again."""
        registry = ToolRegistry()
        counting = CountingTool()
        registry.register_instance(counting)
        orch = Orchestrator(tool_registry=registry)

        plan = [
            PlanStep(
                index=0,
                description="Compute",
                tool_name="counting_tool",
                tool_parameters={"value": 3},
            )
        ]

        first = AgentState()
        first.plan = list(plan)
        first = await orch.execute_plan(first)
        assert counting.call_count == 1

        second = AgentState()
        second.plan = list(plan)
        second = await orch.execute_plan(second)

        assert counting.call_count == 1
        assert second.results[0]["reused_output_of"] == 0


# ═══════════════════════════════════════════════════════════════════
# 6. Multi-step execution using previous observations
# ═══════════════════════════════════════════════════════════════════


class TestMultiStepExecution:
    @pytest.mark.asyncio
    async def test_multi_step_plan_runs_in_dependency_order(self) -> None:
        registry = _build_registry()
        executor = ExecutorAgent(tool_registry=registry)

        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Extract the archive",
                tool_name="archive",
                tool_parameters={"archive_path": "data.zip"},
                priority="high",
            ),
            PlanStep(
                index=1,
                description="Extract strings from extracted files",
                tool_name="strings",
                tool_parameters={"content": "raw binary bytes"},
                depends_on=[0],
                priority="high",
            ),
            PlanStep(
                index=2,
                description="Summarise the findings",
                depends_on=[1],
            ),
        ]

        result = await executor.process({"state": state})

        assert result["success"] is True
        indices = [r["step_index"] for r in result["results"]]
        assert indices == [0, 1, 2]
        assert result["results"][1]["tool_name"] == "strings"
        assert result["results"][1]["output"]["strings"] == ["flag{from_strings}"]
        # Step 0's observation is available to later steps via state.results.
        assert state.results[0].output["files"] == ["flag.txt"]


# ═══════════════════════════════════════════════════════════════════
# Plan-driven ReasoningLoop
# ═══════════════════════════════════════════════════════════════════


class TestReasoningLoopPlanDriven:
    @pytest.mark.asyncio
    async def test_executes_plan_in_order_without_llm_reasoning(self) -> None:
        loop = ReasoningLoop(max_iterations=10)
        plan = [
            {
                "index": 0, "description": "extract", "tool_name": "archive",
                "tool_parameters": {"archive_path": "data.zip"},
                "depends_on": [], "executed": False,
            },
            {
                "index": 1, "description": "strings", "tool_name": "strings",
                "tool_parameters": {"content": "bytes"},
                "depends_on": [0], "executed": False,
            },
            {
                "index": 2, "description": "summarise", "tool_name": None,
                "tool_parameters": {},
                "depends_on": [1], "executed": False,
            },
        ]
        calls: list[str] = []

        async def reasoning_fn(state: dict[str, Any]) -> str:
            calls.append("reason")
            return "thinking"

        async def action_fn(action: str, action_input: dict[str, Any]) -> dict[str, Any]:
            calls.append(action)
            return {"success": True, "output": f"{action} done"}

        result = await loop.run({"plan": plan}, reasoning_fn, action_fn)

        assert calls == ["archive", "strings", "think"]
        assert "reason" not in calls  # no unnecessary LLM reasoning calls
        assert result["iterations"] == 3

    @pytest.mark.asyncio
    async def test_does_not_repeat_failed_action(self) -> None:
        loop = ReasoningLoop(max_iterations=10)
        plan = [
            {
                "index": 0, "description": "fragile", "tool_name": "flaky",
                "tool_parameters": {}, "depends_on": [], "executed": False,
            },
            {
                "index": 1, "description": "independent", "tool_name": "steady",
                "tool_parameters": {}, "depends_on": [], "executed": False,
            },
        ]
        calls: list[str] = []

        async def reasoning_fn(state: dict[str, Any]) -> str:
            return "thinking"

        async def action_fn(action: str, action_input: dict[str, Any]) -> dict[str, Any]:
            calls.append(action)
            if action == "flaky":
                return {"success": False, "error": "boom"}
            return {"success": True, "output": "ok"}

        result = await loop.run({"plan": plan}, reasoning_fn, action_fn)

        assert calls.count("flaky") == 1  # failed action never repeated
        assert calls == ["flaky", "steady"]
        assert result["iterations"] == 2

    @pytest.mark.asyncio
    async def test_trusts_real_tool_output_over_guess(self) -> None:
        """The loop uses registry output, ignoring hallucinated observations."""
        registry = ToolRegistry()
        registry.register_instance(
            EncodingTool(_mock_sandbox("flag{from_real_tool}\n"))
        )
        loop = ReasoningLoop(max_iterations=5)
        plan = [
            {
                "index": 0, "description": "decode", "tool_name": "encoding",
                "tool_parameters": {
                    "input": "ZmxhZ3tmcm9tX3JlYWxfdG9vbH0=",
                    "encoding": "base64",
                    "mode": "decode",
                },
                "depends_on": [], "executed": False,
            }
        ]

        async def reasoning_fn(state: dict[str, Any]) -> str:
            return "decode it"

        async def action_fn(action: str, action_input: dict[str, Any]) -> dict[str, Any]:
            # A model guess that would be wrong if the loop trusted it.
            return {"success": True, "output": "flag{not_real}"}

        result = await loop.run(
            {"plan": plan}, reasoning_fn, action_fn, tool_registry=registry
        )

        assert result["found_flag"] == "flag{from_real_tool}"
