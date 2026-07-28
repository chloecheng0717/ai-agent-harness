"""
Integration Tests for CTF Tool Library (Phase 1)

Tests verify the full pipeline: tool -> sandbox -> ToolRegistry -> ExecutorAgent.
Docker interactions are mocked (same pattern as existing sandbox integration tests)
to avoid requiring a live daemon.

Tests cover:
- FileTool integration with ToolRegistry and ExecutorAgent
- StringsTool integration with ToolRegistry and ExecutorAgent
- PythonTool integration with ToolRegistry and ExecutorAgent
- EncodingTool integration with ToolRegistry and ExecutorAgent
- End-to-end flag detection in reasoning loop with tool execution
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.specialized.executor import ExecutorAgent
from app.models.state import AgentState, PlanStep
from app.models.tool import (
    RiskLevel,
    ToolDefinition,
    ToolExecutionRequest,
    ToolPermission,
    ToolResult,
)
from app.sandbox.docker_sandbox import (
    CleanupPolicy,
    DockerSandbox,
    SandboxConfig,
    SandboxExecutionResult,
)
from app.tools.plugins.file_tool import FileTool
from app.tools.plugins.strings_tool import StringsTool
from app.tools.plugins.python_tool import PythonTool
from app.tools.plugins.encoding_tool import EncodingTool
from app.tools.registry import ToolRegistry


# ── Helpers ──────────────────────────────────────────────────────


def _mock_container(
    exit_code: int = 0,
    stdout_bytes: bytes = b"",
    stderr_bytes: bytes = b"",
) -> MagicMock:
    container = MagicMock()
    container.short_id = "ctf00001"
    container.wait.return_value = {"StatusCode": exit_code, "Error": ""}

    def logs_side_effect(stdout=True, stderr=False):
        if stdout:
            return stdout_bytes
        if stderr:
            return stderr_bytes
        return b""

    container.logs.side_effect = logs_side_effect
    return container


def _mock_docker_client(
    exit_code: int = 0,
    stdout_bytes: bytes = b"",
    stderr_bytes: bytes = b"",
) -> MagicMock:
    client = MagicMock()
    client.ping.return_value = True
    client.images.get.return_value = MagicMock()
    client.images.pull.return_value = MagicMock()
    container = _mock_container(exit_code, stdout_bytes, stderr_bytes)
    client.containers.create.return_value = container
    return client


def _sandbox_with_mock(
    exit_code: int = 0,
    stdout_bytes: bytes = b"",
    stderr_bytes: bytes = b"",
) -> DockerSandbox:
    sandbox = DockerSandbox(SandboxConfig(pull_policy="never"))
    client = _mock_docker_client(exit_code, stdout_bytes, stderr_bytes)
    sandbox._client = client
    return sandbox


# ═══════════════════════════════════════════════════════════════════
# FileTool Integration Tests
# ═══════════════════════════════════════════════════════════════════


class TestFileToolIntegration:
    """Integration tests for FileTool with real sandbox flow."""

    @pytest.mark.asyncio
    async def test_file_tool_registered_and_executed(self) -> None:
        sandbox = _sandbox_with_mock(
            exit_code=0, stdout_bytes=b"application/json\n"
        )
        tool = FileTool(sandbox)
        registry = ToolRegistry()
        registry.register_instance(tool)

        request = ToolExecutionRequest(
            tool_name="file",
            parameters={"path": "/workspace/data.json"},
            agent_permissions=[ToolPermission.READ, ToolPermission.EXECUTE],
        )
        result = await registry.execute(request)

        assert result.success is True
        assert result.output["mime_type"] == "application/json"
        assert result.output["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_file_tool_via_executor_agent(self) -> None:
        sandbox = _sandbox_with_mock(
            exit_code=0, stdout_bytes=b"text/plain\n"
        )
        tool = FileTool(sandbox)
        registry = ToolRegistry()
        registry.register_instance(tool)
        executor = ExecutorAgent(tool_registry=registry)

        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Identify file type",
                tool_name="file",
                tool_parameters={"path": "/workspace/secret.bin"},
                priority="high",
            ),
        ]

        result = await executor.process({"state": state})

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["tool_name"] == "file"
        assert "file" in result["tools_used"]


# ═══════════════════════════════════════════════════════════════════
# StringsTool Integration Tests
# ═══════════════════════════════════════════════════════════════════


class TestStringsToolIntegration:
    """Integration tests for StringsTool with real sandbox flow."""

    @pytest.mark.asyncio
    async def test_strings_tool_registered_and_executed(self) -> None:
        stdout = b"flag_part1\nanother_string\nCTF{partial}\n"
        sandbox = _sandbox_with_mock(exit_code=0, stdout_bytes=stdout)
        tool = StringsTool(sandbox)
        registry = ToolRegistry()
        registry.register_instance(tool)

        request = ToolExecutionRequest(
            tool_name="strings",
            parameters={"path": "/workspace/binary", "min_length": 4},
            agent_permissions=[ToolPermission.READ, ToolPermission.EXECUTE],
        )
        result = await registry.execute(request)

        assert result.success is True
        assert result.output["count"] == 3
        assert "CTF{partial}" in result.output["strings"]

    @pytest.mark.asyncio
    async def test_strings_tool_via_executor_agent(self) -> None:
        stdout = b"hello\nworld\n"
        sandbox = _sandbox_with_mock(exit_code=0, stdout_bytes=stdout)
        tool = StringsTool(sandbox)
        registry = ToolRegistry()
        registry.register_instance(tool)
        executor = ExecutorAgent(tool_registry=registry)

        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Extract strings",
                tool_name="strings",
                tool_parameters={"path": "/workspace/mystery"},
                priority="high",
            ),
        ]

        result = await executor.process({"state": state})

        assert result["success"] is True
        assert "strings" in result["tools_used"]


# ═══════════════════════════════════════════════════════════════════
# PythonTool Integration Tests
# ═══════════════════════════════════════════════════════════════════


class TestPythonToolIntegration:
    """Integration tests for PythonTool with real sandbox flow."""

    @pytest.mark.asyncio
    async def test_python_tool_registered_and_executed(self) -> None:
        sandbox = _sandbox_with_mock(
            exit_code=0, stdout_bytes=b"42\n"
        )
        tool = PythonTool(sandbox)
        registry = ToolRegistry()
        registry.register_instance(tool)

        request = ToolExecutionRequest(
            tool_name="python",
            parameters={"code": "print(6 * 7)"},
            agent_permissions=[ToolPermission.EXECUTE],
        )
        result = await registry.execute(request)

        assert result.success is True
        assert result.output["stdout"] == "42\n"
        assert result.output["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_python_tool_via_executor_agent(self) -> None:
        sandbox = _sandbox_with_mock(
            exit_code=0, stdout_bytes=b"decoded_flag\n"
        )
        tool = PythonTool(sandbox)
        registry = ToolRegistry()
        registry.register_instance(tool)
        executor = ExecutorAgent(tool_registry=registry)

        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Decode data with Python",
                tool_name="python",
                tool_parameters={
                    "code": "import base64; print(base64.b64decode(b'ZGVjb2RlZF9mbGFn').decode())"
                },
                priority="high",
            ),
        ]

        result = await executor.process({"state": state})

        assert result["success"] is True
        assert "python" in result["tools_used"]

    @pytest.mark.asyncio
    async def test_python_tool_error_via_executor(self) -> None:
        sandbox = _sandbox_with_mock(
            exit_code=1,
            stdout_bytes=b"",
            stderr_bytes=b"SyntaxError: invalid syntax\n",
        )
        tool = PythonTool(sandbox)
        registry = ToolRegistry()
        registry.register_instance(tool)
        executor = ExecutorAgent(tool_registry=registry)

        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Run bad code",
                tool_name="python",
                tool_parameters={"code": "def"},
                priority="high",
            ),
        ]

        result = await executor.process({"state": state})

        assert result["success"] is False
        assert len(result["errors"]) > 0


# ═══════════════════════════════════════════════════════════════════
# EncodingTool Integration Tests
# ═══════════════════════════════════════════════════════════════════


class TestEncodingToolIntegration:
    """Integration tests for EncodingTool with real sandbox flow."""

    @pytest.mark.asyncio
    async def test_encoding_tool_registered_and_executed(self) -> None:
        sandbox = _sandbox_with_mock(
            exit_code=0, stdout_bytes=b"flag{encoded}\n"
        )
        tool = EncodingTool(sandbox)
        registry = ToolRegistry()
        registry.register_instance(tool)

        request = ToolExecutionRequest(
            tool_name="encoding",
            parameters={
                "input": "ZmxhZ3tlbmNvZGVkfQ==",
                "encoding": "base64",
                "mode": "decode",
            },
            agent_permissions=[ToolPermission.EXECUTE],
        )
        result = await registry.execute(request)

        assert result.success is True
        assert result.output["mode_used"] == "decode"

    @pytest.mark.asyncio
    async def test_encoding_tool_via_executor_agent(self) -> None:
        sandbox = _sandbox_with_mock(
            exit_code=0, stdout_bytes=b"68656c6c6f"
        )
        tool = EncodingTool(sandbox)
        registry = ToolRegistry()
        registry.register_instance(tool)
        executor = ExecutorAgent(tool_registry=registry)

        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Hex encode string",
                tool_name="encoding",
                tool_parameters={
                    "input": "hello",
                    "encoding": "hex",
                    "mode": "encode",
                },
                priority="high",
            ),
        ]

        result = await executor.process({"state": state})

        assert result["success"] is True
        assert "encoding" in result["tools_used"]


# ═══════════════════════════════════════════════════════════════════
# End-to-End Flag Detection Integration
# ═══════════════════════════════════════════════════════════════════


class TestFlagDetectionIntegration:
    """Tests that flag detection works end-to-end with tools."""

    @pytest.mark.asyncio
    async def test_flag_detected_stops_loop(self) -> None:
        """Reasoning loop stops when tools output a flag."""
        from app.agents.core.reasoning_loop import ReasoningLoop

        loop = ReasoningLoop(max_iterations=10)

        step = 0

        async def reasoning_fn(state):
            nonlocal step
            step += 1
            return "Analyzing the data..."

        async def action_fn(action, action_input):
            # Simulate a tool finding a flag
            return {"stdout": "The secret is flag{you_found_it}\n", "exit_code": 0}

        result = await loop.run({}, reasoning_fn, action_fn)

        assert result["found_flag"] == "flag{you_found_it}"
        assert result["iterations"] < 10

    @pytest.mark.asyncio
    async def test_no_flag_loop_runs_full(self) -> None:
        """Reasoning loop runs full iterations when no flag is present."""
        from app.agents.core.reasoning_loop import ReasoningLoop

        loop = ReasoningLoop(max_iterations=3)

        async def reasoning_fn(state):
            return "Thinking..."

        async def action_fn(action, action_input):
            return {"stdout": "normal output", "exit_code": 0}

        result = await loop.run({}, reasoning_fn, action_fn)

        assert result["found_flag"] is None
        assert result["iterations"] == 3
