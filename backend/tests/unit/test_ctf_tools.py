"""
Unit Tests for CTF Tool Library (Phase 1)

Tests cover (all with mocked DockerSandbox — no daemon required):
- FileTool
- StringsTool
- PythonTool
- EncodingTool
- FlagDetector and ReasoningLoop flag termination
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.sandbox.docker_sandbox import SandboxExecutionResult
from app.tools.plugins.file_tool import FileTool
from app.tools.plugins.strings_tool import StringsTool
from app.tools.plugins.python_tool import PythonTool, _MAX_TIMEOUT
from app.tools.plugins.encoding_tool import EncodingTool
from app.tools.registry import ToolRegistry
from app.models.tool import (
    RiskLevel,
    SandboxCapability,
    ToolExecutionRequest,
    ToolPermission,
)


# ── Helpers ──────────────────────────────────────────────────────


def _ok_result(
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    execution_time: float = 100.0,
) -> SandboxExecutionResult:
    return SandboxExecutionResult(
        success=exit_code == 0,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        execution_time=execution_time,
        container_id="mock1234",
        image="python:3.12-slim",
        timed_out=False,
    )


def _err_result(
    stderr: str = "error",
    exit_code: int = 1,
) -> SandboxExecutionResult:
    return SandboxExecutionResult(
        success=False,
        exit_code=exit_code,
        stdout="",
        stderr=stderr,
        execution_time=50.0,
        container_id="mock1234",
        image="python:3.12-slim",
        timed_out=False,
    )


def _timeout_result() -> SandboxExecutionResult:
    return SandboxExecutionResult(
        success=False,
        exit_code=-1,
        stdout="partial",
        stderr="Execution timed out after 5s",
        execution_time=5000.0,
        container_id="mock1234",
        image="python:3.12-slim",
        timed_out=True,
    )


def _mock_sandbox(return_value: SandboxExecutionResult | None = None) -> MagicMock:
    sandbox = MagicMock()
    sandbox.execute = AsyncMock(return_value=return_value or _ok_result())
    return sandbox


# ═══════════════════════════════════════════════════════════════════
# FileTool Tests
# ═══════════════════════════════════════════════════════════════════


class TestFileTool:
    """Unit tests for FileTool (mocked sandbox)."""

    def test_definition_metadata(self) -> None:
        sandbox = _mock_sandbox()
        tool = FileTool(sandbox)
        assert tool.name == "file"
        assert "ctf" in tool.definition.tags
        assert tool.sandbox_capability == SandboxCapability.CONTAINER
        assert tool.risk_level == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_identify_by_path(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(stdout="text/plain\n")
        )
        tool = FileTool(sandbox)
        result = await tool.execute({"path": "/workspace/secret.txt"})

        assert result.success is True
        assert result.output["mime_type"] == "text/plain"
        assert result.output["description"] == "Plain text file"
        assert result.output["exit_code"] == 0
        sandbox.execute.assert_awaited_once()
        cmd = sandbox.execute.call_args[0][0]
        assert "file --mime-type -b /workspace/secret.txt" in cmd

    @pytest.mark.asyncio
    async def test_identify_by_content(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(stdout="application/json\n")
        )
        tool = FileTool(sandbox)
        result = await tool.execute({"content": '{"key": "value"}'})

        assert result.success is True
        assert result.output["mime_type"] == "application/json"
        ws_kwargs = sandbox.execute.call_args[1]
        assert ws_kwargs["workspace_id"] == "filetool-temp"
        assert ws_kwargs["files"] == {"_ft_target": '{"key": "value"}'}

    @pytest.mark.asyncio
    async def test_no_input_returns_error(self) -> None:
        sandbox = _mock_sandbox()
        tool = FileTool(sandbox)
        result = await tool.execute({})
        assert result.success is False
        assert "must be provided" in result.error

    @pytest.mark.asyncio
    async def test_sandbox_failure_returns_error(self) -> None:
        sandbox = _mock_sandbox(_err_result(stderr="command not found"))
        tool = FileTool(sandbox)
        result = await tool.execute({"path": "/nonexistent"})
        assert result.success is False
        assert "command not found" in result.error

    @pytest.mark.asyncio
    async def test_sandbox_exception_returns_error(self) -> None:
        sandbox = MagicMock()
        sandbox.execute = AsyncMock(side_effect=RuntimeError("boom"))
        tool = FileTool(sandbox)
        result = await tool.execute({"path": "/test"})
        assert result.success is False
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_metadata_populated(self) -> None:
        sandbox = _mock_sandbox(_ok_result(stdout="application/zip\n"))
        tool = FileTool(sandbox)
        result = await tool.execute({"path": "/data.zip"})
        assert result.metadata["container_id"] == "mock1234"
        assert "timed_out" in result.metadata
        assert "command" in result.metadata

    @pytest.mark.asyncio
    async def test_unknown_mime_type(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(stdout="application/x-unknown-type\n")
        )
        tool = FileTool(sandbox)
        result = await tool.execute({"path": "/weird"})
        assert result.output["mime_type"] == "application/x-unknown-type"
        assert "application/x-unknown-type" in result.output["description"]

    def test_tool_registry_integration(self) -> None:
        sandbox = _mock_sandbox()
        tool = FileTool(sandbox)
        registry = ToolRegistry()
        registry.register_instance(tool)
        retrieved = registry.get_tool("file")
        assert retrieved is tool


# ═══════════════════════════════════════════════════════════════════
# StringsTool Tests
# ═══════════════════════════════════════════════════════════════════


class TestStringsTool:
    """Unit tests for StringsTool (mocked sandbox)."""

    def test_definition_metadata(self) -> None:
        sandbox = _mock_sandbox()
        tool = StringsTool(sandbox)
        assert tool.name == "strings"
        assert "ctf" in tool.definition.tags

    @pytest.mark.asyncio
    async def test_extract_by_path(self) -> None:
        output_text = "hello\nworld\nfoo\n"
        sandbox = _mock_sandbox(_ok_result(stdout=output_text))
        tool = StringsTool(sandbox)
        result = await tool.execute({
            "path": "/workspace/binary",
            "min_length": 4,
        })

        assert result.success is True
        assert result.output["strings"] == ["hello", "world", "foo"]
        assert result.output["count"] == 3
        assert result.output["truncated"] is False

    @pytest.mark.asyncio
    async def test_extract_by_content(self) -> None:
        output_text = "alpha\nbeta\n"
        sandbox = _mock_sandbox(_ok_result(stdout=output_text))
        tool = StringsTool(sandbox)
        result = await tool.execute({"content": "raw bytes here"})

        assert result.success is True
        ws_kwargs = sandbox.execute.call_args[1]
        assert ws_kwargs["workspace_id"] == "stringstool-temp"

    @pytest.mark.asyncio
    async def test_custom_min_length(self) -> None:
        sandbox = _mock_sandbox(_ok_result(stdout="a\nab\nabc\n"))
        tool = StringsTool(sandbox)
        result = await tool.execute({"path": "/bin", "min_length": 1})
        cmd = sandbox.execute.call_args[0][0]
        assert "-n 1" in cmd

    @pytest.mark.asyncio
    async def test_truncation_when_large_output(self) -> None:
        # Generate 6000 lines to exceed MAX_STRINGS (5000)
        big_output = "\n".join(f"line{i:06d}" for i in range(6000)) + "\n"
        sandbox = _mock_sandbox(_ok_result(stdout=big_output))
        tool = StringsTool(sandbox)
        result = await tool.execute({"path": "/big"})

        assert result.success is True
        assert result.output["truncated"] is True
        assert result.output["count"] == 5000

    @pytest.mark.asyncio
    async def test_empty_output(self) -> None:
        sandbox = _mock_sandbox(_ok_result(stdout=""))
        tool = StringsTool(sandbox)
        result = await tool.execute({"path": "/empty"})
        assert result.success is True
        assert result.output["strings"] == []
        assert result.output["count"] == 0

    @pytest.mark.asyncio
    async def test_no_input_returns_error(self) -> None:
        sandbox = _mock_sandbox()
        tool = StringsTool(sandbox)
        result = await tool.execute({})
        assert result.success is False
        assert "must be provided" in result.error

    @pytest.mark.asyncio
    async def test_min_length_in_metadata(self) -> None:
        sandbox = _mock_sandbox(_ok_result(stdout="test\n"))
        tool = StringsTool(sandbox)
        result = await tool.execute({"path": "/f", "min_length": 8})
        assert result.metadata["min_length"] == 8

    @pytest.mark.asyncio
    async def test_tool_registry_integration(self) -> None:
        sandbox = _mock_sandbox()
        tool = StringsTool(sandbox)
        registry = ToolRegistry()
        registry.register_instance(tool)
        assert registry.get_tool("strings") is tool


# ═══════════════════════════════════════════════════════════════════
# PythonTool Tests
# ═══════════════════════════════════════════════════════════════════


class TestPythonTool:
    """Unit tests for PythonTool (mocked sandbox)."""

    def test_definition_metadata(self) -> None:
        sandbox = _mock_sandbox()
        tool = PythonTool(sandbox)
        assert tool.name == "python"
        assert "ctf" in tool.definition.tags
        assert tool.risk_level == RiskLevel.HIGH
        assert tool.sandbox_capability == SandboxCapability.CONTAINER

    def test_definition_advertises_max_timeout(self) -> None:
        """The registry must not cap python calls below the tool's own max."""
        sandbox = _mock_sandbox()
        tool = PythonTool(sandbox)
        assert tool.definition.timeout == _MAX_TIMEOUT

    @pytest.mark.asyncio
    async def test_execute_code(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(stdout="42\n", exit_code=0)
        )
        tool = PythonTool(sandbox)
        result = await tool.execute({"code": "print(6 * 7)"})

        assert result.success is True
        assert result.output["stdout"] == "42\n"
        assert result.output["exit_code"] == 0
        assert result.output["timed_out"] is False
        cmd = sandbox.execute.call_args[0][0]
        assert "python -u /workspace/_pt_script.py" in cmd

    @pytest.mark.asyncio
    async def test_execute_script_path(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(stdout="done\n", exit_code=0)
        )
        tool = PythonTool(sandbox)
        result = await tool.execute({"script_path": "/workspace/solve.py"})

        assert result.success is True
        cmd = sandbox.execute.call_args[0][0]
        assert "python -u /workspace/solve.py" in cmd

    @pytest.mark.asyncio
    async def test_no_input_returns_error(self) -> None:
        sandbox = _mock_sandbox()
        tool = PythonTool(sandbox)
        result = await tool.execute({})
        assert result.success is False
        assert "must be provided" in result.error

    @pytest.mark.asyncio
    async def test_both_code_and_path_returns_error(self) -> None:
        sandbox = _mock_sandbox()
        tool = PythonTool(sandbox)
        result = await tool.execute({
            "code": "print(1)",
            "script_path": "/x.py",
        })
        assert result.success is False
        assert "not both" in result.error

    @pytest.mark.asyncio
    async def test_timeout_clamped(self) -> None:
        sandbox = _mock_sandbox(_ok_result(stdout="ok\n"))
        tool = PythonTool(sandbox)
        # Request 999 seconds — should be clamped to 120
        result = await tool.execute({"code": "print('ok')", "timeout": 999})
        assert result.success is True
        assert result.metadata["timeout_requested"] == 120

    @pytest.mark.asyncio
    async def test_timeout_minimum_one(self) -> None:
        sandbox = _mock_sandbox(_ok_result(stdout="ok\n"))
        tool = PythonTool(sandbox)
        result = await tool.execute({"code": "print('ok')", "timeout": 0})
        assert result.metadata["timeout_requested"] == 1

    @pytest.mark.asyncio
    async def test_timeout_result(self) -> None:
        sandbox = _mock_sandbox(_timeout_result())
        tool = PythonTool(sandbox)
        result = await tool.execute({"code": "while True: pass"})
        assert result.success is False
        assert result.output["timed_out"] is True
        assert result.metadata["timed_out"] is True

    @pytest.mark.asyncio
    async def test_stderr_captured(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(
                stdout="",
                stderr="Traceback: ZeroDivisionError",
                exit_code=1,
            )
        )
        tool = PythonTool(sandbox)
        result = await tool.execute({"code": "1/0"})
        assert result.success is False
        assert "ZeroDivisionError" in result.output["stderr"]

    @pytest.mark.asyncio
    async def test_sandbox_exception(self) -> None:
        sandbox = MagicMock()
        sandbox.execute = AsyncMock(side_effect=RuntimeError("docker down"))
        tool = PythonTool(sandbox)
        result = await tool.execute({"code": "print(1)"})
        assert result.success is False
        assert "docker down" in result.error

    @pytest.mark.asyncio
    async def test_tool_registry_integration(self) -> None:
        sandbox = _mock_sandbox()
        tool = PythonTool(sandbox)
        registry = ToolRegistry()
        registry.register_instance(tool)
        assert registry.get_tool("python") is tool


# ═══════════════════════════════════════════════════════════════════
# EncodingTool Tests
# ═══════════════════════════════════════════════════════════════════


class TestEncodingTool:
    """Unit tests for EncodingTool (mocked sandbox)."""

    def test_definition_metadata(self) -> None:
        sandbox = _mock_sandbox()
        tool = EncodingTool(sandbox)
        assert tool.name == "encoding"
        assert "ctf" in tool.definition.tags
        assert tool.risk_level == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_base64_decode(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(stdout="flag{test123}\n")
        )
        tool = EncodingTool(sandbox)
        result = await tool.execute({
            "input": "ZmxhZ3t0ZXN0MTIzfQ==",
            "encoding": "base64",
            "mode": "decode",
        })
        assert result.success is True
        assert result.output["output"] == "flag{test123}\n"
        assert result.output["mode_used"] == "decode"

    @pytest.mark.asyncio
    async def test_base64_encode(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(stdout="aGVsbG8=")
        )
        tool = EncodingTool(sandbox)
        result = await tool.execute({
            "input": "hello",
            "encoding": "base64",
            "mode": "encode",
        })
        assert result.success is True
        assert result.output["mode_used"] == "encode"

    @pytest.mark.asyncio
    async def test_hex_decode(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(stdout="flag{hex}\n")
        )
        tool = EncodingTool(sandbox)
        result = await tool.execute({
            "input": "666c61677b6865787d0a",
            "encoding": "hex",
            "mode": "decode",
        })
        assert result.success is True
        assert result.output["mode_used"] == "decode"

    @pytest.mark.asyncio
    async def test_hex_encode(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(stdout="666c6167")
        )
        tool = EncodingTool(sandbox)
        result = await tool.execute({
            "input": "flag",
            "encoding": "hex",
            "mode": "encode",
        })
        assert result.success is True
        assert result.output["mode_used"] == "encode"

    @pytest.mark.asyncio
    async def test_url_decode(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(stdout="hello world\n")
        )
        tool = EncodingTool(sandbox)
        result = await tool.execute({
            "input": "hello%20world%0A",
            "encoding": "url",
            "mode": "decode",
        })
        assert result.success is True
        assert result.output["mode_used"] == "decode"

    @pytest.mark.asyncio
    async def test_url_encode(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(stdout="hello%20world")
        )
        tool = EncodingTool(sandbox)
        result = await tool.execute({
            "input": "hello world",
            "encoding": "url",
            "mode": "encode",
        })
        assert result.success is True
        assert result.output["mode_used"] == "encode"

    @pytest.mark.asyncio
    async def test_auto_detect_decode_base64(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(stdout="decoded\n")
        )
        tool = EncodingTool(sandbox)
        # Looks like base64
        result = await tool.execute({
            "input": "ZGVjb2RlZA==",
            "encoding": "base64",
            "mode": "auto",
        })
        assert result.success is True
        assert result.output["mode_used"] == "decode"

    @pytest.mark.asyncio
    async def test_auto_detect_encode_plaintext(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(stdout="68656c6c6f")
        )
        tool = EncodingTool(sandbox)
        result = await tool.execute({
            "input": "hello",
            "encoding": "hex",
            "mode": "auto",
        })
        assert result.success is True
        assert result.output["mode_used"] == "encode"

    @pytest.mark.asyncio
    async def test_no_input_returns_error(self) -> None:
        sandbox = _mock_sandbox()
        tool = EncodingTool(sandbox)
        result = await tool.execute({"input": "", "encoding": "base64"})
        assert result.success is False
        assert "No input" in result.error

    @pytest.mark.asyncio
    async def test_invalid_encoding_returns_error(self) -> None:
        sandbox = _mock_sandbox()
        tool = EncodingTool(sandbox)
        result = await tool.execute({
            "input": "test",
            "encoding": "rot13",
        })
        assert result.success is False
        assert "Unsupported encoding" in result.error

    @pytest.mark.asyncio
    async def test_invalid_mode_returns_error(self) -> None:
        sandbox = _mock_sandbox()
        tool = EncodingTool(sandbox)
        result = await tool.execute({
            "input": "test",
            "encoding": "base64",
            "mode": "invalid",
        })
        assert result.success is False
        assert "Invalid mode" in result.error

    @pytest.mark.asyncio
    async def test_valid_utf8_and_printable_flags(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(stdout="hello world\n")
        )
        tool = EncodingTool(sandbox)
        result = await tool.execute({
            "input": "aGVsbG8gd29ybGQ=",
            "encoding": "base64",
            "mode": "decode",
        })
        assert result.output["valid_utf8"] is True
        assert result.output["printable"] is True

    @pytest.mark.asyncio
    async def test_tool_registry_integration(self) -> None:
        sandbox = _mock_sandbox()
        tool = EncodingTool(sandbox)
        registry = ToolRegistry()
        registry.register_instance(tool)
        assert registry.get_tool("encoding") is tool

    def test_detect_mode_heuristics(self) -> None:
        # Padded base64 → decode
        assert EncodingTool._detect_mode("Zm9v", "base64") == "encode"  # unpadded
        assert EncodingTool._detect_mode("Zm9v=", "base64") == "decode"
        assert EncodingTool._detect_mode("aGVsbG8=", "base64") == "decode"
        # Base64 with +/ → decode
        assert EncodingTool._detect_mode("a+b/c", "base64") == "decode"
        assert EncodingTool._detect_mode("hello", "base64") == "encode"
        # Hex
        assert EncodingTool._detect_mode("48656c6c6f", "hex") == "decode"
        assert EncodingTool._detect_mode("hello", "hex") == "encode"
        # URL
        assert EncodingTool._detect_mode("hello%20world", "url") == "decode"
        assert EncodingTool._detect_mode("hello world", "url") == "encode"
        # Empty
        assert EncodingTool._detect_mode("", "base64") == "encode"

    def test_is_printable(self) -> None:
        assert EncodingTool._is_printable("hello world\n") is True
        assert EncodingTool._is_printable("\x00\x01\x02") is False
        assert EncodingTool._is_printable("") is True

    def test_is_valid_utf8(self) -> None:
        assert EncodingTool._is_valid_utf8("hello") is True
        assert EncodingTool._is_valid_utf8("\u00e9\u00e8\u00ea") is True


# ═══════════════════════════════════════════════════════════════════
# FlagDetector Tests
# ═══════════════════════════════════════════════════════════════════


class TestFlagDetector:
    """Unit tests for FlagDetector."""

    def test_detects_flag_pattern(self) -> None:
        from app.agents.core.reasoning_loop import FlagDetector

        detector = FlagDetector()
        assert detector.scan("The answer is flag{abc123}") == "flag{abc123}"

    def test_detects_ctf_pattern(self) -> None:
        from app.agents.core.reasoning_loop import FlagDetector

        detector = FlagDetector()
        assert detector.scan("Found CTF{secret_value}") == "CTF{secret_value}"

    def test_case_insensitive(self) -> None:
        from app.agents.core.reasoning_loop import FlagDetector

        detector = FlagDetector()
        assert detector.scan("Flag{mixed_case}") == "Flag{mixed_case}"
        assert detector.scan("ctf{lowercase}") == "ctf{lowercase}"

    def test_no_match_returns_none(self) -> None:
        from app.agents.core.reasoning_loop import FlagDetector

        detector = FlagDetector()
        assert detector.scan("no flags here") is None
        assert detector.scan("") is None
        assert detector.scan("flag{") is None  # incomplete

    def test_custom_pattern(self) -> None:
        from app.agents.core.reasoning_loop import FlagDetector

        detector = FlagDetector(flag_patterns=[r"FLAG_(\w+)"])
        assert detector.scan("Got FLAG_hello") == "FLAG_hello"

    def test_scan_dict_recursive(self) -> None:
        from app.agents.core.reasoning_loop import FlagDetector

        detector = FlagDetector()
        data = {
            "output": "nothing",
            "nested": {"result": "The flag is flag{n3st3d} here"},
        }
        assert detector.scan_dict(data) == "flag{n3st3d}"

    def test_scan_dict_list(self) -> None:
        from app.agents.core.reasoning_loop import FlagDetector

        detector = FlagDetector()
        data = ["nope", "found it CTF{list_item}", "nope"]
        assert detector.scan_dict(data) == "CTF{list_item}"

    def test_scan_dict_no_match(self) -> None:
        from app.agents.core.reasoning_loop import FlagDetector

        detector = FlagDetector()
        assert detector.scan_dict({"key": "value"}) is None
        assert detector.scan_dict([]) is None


# ═══════════════════════════════════════════════════════════════════
# ReasoningLoop Flag Termination Tests
# ═══════════════════════════════════════════════════════════════════


class TestReasoningLoopFlagTermination:
    """Tests for flag-detection loop termination in ReasoningLoop."""

    @pytest.mark.asyncio
    async def test_loop_stops_on_flag_in_observation(self) -> None:
        """Loop should stop when a flag is found in action output."""
        from app.agents.core.reasoning_loop import ReasoningLoop

        loop = ReasoningLoop(max_iterations=10)

        call_count = 0

        async def reasoning_fn(state):
            return "Thinking..."

        async def action_fn(action, action_input):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return {"output": "The flag is flag{found_it}"}
            return {"output": "nothing yet"}

        result = await loop.run({}, reasoning_fn, action_fn)

        # Should have stopped early (only 2 iterations, not 10)
        assert result["iterations"] == 2
        assert result["found_flag"] == "flag{found_it}"
        assert result["state"].get("found_flag") == "flag{found_it}"

    @pytest.mark.asyncio
    async def test_flag_in_thought_does_not_trigger(self) -> None:
        """Flags in free-text agent thought should NOT trigger detection."""
        from app.agents.core.reasoning_loop import ReasoningLoop

        loop = ReasoningLoop(max_iterations=5)
        call_count = 0

        async def reasoning_fn(state):
            nonlocal call_count
            call_count += 1
            return "I think the flag is CTF{thought_flag}"

        async def action_fn(action, action_input):
            return {"output": "no flag here"}

        result = await loop.run({}, reasoning_fn, action_fn)

        assert result["found_flag"] is None
        assert result["iterations"] == 5

    @pytest.mark.asyncio
    async def test_loop_runs_full_when_no_flag(self) -> None:
        """Loop should run all iterations when no flag is found."""
        from app.agents.core.reasoning_loop import ReasoningLoop

        loop = ReasoningLoop(max_iterations=3)

        async def reasoning_fn(state):
            return "Thinking..."

        async def action_fn(action, action_input):
            return {"output": "no flag here"}

        result = await loop.run({}, reasoning_fn, action_fn)

        assert result["iterations"] == 3
        assert result["found_flag"] is None

    @pytest.mark.asyncio
    async def test_found_flag_in_state(self) -> None:
        """When flag is found, it should be stored in state."""
        from app.agents.core.reasoning_loop import ReasoningLoop

        loop = ReasoningLoop(max_iterations=5)

        async def reasoning_fn(state):
            return "thinking"

        async def action_fn(action, action_input):
            return "flag{state_test}"

        result = await loop.run({}, reasoning_fn, action_fn)

        assert result["found_flag"] == "flag{state_test}"
        assert result["state"]["found_flag"] == "flag{state_test}"

    @pytest.mark.asyncio
    async def test_loop_returns_found_flag_key(self) -> None:
        """Result dict always contains found_flag key."""
        from app.agents.core.reasoning_loop import ReasoningLoop

        loop = ReasoningLoop(max_iterations=1)

        async def reasoning_fn(state):
            return "nope"

        async def action_fn(action, action_input):
            return {"output": "clean"}

        result = await loop.run({}, reasoning_fn, action_fn)
        assert "found_flag" in result
        assert result["found_flag"] is None

    @pytest.mark.asyncio
    async def test_custom_flag_pattern(self) -> None:
        """Custom flag patterns should be respected."""
        from app.agents.core.reasoning_loop import ReasoningLoop

        loop = ReasoningLoop(
            max_iterations=10,
            flag_patterns=[r"SOLVE\{[^\}]+\}"],
        )

        async def reasoning_fn(state):
            return "thinking"

        async def action_fn(action, action_input):
            return {"output": "SOLVE{custom_pattern}"}

        result = await loop.run({}, reasoning_fn, action_fn)

        assert result["found_flag"] == "SOLVE{custom_pattern}"


# ═══════════════════════════════════════════════════════════════════
# CTF ToolRegistry Integration Tests
# ═══════════════════════════════════════════════════════════════════


class TestCTFToolRegistryIntegration:
    """Integration tests for CTF tools with ToolRegistry (mocked)."""

    def setup_method(self) -> None:
        self.sandbox = _mock_sandbox(_ok_result(stdout="ok\n"))
        self.registry = ToolRegistry()
        self.registry.register_instance(FileTool(self.sandbox))
        self.registry.register_instance(StringsTool(self.sandbox))
        self.registry.register_instance(PythonTool(self.sandbox))
        self.registry.register_instance(EncodingTool(self.sandbox))

    def test_all_ctf_tools_registered(self) -> None:
        tools = self.registry.list_tools(tags=["ctf"])
        names = {t["name"] for t in tools}
        assert names == {"file", "strings", "python", "encoding"}

    @pytest.mark.asyncio
    async def test_file_tool_via_registry(self) -> None:
        self.sandbox.execute = AsyncMock(
            return_value=_ok_result(stdout="text/plain\n")
        )
        request = ToolExecutionRequest(
            tool_name="file",
            parameters={"path": "/test.txt"},
            agent_permissions=[ToolPermission.READ, ToolPermission.EXECUTE],
        )
        result = await self.registry.execute(request)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_strings_tool_via_registry(self) -> None:
        self.sandbox.execute = AsyncMock(
            return_value=_ok_result(stdout="abc\ndef\n")
        )
        request = ToolExecutionRequest(
            tool_name="strings",
            parameters={"path": "/bin"},
            agent_permissions=[ToolPermission.READ, ToolPermission.EXECUTE],
        )
        result = await self.registry.execute(request)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_python_tool_via_registry(self) -> None:
        self.sandbox.execute = AsyncMock(
            return_value=_ok_result(stdout="42\n")
        )
        request = ToolExecutionRequest(
            tool_name="python",
            parameters={"code": "print(42)"},
            agent_permissions=[ToolPermission.EXECUTE],
        )
        result = await self.registry.execute(request)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_encoding_tool_via_registry(self) -> None:
        self.sandbox.execute = AsyncMock(
            return_value=_ok_result(stdout="aGVsbG8=")
        )
        request = ToolExecutionRequest(
            tool_name="encoding",
            parameters={"input": "hello", "encoding": "base64", "mode": "encode"},
            agent_permissions=[ToolPermission.EXECUTE],
        )
        result = await self.registry.execute(request)
        assert result.success is True
