"""
Integration Tests for Docker Sandbox

Tests cover:
- Sandbox integration with ToolRegistry
- Sandbox integration with ExecutorAgent
- Sandbox used as a BaseTool backend
- Sandbox with real workspace workflows
- Sandbox configuration from Settings

Note: These tests mock Docker interactions to avoid requiring a running daemon.
Tests that require a live Docker daemon are marked with @pytest.mark.docker.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.base import AgentConfig
from app.agents.specialized.executor import ExecutorAgent
from app.models.agent import AgentType
from app.models.execution import ExecutionResult
from app.models.state import AgentState, PlanStep
from app.models.tool import (
    RiskLevel,
    SandboxCapability,
    ToolDefinition,
    ToolExecutionRequest,
    ToolPermission,
    ToolResult,
)
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry
from app.sandbox.docker_sandbox import (
    CleanupPolicy,
    DockerSandbox,
    SandboxConfig,
    SandboxExecutionResult,
)


# ── Sandbox-Backed Tool ──────────────────────────────────────────


class SandboxBackedTool(BaseTool):
    """
    A tool that executes code inside a Docker sandbox.

    This demonstrates how the DockerSandbox integrates with the
    existing BaseTool / ToolRegistry framework.
    """

    def __init__(self, sandbox: DockerSandbox) -> None:
        definition = ToolDefinition(
            name="sandbox_code_executor",
            description="Execute Python code in an isolated Docker sandbox",
            version="1.0.0",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds",
                    },
                },
                "required": ["code"],
            },
            returns={
                "type": "object",
                "properties": {
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                    "exit_code": {"type": "integer"},
                },
            },
            tags=["code", "execution", "sandbox", "python"],
            permissions=[ToolPermission.EXECUTE],
            risk_level=RiskLevel.HIGH,
            sandbox_capability=SandboxCapability.CONTAINER,
        )
        super().__init__(definition)
        self.sandbox = sandbox

    async def execute(
        self,
        parameters: Dict[str, Any],
        workspace: Any = None,
    ) -> ToolResult:
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", 30)

        if not code:
            return ToolResult(success=False, error="No code provided")

        result = await self.sandbox.execute(
            f"python -c '{code}'",
            workspace=workspace,
            timeout=timeout,
        )

        return ToolResult(
            success=result.success,
            output={
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
            },
            error=result.stderr if not result.success else None,
            execution_time=result.execution_time,
            metadata={
                "container_id": result.container_id,
                "image": result.image,
                "timed_out": result.timed_out,
            },
        )


# ── ToolRegistry Integration Tests ──────────────────────────────


class TestSandboxToolRegistryIntegration:
    """Tests for sandbox tool integration with ToolRegistry."""

    @pytest.mark.asyncio
    async def test_sandbox_tool_registered_and_executed(self) -> None:
        """Test registering and executing a sandbox-backed tool."""
        sandbox = DockerSandbox(SandboxConfig(pull_policy="never"))
        tool = SandboxBackedTool(sandbox)
        registry = ToolRegistry()
        registry.register_instance(tool)

        mock_container = MagicMock()
        mock_container.short_id = "def5678"
        mock_container.wait.return_value = {"StatusCode": 0, "Error": ""}
        mock_container.logs.return_value = b"hello sandbox"

        with patch.object(sandbox, "_get_client") as mock_get_client:
            client = MagicMock()
            client.ping.return_value = True
            client.images.get.return_value = MagicMock()
            client.containers.create.return_value = mock_container
            mock_get_client.return_value = client

            request = ToolExecutionRequest(
                tool_name="sandbox_code_executor",
                parameters={"code": "print('hello sandbox')"},
                agent_permissions=[
                    ToolPermission.EXECUTE,
                    ToolPermission.READ,
                    ToolPermission.WRITE,
                ],
            )
            result = await registry.execute(request)

            assert result.success is True
            assert result.output["exit_code"] == 0
            assert "container_id" in result.metadata

    @pytest.mark.asyncio
    async def test_sandbox_tool_permissions(self) -> None:
        """Test sandbox tool permission requirements."""
        sandbox = DockerSandbox()
        tool = SandboxBackedTool(sandbox)

        assert ToolPermission.EXECUTE in tool.permissions
        assert tool.risk_level == RiskLevel.HIGH
        assert tool.sandbox_capability == SandboxCapability.CONTAINER

    @pytest.mark.asyncio
    async def test_sandbox_tool_empty_code_fails(self) -> None:
        """Test sandbox tool with empty code returns error."""
        sandbox = DockerSandbox()
        tool = SandboxBackedTool(sandbox)

        result = await tool.execute({"code": ""})
        assert result.success is False
        assert "No code provided" in result.error


# ── ExecutorAgent Integration Tests ──────────────────────────────


class TestSandboxExecutorIntegration:
    """Tests for sandbox integration with ExecutorAgent."""

    @pytest.mark.asyncio
    async def test_executor_runs_sandbox_tool(self) -> None:
        """Test ExecutorAgent executing a sandbox-backed tool step."""
        sandbox = DockerSandbox(SandboxConfig(pull_policy="never"))
        tool = SandboxBackedTool(sandbox)
        registry = ToolRegistry()
        registry.register_instance(tool)

        executor = ExecutorAgent(tool_registry=registry)

        mock_container = MagicMock()
        mock_container.short_id = "sandbox01"
        mock_container.wait.return_value = {"StatusCode": 0, "Error": ""}
        mock_container.logs.return_value = b"42"

        with patch.object(sandbox, "_get_client") as mock_get_client:
            client = MagicMock()
            client.ping.return_value = True
            client.images.get.return_value = MagicMock()
            client.containers.create.return_value = mock_container
            mock_get_client.return_value = client

            state = AgentState()
            state.plan = [
                PlanStep(
                    index=0,
                    description="Run code in sandbox",
                    tool_name="sandbox_code_executor",
                    tool_parameters={"code": "print(6 * 7)"},
                    priority="high",
                ),
            ]

            result = await executor.process({"state": state})

            assert result["success"] is True
            assert len(result["results"]) == 1
            assert result["results"][0]["tool_name"] == "sandbox_code_executor"
            assert "sandbox_code_executor" in result["tools_used"]

    @pytest.mark.asyncio
    async def test_executor_mixed_sandbox_and_regular_tools(self) -> None:
        """Test ExecutorAgent with both sandbox and regular tools."""
        from app.tools.plugins.calculator import CalculatorTool

        sandbox = DockerSandbox(SandboxConfig(pull_policy="never"))
        sandbox_tool = SandboxBackedTool(sandbox)

        registry = ToolRegistry()
        registry.register_instance(sandbox_tool)
        registry.register_instance(CalculatorTool())

        executor = ExecutorAgent(tool_registry=registry)

        mock_container = MagicMock()
        mock_container.short_id = "mix0001"
        mock_container.wait.return_value = {"StatusCode": 0, "Error": ""}
        mock_container.logs.return_value = b"computed"

        with patch.object(sandbox, "_get_client") as mock_get_client:
            client = MagicMock()
            client.ping.return_value = True
            client.images.get.return_value = MagicMock()
            client.containers.create.return_value = mock_container
            mock_get_client.return_value = client

            state = AgentState()
            state.plan = [
                PlanStep(
                    index=0,
                    description="Calculate sum",
                    tool_name="calculator",
                    tool_parameters={"expression": "2 + 2"},
                ),
                PlanStep(
                    index=1,
                    description="Run in sandbox",
                    tool_name="sandbox_code_executor",
                    tool_parameters={"code": "print('done')"},
                    depends_on=[0],
                ),
            ]

            result = await executor.process({"state": state})

            assert result["success"] is True
            assert len(result["results"]) == 2
            assert "calculator" in result["tools_used"]
            assert "sandbox_code_executor" in result["tools_used"]


# ── Workspace Workflow Tests ─────────────────────────────────────


class TestWorkspaceWorkflow:
    """Tests for complete workspace-based workflows."""

    @pytest.mark.asyncio
    async def test_multi_file_workspace(self) -> None:
        """Test executing with multiple workspace files."""
        workspace_dir = tempfile.mkdtemp()
        try:
            config = SandboxConfig(
                workspace_dir=workspace_dir,
                pull_policy="never",
                cleanup_policy=CleanupPolicy.NEVER,
            )
            sandbox = DockerSandbox(config)

            mock_container = MagicMock()
            mock_container.short_id = "ws00001"
            mock_container.wait.return_value = {"StatusCode": 0, "Error": ""}
            mock_container.logs.return_value = b"result: 42"

            with patch.object(sandbox, "_get_client") as mock_get_client:
                client = MagicMock()
                client.ping.return_value = True
                client.images.get.return_value = MagicMock()
                client.containers.create.return_value = mock_container
                mock_get_client.return_value = client

                files = {
                    "main.py": "print('result: 42')",
                    "config.json": '{"debug": true}',
                    "data/input.csv": "a,b,c\n1,2,3",
                }

                result = await sandbox.execute(
                    "python /workspace/main.py",
                    workspace_id="multi-file-task",
                    files=files,
                )

                assert result.success is True

                # Verify workspace was created with files
                ws_path = Path(workspace_dir) / "sandbox_workspaces" / "multi-file-task"
                assert ws_path.exists()
                assert (ws_path / "main.py").exists()
                assert (ws_path / "config.json").exists()
                assert (ws_path / "data" / "input.csv").exists()
        finally:
            shutil.rmtree(workspace_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_workspace_cleanup_on_success(self) -> None:
        """Test workspace cleanup with ON_SUCCESS policy."""
        workspace_dir = tempfile.mkdtemp()
        try:
            config = SandboxConfig(
                workspace_dir=workspace_dir,
                pull_policy="never",
                cleanup_policy=CleanupPolicy.ON_SUCCESS,
            )
            sandbox = DockerSandbox(config)

            mock_container = MagicMock()
            mock_container.short_id = "cln0001"
            mock_container.wait.return_value = {"StatusCode": 0, "Error": ""}
            mock_container.logs.return_value = b"done"

            with patch.object(sandbox, "_get_client") as mock_get_client:
                client = MagicMock()
                client.ping.return_value = True
                client.images.get.return_value = MagicMock()
                client.containers.create.return_value = mock_container
                mock_get_client.return_value = client

                result = await sandbox.execute(
                    "echo done",
                    workspace_id="cleanup-test",
                    files={"test.txt": "content"},
                )

                assert result.success is True
                # Workspace should be cleaned up on success
                ws_path = Path(workspace_dir) / "sandbox_workspaces" / "cleanup-test"
                assert not ws_path.exists()
        finally:
            shutil.rmtree(workspace_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_workspace_preserved_on_failure(self) -> None:
        """Test workspace is preserved when ON_SUCCESS policy is used and execution fails."""
        workspace_dir = tempfile.mkdtemp()
        try:
            config = SandboxConfig(
                workspace_dir=workspace_dir,
                pull_policy="never",
                cleanup_policy=CleanupPolicy.ON_SUCCESS,
            )
            sandbox = DockerSandbox(config)

            mock_container = MagicMock()
            mock_container.short_id = "fail001"
            mock_container.wait.return_value = {"StatusCode": 1, "Error": ""}
            mock_container.logs.return_value = b"error"

            with patch.object(sandbox, "_get_client") as mock_get_client:
                client = MagicMock()
                client.ping.return_value = True
                client.images.get.return_value = MagicMock()
                client.containers.create.return_value = mock_container
                mock_get_client.return_value = client

                result = await sandbox.execute(
                    "false",
                    workspace_id="preserve-test",
                    files={"debug.txt": "debug info"},
                )

                assert result.success is False
                # Workspace should be preserved on failure
                ws_path = Path(workspace_dir) / "sandbox_workspaces" / "preserve-test"
                assert ws_path.exists()
                assert (ws_path / "debug.txt").exists()
        finally:
            shutil.rmtree(workspace_dir, ignore_errors=True)


# ── Config Integration Tests ─────────────────────────────────────


class TestConfigIntegration:
    """Tests for sandbox configuration from Settings."""

    def test_sandbox_config_from_settings(self) -> None:
        """Test creating SandboxConfig from application Settings."""
        from app.config import Settings

        settings = Settings(
            sandbox_enabled=True,
            sandbox_image="python:3.11-slim",
            sandbox_memory_limit="256m",
            sandbox_cpu_limit=0.5,
            sandbox_timeout=15,
            sandbox_network_disabled=False,
        )

        config = SandboxConfig(
            image=settings.sandbox_image,
            memory_limit=settings.sandbox_memory_limit,
            cpu_limit=settings.sandbox_cpu_limit,
            timeout=settings.sandbox_timeout,
            network_disabled=settings.sandbox_network_disabled,
        )

        assert config.image == "python:3.11-slim"
        assert config.memory_limit == "256m"
        assert config.cpu_limit == 0.5
        assert config.timeout == 15
        assert config.network_disabled is False

    def test_sandbox_defaults_match_settings(self) -> None:
        """Test that SandboxConfig defaults align with Settings defaults."""
        from app.config import Settings

        settings = Settings()
        config = SandboxConfig()

        assert config.memory_limit == settings.sandbox_memory_limit
        assert config.cpu_limit == settings.sandbox_cpu_limit
        assert config.timeout == settings.sandbox_timeout
        assert config.network_disabled == settings.sandbox_network_disabled
