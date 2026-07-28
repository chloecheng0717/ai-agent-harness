"""
Unit Tests for Docker Sandbox

Tests cover:
- SandboxConfig validation and defaults
- SandboxExecutionResult creation
- DockerSandbox initialization
- Command execution (mocked Docker)
- Timeout handling
- Workspace file preparation and cleanup
- Batch execution (sequential and parallel)
- Container cleanup policies
- Error handling and edge cases
- Health check
- Context manager lifecycle
- Integration with ToolRegistry / ExecutorAgent patterns

All Docker interactions are mocked — no daemon required.
"""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.sandbox.docker_sandbox import (
    CleanupPolicy,
    ContainerExecutionError,
    DockerNotAvailableError,
    DockerSandbox,
    ImageBuildError,
    SandboxConfig,
    SandboxError,
    SandboxExecutionResult,
    SandboxTimeoutError,
    WorkspaceSetupError,
)


# ── Helpers ──────────────────────────────────────────────────────


def _default_config(**overrides: Any) -> SandboxConfig:
    """Create a default SandboxConfig with optional overrides."""
    return SandboxConfig(**overrides)


def _mock_container(
    exit_code: int = 0,
    stdout: str = "output",
    stderr: str = "",
) -> MagicMock:
    """Create a mock Docker container."""
    container = MagicMock()
    container.short_id = "abc1234"
    container.wait.return_value = {"StatusCode": exit_code, "Error": ""}
    container.logs.side_effect = lambda stdout=True, stderr=False: (
        stdout if stdout else (stderr if stderr else b"")
    )
    # Make logs return bytes when stdout/stderr is True
    original_logs = container.logs

    def logs_side_effect(stdout=True, stderr=False):
        if stdout:
            return stdout.encode() if isinstance(stdout, str) else b"output"
        if stderr:
            return stderr.encode() if isinstance(stderr, str) else b""
        return b""

    container.logs.side_effect = logs_side_effect
    return container


def _mock_docker_client(containers: Any = None) -> MagicMock:
    """Create a mock Docker client."""
    client = MagicMock()
    client.ping.return_value = True
    client.images.get.return_value = MagicMock()
    client.images.pull.return_value = MagicMock()

    if containers is None:
        containers = _mock_container()
    client.containers.create.return_value = containers

    return client


# ── SandboxConfig Tests ──────────────────────────────────────────


class TestSandboxConfig:
    """Tests for SandboxConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = SandboxConfig()
        assert config.image == "python:3.12-slim"
        assert config.memory_limit == "512m"
        assert config.cpu_limit == 1.0
        assert config.timeout == 30
        assert config.network_disabled is True
        assert config.cleanup_policy == CleanupPolicy.ALWAYS
        assert config.working_dir == "/workspace"
        assert config.read_only_rootfs is False
        assert config.pull_policy == "if_missing"

    def test_cpu_quota_calculation(self) -> None:
        """Test that cpu_quota is auto-calculated from cpu_limit and cpu_period."""
        config = SandboxConfig(cpu_period=100_000, cpu_limit=2.0)
        assert config.cpu_quota == 200_000

    def test_cpu_quota_fractional(self) -> None:
        """Test cpu_quota with fractional CPU limit."""
        config = SandboxConfig(cpu_period=100_000, cpu_limit=0.5)
        assert config.cpu_quota == 50_000

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SandboxConfig(
            image="ubuntu:22.04",
            memory_limit="1g",
            cpu_limit=4.0,
            timeout=120,
            network_disabled=False,
            cleanup_policy=CleanupPolicy.NEVER,
            working_dir="/app",
        )
        assert config.image == "ubuntu:22.04"
        assert config.memory_limit == "1g"
        assert config.cpu_limit == 4.0
        assert config.timeout == 120
        assert config.network_disabled is False
        assert config.cleanup_policy == CleanupPolicy.NEVER
        assert config.working_dir == "/app"

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = SandboxConfig()
        with pytest.raises(AttributeError):
            config.image = "new:image"  # type: ignore[misc]

    def test_environment_defaults_empty(self) -> None:
        """Test that environment defaults to empty dict."""
        config = SandboxConfig()
        assert config.environment == {}

    def test_tmpfs_defaults(self) -> None:
        """Test tmpfs default mounts."""
        config = SandboxConfig()
        assert "/tmp" in config.tmpfs_mounts


class TestCleanupPolicy:
    """Tests for CleanupPolicy enum."""

    def test_values(self) -> None:
        assert CleanupPolicy.ALWAYS.value == "always"
        assert CleanupPolicy.ON_SUCCESS.value == "on_success"
        assert CleanupPolicy.NEVER.value == "never"


# ── SandboxExecutionResult Tests ─────────────────────────────────


class TestSandboxExecutionResult:
    """Tests for SandboxExecutionResult dataclass."""

    def test_defaults(self) -> None:
        """Test default result values."""
        result = SandboxExecutionResult()
        assert result.success is False
        assert result.exit_code == -1
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.execution_time == 0.0
        assert result.container_id is None
        assert result.image == ""
        assert result.timed_out is False
        assert result.metadata == {}

    def test_successful_result(self) -> None:
        """Test creating a successful result."""
        result = SandboxExecutionResult(
            success=True,
            exit_code=0,
            stdout="hello world\n",
            execution_time=150.5,
            container_id="abc1234",
            image="python:3.12-slim",
        )
        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == "hello world\n"
        assert result.execution_time == 150.5
        assert result.container_id == "abc1234"

    def test_failed_result(self) -> None:
        """Test creating a failed result."""
        result = SandboxExecutionResult(
            success=False,
            exit_code=1,
            stderr="error message",
            execution_time=50.0,
        )
        assert result.success is False
        assert result.exit_code == 1
        assert result.stderr == "error message"

    def test_timed_out_result(self) -> None:
        """Test creating a timed-out result."""
        result = SandboxExecutionResult(
            success=False,
            exit_code=-1,
            timed_out=True,
            stderr="Execution timed out after 10s",
        )
        assert result.timed_out is True
        assert result.exit_code == -1


# ── DockerSandbox Initialization Tests ───────────────────────────


class TestDockerSandboxInit:
    """Tests for DockerSandbox initialization."""

    def test_default_init(self) -> None:
        """Test initialization with default config."""
        sandbox = DockerSandbox()
        assert sandbox.config is not None
        assert sandbox.config.image == "python:3.12-slim"
        assert sandbox._client is None
        assert sandbox._active_containers == {}

    def test_custom_config_init(self) -> None:
        """Test initialization with custom config."""
        config = SandboxConfig(image="ubuntu:22.04", timeout=60)
        sandbox = DockerSandbox(config)
        assert sandbox.config.image == "ubuntu:22.04"
        assert sandbox.config.timeout == 60

    def test_is_logger_bound(self) -> None:
        """Test that logger is properly bound."""
        sandbox = DockerSandbox()
        assert sandbox.logger is not None


# ── DockerSandbox Health Check Tests ─────────────────────────────


class TestHealthCheck:
    """Tests for DockerSandbox health check."""

    @pytest.mark.asyncio
    async def test_health_check_success(self) -> None:
        """Test successful health check."""
        sandbox = DockerSandbox()
        sandbox._client = _mock_docker_client()

        result = await sandbox.check_health()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_docker_not_installed(self) -> None:
        """Test health check when Docker SDK is not installed."""
        sandbox = DockerSandbox()

        with patch("app.sandbox.docker_sandbox._DOCKER_AVAILABLE", False):
            result = await sandbox.check_health()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self) -> None:
        """Test health check when Docker daemon is unreachable."""
        sandbox = DockerSandbox()

        with patch("app.sandbox.docker_sandbox.docker") as mock_docker:
            mock_docker.from_env.side_effect = Exception("Cannot connect")
            mock_docker.errors = MagicMock()
            result = await sandbox.check_health()
            assert result is False


# ── DockerSandbox Execute Tests ──────────────────────────────────


class TestExecute:
    """Tests for DockerSandbox.execute."""

    @pytest.mark.asyncio
    async def test_execute_simple_command(self) -> None:
        """Test executing a simple command."""
        sandbox = DockerSandbox(SandboxConfig(pull_policy="never"))
        mock_container = _mock_container(exit_code=0, stdout="hello\n")

        with patch.object(sandbox, "_get_client") as mock_get_client:
            client = _mock_docker_client(mock_container)
            mock_get_client.return_value = client

            result = await sandbox.execute("echo hello")

            assert result.success is True
            assert result.exit_code == 0
            assert result.container_id == "abc1234"

    @pytest.mark.asyncio
    async def test_execute_command_failure(self) -> None:
        """Test executing a command that fails."""
        sandbox = DockerSandbox(SandboxConfig(pull_policy="never"))
        mock_container = _mock_container(exit_code=1)

        with patch.object(sandbox, "_get_client") as mock_get_client:
            client = _mock_docker_client(mock_container)
            mock_get_client.return_value = client

            result = await sandbox.execute("false")

            assert result.success is False
            assert result.exit_code == 1

    @pytest.mark.asyncio
    async def test_execute_with_timeout_override(self) -> None:
        """Test executing with a timeout override."""
        sandbox = DockerSandbox(SandboxConfig(pull_policy="never"))
        mock_container = _mock_container(exit_code=0)

        with patch.object(sandbox, "_get_client") as mock_get_client:
            client = _mock_docker_client(mock_container)
            mock_get_client.return_value = client

            result = await sandbox.execute("echo test", timeout=5)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_timeout_kills_container(self) -> None:
        """Test that timeout kills the container."""
        sandbox = DockerSandbox(
            SandboxConfig(pull_policy="never", timeout=1)
        )
        mock_container = _mock_container(exit_code=0)

        with patch.object(sandbox, "_get_client") as mock_get_client:
            client = _mock_docker_client(mock_container)
            mock_get_client.return_value = client

            # Make container.wait raise TimeoutError
            async def slow_wait(timeout=None):
                await asyncio.sleep(10)

            with patch("app.sandbox.docker_sandbox.asyncio") as mock_asyncio:
                loop = asyncio.get_event_loop()
                mock_asyncio.get_event_loop.return_value = loop
                mock_asyncio.wait_for = AsyncMock(
                    side_effect=asyncio.TimeoutError
                )
                mock_asyncio.TimeoutError = asyncio.TimeoutError

                result = await sandbox.execute("sleep 100")
                assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_execute_with_workspace_files(self) -> None:
        """Test executing with workspace file creation."""
        workspace_dir = tempfile.mkdtemp()
        try:
            config = SandboxConfig(
                workspace_dir=workspace_dir,
                pull_policy="never",
                cleanup_policy=CleanupPolicy.NEVER,
            )
            sandbox = DockerSandbox(config)
            mock_container = _mock_container(exit_code=0)

            with patch.object(sandbox, "_get_client") as mock_get_client:
                client = _mock_docker_client(mock_container)
                mock_get_client.return_value = client

                result = await sandbox.execute(
                    "cat /workspace/test.txt",
                    workspace_id="test-task-1",
                    files={"test.txt": "hello workspace"},
                )

                # Verify workspace was created
                ws_path = Path(workspace_dir) / "sandbox_workspaces" / "test-task-1"
                assert ws_path.exists()
                assert (ws_path / "test.txt").read_text() == "hello workspace"

                assert result.success is True
        finally:
            shutil.rmtree(workspace_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_execute_with_environment(self) -> None:
        """Test executing with custom environment variables."""
        sandbox = DockerSandbox(
            SandboxConfig(pull_policy="never", environment={"BASE_VAR": "base"})
        )
        mock_container = _mock_container(exit_code=0)

        with patch.object(sandbox, "_get_client") as mock_get_client:
            client = _mock_docker_client(mock_container)
            mock_get_client.return_value = client

            result = await sandbox.execute(
                "echo $MY_VAR",
                environment={"MY_VAR": "custom_value"},
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_custom_image(self) -> None:
        """Test executing with a custom image override."""
        sandbox = DockerSandbox(SandboxConfig(pull_policy="never"))
        mock_container = _mock_container(exit_code=0)

        with patch.object(sandbox, "_get_client") as mock_get_client:
            client = _mock_docker_client(mock_container)
            mock_get_client.return_value = client

            result = await sandbox.execute(
                "echo test", image="ubuntu:22.04"
            )
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_docker_not_available(self) -> None:
        """Test execution when Docker is not available."""
        sandbox = DockerSandbox()

        with patch.object(sandbox, "_get_client", side_effect=DockerNotAvailableError("No Docker")):
            with pytest.raises(DockerNotAvailableError):
                await sandbox.execute("echo hello")

    @pytest.mark.asyncio
    async def test_execute_exception_during_execution(self) -> None:
        """Test handling of unexpected exceptions during execution."""
        sandbox = DockerSandbox(SandboxConfig(pull_policy="never"))

        with patch.object(sandbox, "_get_client") as mock_get_client:
            client = MagicMock()
            client.ping.return_value = True
            client.images.get.return_value = MagicMock()
            client.containers.create.side_effect = RuntimeError("Unexpected error")
            mock_get_client.return_value = client

            with pytest.raises(SandboxError, match="Unexpected error"):
                await sandbox.execute("echo hello")


# ── Workspace Tests ──────────────────────────────────────────────


class TestWorkspace:
    """Tests for workspace preparation and cleanup."""

    def test_prepare_workspace_creates_directory(self) -> None:
        """Test that workspace directory is created."""
        workspace_dir = tempfile.mkdtemp()
        try:
            config = SandboxConfig(workspace_dir=workspace_dir)
            sandbox = DockerSandbox(config)

            path = sandbox._prepare_workspace("task-123")
            assert path.exists()
            assert path.is_dir()
            assert path.name == "task-123"
        finally:
            shutil.rmtree(workspace_dir, ignore_errors=True)

    def test_prepare_workspace_creates_files(self) -> None:
        """Test that workspace files are created."""
        workspace_dir = tempfile.mkdtemp()
        try:
            config = SandboxConfig(workspace_dir=workspace_dir)
            sandbox = DockerSandbox(config)

            files = {
                "script.py": "print('hello')",
                "data/input.txt": "test data",
            }
            path = sandbox._prepare_workspace("task-456", files=files)

            assert (path / "script.py").read_text() == "print('hello')"
            assert (path / "data" / "input.txt").read_text() == "test data"
        finally:
            shutil.rmtree(workspace_dir, ignore_errors=True)

    def test_cleanup_workspace_removes_directory(self) -> None:
        """Test that workspace cleanup removes the directory."""
        workspace_dir = tempfile.mkdtemp()
        try:
            config = SandboxConfig(workspace_dir=workspace_dir)
            sandbox = DockerSandbox(config)

            sandbox._prepare_workspace("task-789")
            ws_path = Path(workspace_dir) / "sandbox_workspaces" / "task-789"
            assert ws_path.exists()

            sandbox._cleanup_workspace("task-789")
            assert not ws_path.exists()
        finally:
            shutil.rmtree(workspace_dir, ignore_errors=True)

    def test_cleanup_nonexistent_workspace(self) -> None:
        """Test cleanup of non-existent workspace doesn't raise."""
        config = SandboxConfig(workspace_dir=tempfile.mkdtemp())
        sandbox = DockerSandbox(config)
        # Should not raise
        sandbox._cleanup_workspace("nonexistent-task")

    def test_prepare_workspace_with_empty_files(self) -> None:
        """Test workspace preparation with empty files dict."""
        workspace_dir = tempfile.mkdtemp()
        try:
            config = SandboxConfig(workspace_dir=workspace_dir)
            sandbox = DockerSandbox(config)

            path = sandbox._prepare_workspace("task-empty", files={})
            assert path.exists()
            # Only the directory should exist
            assert list(path.iterdir()) == []
        finally:
            shutil.rmtree(workspace_dir, ignore_errors=True)


# ── Batch Execution Tests ────────────────────────────────────────


class TestBatchExecution:
    """Tests for DockerSandbox.execute_batch."""

    @pytest.mark.asyncio
    async def test_batch_empty_commands(self) -> None:
        """Test batch with no commands returns empty list."""
        sandbox = DockerSandbox()
        result = await sandbox.execute_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_batch_sequential_success(self) -> None:
        """Test sequential batch execution."""
        sandbox = DockerSandbox(SandboxConfig(pull_policy="never"))
        mock_container = _mock_container(exit_code=0)

        with patch.object(sandbox, "_get_client") as mock_get_client:
            client = _mock_docker_client(mock_container)
            mock_get_client.return_value = client

            results = await sandbox.execute_batch(
                ["echo 1", "echo 2", "echo 3"],
            )
            assert len(results) == 3
            assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_batch_sequential_stops_on_failure(self) -> None:
        """Test sequential batch stops on first failure."""
        sandbox = DockerSandbox(SandboxConfig(pull_policy="never"))

        call_count = 0

        def make_container():
            nonlocal call_count
            call_count += 1
            return _mock_container(exit_code=0 if call_count <= 1 else 1)

        with patch.object(sandbox, "_get_client") as mock_get_client:
            client = MagicMock()
            client.ping.return_value = True
            client.images.get.return_value = MagicMock()
            client.containers.create.side_effect = lambda **kw: make_container()
            mock_get_client.return_value = client

            results = await sandbox.execute_batch(
                ["echo 1", "false", "echo 3"],
            )
            # First succeeds, second fails, third is skipped
            assert len(results) == 3
            assert results[0].success is True
            assert results[1].success is False
            assert results[2].success is False
            assert "Skipped" in results[2].stderr

    @pytest.mark.asyncio
    async def test_batch_parallel(self) -> None:
        """Test parallel batch execution."""
        sandbox = DockerSandbox(SandboxConfig(pull_policy="never"))
        mock_container = _mock_container(exit_code=0)

        with patch.object(sandbox, "_get_client") as mock_get_client:
            client = _mock_docker_client(mock_container)
            mock_get_client.return_value = client

            results = await sandbox.execute_batch(
                ["echo 1", "echo 2", "echo 3"],
                parallel=True,
            )
            assert len(results) == 3
            assert all(r.success for r in results)


# ── Cleanup Tests ────────────────────────────────────────────────


class TestCleanup:
    """Tests for container and sandbox cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_all_removes_containers(self) -> None:
        """Test that cleanup_all removes all tracked containers."""
        sandbox = DockerSandbox()
        container1 = MagicMock()
        container2 = MagicMock()
        sandbox._active_containers = {"c1": container1, "c2": container2}

        removed = await sandbox.cleanup_all()
        assert removed == 2
        assert len(sandbox._active_containers) == 0

    @pytest.mark.asyncio
    async def test_cleanup_all_handles_errors(self) -> None:
        """Test that cleanup_all handles removal errors gracefully."""
        sandbox = DockerSandbox()
        container1 = MagicMock()
        container1.remove.side_effect = Exception("Remove failed")
        container2 = MagicMock()
        sandbox._active_containers = {"c1": container1, "c2": container2}

        removed = await sandbox.cleanup_all()
        assert removed == 1  # Only c2 was removed
        assert len(sandbox._active_containers) == 0

    @pytest.mark.asyncio
    async def test_close_clears_client(self) -> None:
        """Test that close() clears the client reference."""
        sandbox = DockerSandbox()
        sandbox._client = MagicMock()

        await sandbox.close()
        assert sandbox._client is None

    @pytest.mark.asyncio
    async def test_cleanup_policy_never_keeps_container(self) -> None:
        """Test that NEVER policy keeps containers."""
        config = SandboxConfig(cleanup_policy=CleanupPolicy.NEVER, pull_policy="never")
        sandbox = DockerSandbox(config)
        mock_container = _mock_container(exit_code=0)

        with patch.object(sandbox, "_get_client") as mock_get_client:
            client = _mock_docker_client(mock_container)
            mock_get_client.return_value = client

            result = await sandbox.execute("echo test")
            assert result.success is True
            # Container should still be tracked
            assert len(sandbox._active_containers) == 1


# ── Context Manager Tests ────────────────────────────────────────


class TestContextManager:
    """Tests for async context manager usage."""

    @pytest.mark.asyncio
    async def test_context_manager_enter_exit(self) -> None:
        """Test async context manager usage."""
        async with DockerSandbox() as sandbox:
            assert sandbox is not None
            sandbox._client = MagicMock()

        # After exit, client should be None
        assert sandbox._client is None

    @pytest.mark.asyncio
    async def test_context_manager_cleans_up(self) -> None:
        """Test that context manager cleans up containers."""
        async with DockerSandbox() as sandbox:
            container = MagicMock()
            sandbox._active_containers = {"c1": container}

        container.remove.assert_called_once_with(force=True)
        assert len(sandbox._active_containers) == 0


# ── Error Class Tests ────────────────────────────────────────────


class TestExceptions:
    """Tests for sandbox exception classes."""

    def test_sandbox_error(self) -> None:
        """Test SandboxError."""
        exc = SandboxError("test error")
        assert str(exc) == "test error"
        assert isinstance(exc, Exception)

    def test_docker_not_available_error(self) -> None:
        """Test DockerNotAvailableError."""
        exc = DockerNotAvailableError("docker not found")
        assert isinstance(exc, SandboxError)
        assert "docker not found" in str(exc)

    def test_container_execution_error(self) -> None:
        """Test ContainerExecutionError."""
        exc = ContainerExecutionError(exit_code=1, stderr="error", stdout="output")
        assert exc.exit_code == 1
        assert exc.stderr == "error"
        assert exc.stdout == "output"
        assert isinstance(exc, SandboxError)

    def test_sandbox_timeout_error(self) -> None:
        """Test SandboxTimeoutError."""
        exc = SandboxTimeoutError("timed out")
        assert isinstance(exc, SandboxError)

    def test_image_build_error(self) -> None:
        """Test ImageBuildError."""
        exc = ImageBuildError("build failed")
        assert isinstance(exc, SandboxError)

    def test_workspace_setup_error(self) -> None:
        """Test WorkspaceSetupError."""
        exc = WorkspaceSetupError("setup failed")
        assert isinstance(exc, SandboxError)


# ── Integration Pattern Tests ────────────────────────────────────


class TestIntegrationPatterns:
    """Tests demonstrating integration with the AI Agent Harness patterns."""

    @pytest.mark.asyncio
    async def test_sandbox_as_tool_execution_backend(self) -> None:
        """Test using sandbox as a tool execution backend."""
        sandbox = DockerSandbox(
            SandboxConfig(
                pull_policy="never",
                memory_limit="256m",
                cpu_limit=0.5,
                timeout=10,
            )
        )
        mock_container = _mock_container(exit_code=0, stdout="42\n")

        with patch.object(sandbox, "_get_client") as mock_get_client:
            client = _mock_docker_client(mock_container)
            mock_get_client.return_value = client

            result = await sandbox.execute(
                "python -c 'print(6 * 7)'",
                environment={"PYTHONUNBUFFERED": "1"},
            )

            assert result.success is True
            assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_sandbox_result_as_tool_result(self) -> None:
        """Test converting SandboxExecutionResult to ToolResult-like dict."""
        from app.models.tool import ToolResult

        sandbox_result = SandboxExecutionResult(
            success=True,
            exit_code=0,
            stdout='{"result": 42}',
            execution_time=150.0,
            container_id="abc1234",
        )

        # Convert to ToolResult-compatible format
        tool_result = ToolResult(
            success=sandbox_result.success,
            output={"stdout": sandbox_result.stdout, "exit_code": sandbox_result.exit_code},
            execution_time=sandbox_result.execution_time,
            metadata={"container_id": sandbox_result.container_id},
        )

        assert tool_result.success is True
        assert tool_result.execution_time == 150.0
        assert tool_result.metadata["container_id"] == "abc1234"

    @pytest.mark.asyncio
    async def test_sandbox_isolation_between_executions(self) -> None:
        """Test that each execution uses a fresh container."""
        sandbox = DockerSandbox(SandboxConfig(pull_policy="never"))
        mock_container = _mock_container(exit_code=0)

        with patch.object(sandbox, "_get_client") as mock_get_client:
            client = _mock_docker_client(mock_container)
            mock_get_client.return_value = client

            r1 = await sandbox.execute("echo 1")
            r2 = await sandbox.execute("echo 2")

            # Each execution creates a new container
            assert client.containers.create.call_count == 2
