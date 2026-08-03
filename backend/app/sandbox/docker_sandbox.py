"""
Docker Execution Sandbox

Production-ready Docker-based sandbox for executing untrusted code and
commands in isolated containers. Provides per-task workspace mounting,
resource limits, stdout/stderr capture, and automatic cleanup.

This module is designed for integration with the AI Agent Harness
ToolRegistry and ExecutorAgent, but can also be used standalone.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from app.sandbox.workspace import Workspace

try:
    import docker
    from docker.errors import (
        APIError,
        BuildError,
        ContainerError,
        ImageNotFound,
        NotFound,
    )
    from docker.models.containers import Container
    from docker.types import Mount

    _DOCKER_AVAILABLE = True
except ImportError:
    docker = None  # type: ignore[assignment]
    _DOCKER_AVAILABLE = False


# ── Exceptions ──────────────────────────────────────────────────


class SandboxError(Exception):
    """Base exception for sandbox operations."""


class DockerNotAvailableError(SandboxError):
    """Raised when the Docker daemon is not reachable."""


class ContainerExecutionError(SandboxError):
    """Raised when a container fails to execute a command."""

    def __init__(
        self,
        exit_code: int,
        stderr: str,
        stdout: str = "",
        container_id: Optional[str] = None,
        command: str = "",
    ) -> None:
        self.exit_code = exit_code
        self.stderr = stderr
        self.stdout = stdout
        self.container_id = container_id
        self.command = command
        super().__init__(
            "Container execution failed"
            f"{' (container ' + container_id + ')' if container_id else ''}"
            f"{' command: ' + command if command else ''}"
            f" exited with code {exit_code}."
            f"\nstdout:\n{stdout}"
            f"\nstderr:\n{stderr}"
        )


class SandboxTimeoutError(SandboxError):
    """Raised when execution exceeds the configured timeout."""


class ImageBuildError(SandboxError):
    """Raised when building a Docker image fails."""


class WorkspaceSetupError(SandboxError):
    """Raised when workspace preparation fails."""


# ── Configuration ───────────────────────────────────────────────


class CleanupPolicy(str, Enum):
    """Container cleanup policies."""

    ALWAYS = "always"
    ON_SUCCESS = "on_success"
    NEVER = "never"


@dataclass(frozen=True)
class SandboxConfig:
    """
    Configuration for a DockerSandbox instance.

    Attributes:
        image: Docker image to use (default: ``python:3.12-slim``).
        memory_limit: Memory limit for the container (e.g. ``"512m"``).
        cpu_limit: CPU limit as a float (number of CPUs, e.g. ``1.0``).
        cpu_period: CPU CFS scheduler period in microseconds.
        cpu_quota: CPU CFS quota in microseconds per ``cpu_period``.
        timeout: Maximum execution time in seconds.
        workspace_dir: Base directory for per-task workspace mounts.
        cleanup_policy: When to remove containers after execution.
        network_disabled: Disable networking inside the container.
        read_only_rootfs: Mount the root filesystem as read-only.
        tmpfs_mounts: tmpfs mounts inside the container.
        environment: Extra environment variables for the container.
        user: User inside the container (e.g. ``"1000:1000"``).
        working_dir: Working directory inside the container.
        additional_volumes: Extra volume mounts.
        pull_policy: When to pull the image (``"always"``, ``"if_missing"``, ``"never"``).
    """

    image: str = "python:3.12-slim"
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    cpu_period: int = 100_000
    cpu_quota: int = 100_000
    timeout: int = 30
    workspace_dir: Optional[str] = None
    cleanup_policy: CleanupPolicy = CleanupPolicy.ALWAYS
    network_disabled: bool = True
    read_only_rootfs: bool = False
    tmpfs_mounts: Dict[str, str] = field(
        default_factory=lambda: {"/tmp": "size=100m"}
    )
    environment: Dict[str, str] = field(default_factory=dict)
    user: Optional[str] = None
    working_dir: str = "/workspace"
    additional_volumes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    pull_policy: str = "if_missing"

    def __post_init__(self) -> None:
        if self.cpu_limit > 0:
            object.__setattr__(
                self,
                "cpu_quota",
                int(self.cpu_period * self.cpu_limit),
            )


# ── Execution Result ────────────────────────────────────────────


@dataclass
class SandboxExecutionResult:
    """
    Result of a sandboxed command execution.

    Attributes:
        success: Whether the command exited with code 0.
        exit_code: The process exit code.
        stdout: Captured standard output.
        stderr: Captured standard error.
        execution_time: Wall-clock execution time in milliseconds.
        container_id: ID of the container used (if created).
        image: Docker image used for execution.
        timed_out: Whether the execution was killed by the timeout.
        metadata: Additional metadata about the execution.
    """

    success: bool = False
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    container_id: Optional[str] = None
    image: str = ""
    timed_out: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Docker Sandbox ──────────────────────────────────────────────


class DockerSandbox:
    """
    Production-ready Docker execution sandbox.

    Creates isolated Docker containers for each execution, mounts a
    per-task workspace, executes commands, and captures output. Supports
    configurable timeout, memory limit, CPU limit, and automatic cleanup.

    Usage::

        sandbox = DockerSandbox(SandboxConfig(timeout=10))
        result = await sandbox.execute("echo hello")
        print(result.stdout)  # "hello\n"

    With workspace::

        sandbox = DockerSandbox(SandboxConfig(workspace_dir="/tmp/workspaces"))
        result = await sandbox.execute(
            "python script.py",
            workspace_id="task-123",
            files={"script.py": "print('hello')"},
        )

    Parameters
    ----------
    config : SandboxConfig | None
        Sandbox configuration.  Uses defaults when ``None``.
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self.config = config or SandboxConfig()
        self._client: Optional[Any] = None
        self._active_containers: Dict[str, Any] = {}
        self._persistent_workspaces: Dict[str, Workspace] = {}

        self.logger = structlog.get_logger().bind(
            component="docker_sandbox",
            image=self.config.image,
        )

        self.logger.info(
            "Docker sandbox initialized",
            memory_limit=self.config.memory_limit,
            cpu_limit=self.config.cpu_limit,
            timeout=self.config.timeout,
            cleanup_policy=self.config.cleanup_policy.value,
            network_disabled=self.config.network_disabled,
        )

    # ── Client management ──────────────────────────────────────

    def _get_client(self) -> Any:
        """
        Get or create the Docker client.

        Raises
        ------
        DockerNotAvailableError
            If the Docker daemon is not reachable.
        """
        if self._client is None:
            if not _DOCKER_AVAILABLE:
                raise DockerNotAvailableError(
                    "Docker SDK not installed. "
                    "Install with: pip install docker"
                )
            try:
                self._client = docker.from_env()
                self._client.ping()
            except Exception as exc:
                raise DockerNotAvailableError(
                    f"Cannot connect to Docker daemon: {exc}"
                ) from exc
        return self._client

    async def check_health(self) -> bool:
        """
        Check if Docker daemon is available and responsive.

        Returns
        -------
        bool
            ``True`` if Docker is healthy, ``False`` otherwise.
        """
        try:
            client = self._get_client()
            await asyncio.get_event_loop().run_in_executor(None, client.ping)
            return True
        except (DockerNotAvailableError, Exception) as exc:
            self.logger.warning("Docker health check failed", error=str(exc))
            return False

    # ── Image management ───────────────────────────────────────

    async def ensure_image(self) -> str:
        """
        Ensure the configured Docker image is available locally.

        Respects the configured ``pull_policy``.

        Returns
        -------
        str
            The image name/tag that was used.

        Raises
        ------
        ImageBuildError
            If pulling the image fails.
        """
        client = self._get_client()
        image_name = self.config.image

        if self.config.pull_policy == "never":
            self.logger.debug("Pull policy is 'never', skipping image check")
            return image_name

        def _check_and_pull() -> str:
            try:
                if self.config.pull_policy == "always":
                    self.logger.info("Pulling image (policy=always)", image=image_name)
                    client.images.pull(image_name)
                    return image_name

                # "if_missing" policy
                try:
                    client.images.get(image_name)
                    self.logger.debug("Image found locally", image=image_name)
                except ImageNotFound:
                    self.logger.info(
                        "Image not found locally, pulling", image=image_name
                    )
                    client.images.pull(image_name)

                return image_name
            except Exception as exc:
                raise ImageBuildError(
                    f"Failed to ensure image '{image_name}': {exc}"
                ) from exc

        return await asyncio.get_event_loop().run_in_executor(None, _check_and_pull)

    # ── Workspace management ───────────────────────────────────

    def create_workspace(self, workspace_id: Optional[str] = None) -> Workspace:
        """
        Create a persistent workspace directory for an execution run.

        The workspace persists across multiple :meth:`execute` calls and
        is only removed when :meth:`delete_workspace` is invoked (typically
        when the entire agent run finishes) or when :meth:`close` is called.

        Parameters
        ----------
        workspace_id : str | None
            Explicit workspace ID.  A unique ID is generated when ``None``,
            which supports multiple concurrent runs.

        Returns
        -------
        Workspace
            Handle to the created workspace.

        Raises
        ------
        WorkspaceSetupError
            If workspace preparation fails.
        """
        workspace = Workspace.create(
            base_dir=self.config.workspace_dir or tempfile.gettempdir(),
            workspace_id=workspace_id,
            container_path=self.config.working_dir,
        )
        self._persistent_workspaces[workspace.workspace_id] = workspace
        self.logger.info(
            "Persistent workspace created",
            workspace_id=workspace.workspace_id,
            path=str(workspace.host_path),
        )
        return workspace

    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """
        Get a previously created persistent workspace by ID.

        Parameters
        ----------
        workspace_id : str
            Workspace ID returned by :meth:`create_workspace`.

        Returns
        -------
        Workspace | None
            The workspace handle, or ``None`` if it was never created.
        """
        return self._persistent_workspaces.get(workspace_id)

    def delete_workspace(self, workspace_id: str) -> None:
        """
        Remove a persistent workspace directory.

        Parameters
        ----------
        workspace_id : str
            Workspace ID returned by :meth:`create_workspace`.
        """
        workspace = self._persistent_workspaces.pop(workspace_id, None)
        if workspace is not None:
            workspace.cleanup()
            self.logger.info(
                "Persistent workspace deleted",
                workspace_id=workspace_id,
            )
        else:
            self._cleanup_workspace(workspace_id)

    def _write_files(
        self,
        workspace_path: Path,
        files: Dict[str, str],
    ) -> None:
        """
        Write the given files into an existing workspace directory.

        Parameters
        ----------
        workspace_path : Path
            Host-side path of the workspace directory.
        files : dict
            Mapping of filename -> content to create in the workspace.
        """
        for filename, content in files.items():
            file_path = workspace_path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

    def _prepare_workspace(
        self,
        workspace_id: str,
        files: Optional[Dict[str, str]] = None,
    ) -> Path:
        """
        Prepare a workspace directory with the given files.

        Parameters
        ----------
        workspace_id : str
            Unique identifier for this workspace.
        files : dict | None
            Mapping of filename -> content to create in the workspace.

        Returns
        -------
        Path
            Path to the prepared workspace directory.

        Raises
        ------
        WorkspaceSetupError
            If workspace preparation fails.
        """
        base = self.config.workspace_dir or tempfile.gettempdir()
        workspace_path = Path(base) / "sandbox_workspaces" / workspace_id

        try:
            workspace_path.mkdir(parents=True, exist_ok=True)

            if files:
                self._write_files(workspace_path, files)

            self.logger.debug(
                "Workspace prepared",
                workspace_id=workspace_id,
                path=str(workspace_path),
                file_count=len(files) if files else 0,
            )
            return workspace_path
        except Exception as exc:
            raise WorkspaceSetupError(
                f"Failed to prepare workspace '{workspace_id}': {exc}"
            ) from exc

    def _cleanup_workspace(self, workspace_id: str) -> None:
        """
        Remove a workspace directory.

        Parameters
        ----------
        workspace_id : str
            The workspace identifier to clean up.
        """
        base = self.config.workspace_dir or tempfile.gettempdir()
        workspace_path = Path(base) / "sandbox_workspaces" / workspace_id

        if workspace_path.exists():
            try:
                shutil.rmtree(workspace_path)
                self.logger.debug(
                    "Workspace cleaned up", workspace_id=workspace_id
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to clean workspace",
                    workspace_id=workspace_id,
                    error=str(exc),
                )

    # ── Core execution ─────────────────────────────────────────

    async def execute(
        self,
        command: str,
        *,
        workspace_id: Optional[str] = None,
        files: Optional[Dict[str, str]] = None,
        workspace: Optional[Workspace] = None,
        environment: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        working_dir: Optional[str] = None,
        user: Optional[str] = None,
        image: Optional[str] = None,
    ) -> SandboxExecutionResult:
        """
        Execute a command inside an isolated Docker container.

        Parameters
        ----------
        command : str
            Shell command to execute.
        workspace_id : str | None
            If provided, creates and mounts a per-task workspace.
        files : dict | None
            Files to place in the workspace (requires ``workspace_id``
            or ``workspace``).
        workspace : Workspace | None
            A persistent shared workspace to mount.  Takes precedence
            over ``workspace_id`` and is never removed by this method —
            the caller owns its lifecycle.
        environment : dict | None
            Extra environment variables (merged with config defaults).
        timeout : int | None
            Override the default timeout for this execution.
        working_dir : str | None
            Override the container working directory.
        user : str | None
            Override the container user.
        image : str | None
            Override the Docker image for this execution.

        Returns
        -------
        SandboxExecutionResult
            Captured stdout, stderr, exit code, timing, and metadata.
        """
        effective_timeout = timeout or self.config.timeout
        effective_image = image or self.config.image
        effective_working_dir = working_dir or self.config.working_dir
        effective_user = user or self.config.user
        container: Optional[Container] = None
        container_id: Optional[str] = None
        workspace_path: Optional[Path] = None

        # Merge environment variables
        merged_env = {**self.config.environment}
        if environment:
            merged_env.update(environment)

        start_time = time.monotonic()

        self.logger.info(
            "Executing command in sandbox",
            command=command[:200],
            timeout=effective_timeout,
            workspace_id=workspace_id,
            image=effective_image,
        )

        try:
            # Prepare workspace if requested. Persistent workspaces are
            # mounted in place and owned by the caller; per-call workspaces
            # are created fresh and cleaned up when execution finishes.
            if workspace is not None:
                workspace_path = workspace.host_path
                if files:
                    self._write_files(workspace_path, files)
            elif workspace_id is not None:
                workspace_path = self._prepare_workspace(workspace_id, files)

            # Ensure image is available
            await self.ensure_image()

            # Build container configuration
            container_config: Dict[str, Any] = {
                "image": effective_image,
                "command": ["sh", "-c", command],
                "detach": True,
                "working_dir": effective_working_dir,
                "mem_limit": self.config.memory_limit,
                "cpu_period": self.config.cpu_period,
                "cpu_quota": self.config.cpu_quota,
                "network_disabled": self.config.network_disabled,
                "read_only": self.config.read_only_rootfs,
                "auto_remove": False,  # We handle removal ourselves
                "labels": {
                    "managed-by": "ai-agent-harness",
                    "sandbox": "true",
                },
            }

            if effective_user:
                container_config["user"] = effective_user

            if merged_env:
                container_config["environment"] = merged_env

            if self.config.tmpfs_mounts:
                container_config["tmpfs"] = self.config.tmpfs_mounts

            # Build mounts
            mounts: List[Mount] = []
            if workspace_path:
                mounts.append(
                    Mount(
                        target=self.config.working_dir,
                        source=str(workspace_path),
                        type="bind",
                    )
                )

            for target, source in self.config.additional_volumes.items():
                mounts.append(
                    Mount(
                        target=target,
                        source=source.get("source", ""),
                        type=source.get("type", "bind"),
                    )
                )

            if mounts:
                container_config["mounts"] = mounts

            # Create and start container
            client = self._get_client()
            container = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.containers.create(**container_config),
            )
            container_id = container.short_id
            self._active_containers[container_id] = container

            self.logger.debug(
                "Container created",
                container_id=container_id,
                image=effective_image,
                network_disabled=self.config.network_disabled,
                command=command[:200],
            )

            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: container.start(),
            )

            # Wait for completion with timeout
            try:
                result_data = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: container.wait(timeout=effective_timeout),
                    ),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                # Container timed out — kill it
                self.logger.warning(
                    "Container execution timed out, killing",
                    container_id=container_id,
                    timeout=effective_timeout,
                )
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: container.kill(signal="SIGKILL"),
                )
                # Re-wait after kill
                try:
                    result_data = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: container.wait(timeout=5),
                    )
                except Exception:
                    result_data = {"StatusCode": -1, "Error": "Timeout"}

                elapsed = (time.monotonic() - start_time) * 1000
                stdout = await self._capture_output(container, "stdout")
                stderr = await self._capture_output(container, "stderr")

                return SandboxExecutionResult(
                    success=False,
                    exit_code=-1,
                    stdout=stdout,
                    stderr=stderr or f"Execution timed out after {effective_timeout}s",
                    execution_time=elapsed,
                    container_id=container_id,
                    image=effective_image,
                    timed_out=True,
                    metadata={
                        "network_disabled": self.config.network_disabled,
                        "command": command,
                        "workspace_path": str(workspace_path) if workspace_path else None,
                    },
                )

            exit_code = result_data.get("StatusCode", -1)
            error_msg = result_data.get("Error", "")

            # Capture output
            stdout = await self._capture_output(container, "stdout")
            stderr = await self._capture_output(container, "stderr")
            if error_msg and not stderr:
                stderr = error_msg

            elapsed = (time.monotonic() - start_time) * 1000
            success = exit_code == 0

            result = SandboxExecutionResult(
                success=success,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time=elapsed,
                container_id=container_id,
                image=effective_image,
                timed_out=False,
                metadata={
                    "network_disabled": self.config.network_disabled,
                    "command": command,
                    "workspace_path": str(workspace_path) if workspace_path else None,
                },
            )

            self.logger.info(
                "Command execution completed",
                container_id=container_id,
                exit_code=exit_code,
                execution_time_ms=elapsed,
                stdout_len=len(stdout),
                stderr_len=len(stderr),
                success=success,
                network_disabled=self.config.network_disabled,
            )

            return result

        except DockerNotAvailableError:
            raise
        except SandboxError:
            raise
        except Exception as exc:
            elapsed = (time.monotonic() - start_time) * 1000
            self.logger.error(
                "Unexpected sandbox error",
                error=str(exc),
                error_type=type(exc).__name__,
                execution_time_ms=elapsed,
                container_id=container_id,
                command=command[:200],
                exc_info=True,
            )
            raise SandboxError(
                f"Unexpected error during sandbox execution: {exc}"
            ) from exc

        finally:
            # Cleanup container
            if container is not None:
                await self._cleanup_container(container, container_id)

            # Cleanup per-call workspace only. Persistent workspaces are
            # owned by the caller (e.g. ExecutorAgent) and are removed when
            # the entire agent execution finishes.
            if workspace is None and workspace_id is not None:
                if self.config.cleanup_policy == CleanupPolicy.ALWAYS:
                    self._cleanup_workspace(workspace_id)
                elif (
                    self.config.cleanup_policy == CleanupPolicy.ON_SUCCESS
                    and "result" in dir()
                    and result.success  # type: ignore[possibly-undefined]
                ):
                    self._cleanup_workspace(workspace_id)

    async def _capture_output(self, container: Any, stream: str) -> str:
        """
        Capture stdout or stderr from a container.

        Parameters
        ----------
        container : Container
            The Docker container.
        stream : str
            ``"stdout"`` or ``"stderr"``.

        Returns
        -------
        str
            Decoded output string.
        """
        try:
            output = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: container.logs(stdout=stream == "stdout", stderr=stream == "stderr"),
            )
            if isinstance(output, bytes):
                return output.decode("utf-8", errors="replace")
            return str(output)
        except Exception as exc:
            self.logger.warning(
                f"Failed to capture {stream}",
                error=str(exc),
            )
            return ""

    async def _cleanup_container(
        self,
        container: Any,
        container_id: Optional[str],
    ) -> None:
        """
        Clean up a container based on the configured cleanup policy.

        Parameters
        ----------
        container : Container
            The Docker container to clean up.
        container_id : str | None
            Short ID for logging.
        """
        if self.config.cleanup_policy == CleanupPolicy.NEVER:
            self.logger.debug(
                "Skipping container cleanup (policy=never)",
                container_id=container_id,
            )
            return

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: container.remove(force=True),
            )
            self._active_containers.pop(container_id, None)
            self.logger.debug(
                "Container removed",
                container_id=container_id,
            )
        except Exception as exc:
            self.logger.warning(
                "Failed to remove container",
                container_id=container_id,
                error=str(exc),
            )

    # ── Batch execution ────────────────────────────────────────

    async def execute_batch(
        self,
        commands: List[str],
        *,
        workspace_id: Optional[str] = None,
        files: Optional[Dict[str, str]] = None,
        workspace: Optional[Workspace] = None,
        environment: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        parallel: bool = False,
    ) -> List[SandboxExecutionResult]:
        """
        Execute multiple commands in sequence or parallel.

        Parameters
        ----------
        commands : list[str]
            Commands to execute.
        workspace_id : str | None
            Shared workspace for all commands.
        files : dict | None
            Files to place in the workspace.
        workspace : Workspace | None
            Persistent shared workspace for all commands.
        environment : dict | None
            Extra environment variables.
        timeout : int | None
            Timeout per command.
        parallel : bool
            If ``True``, run commands concurrently.

        Returns
        -------
        list[SandboxExecutionResult]
            Results in the same order as the input commands.
        """
        if not commands:
            return []

        self.logger.info(
            "Executing command batch",
            count=len(commands),
            parallel=parallel,
            workspace_id=workspace_id,
        )

        if parallel:
            tasks = [
                self.execute(
                    cmd,
                    workspace_id=workspace_id,
                    workspace=workspace,
                    files=files if i == 0 else None,
                    environment=environment,
                    timeout=timeout,
                )
                for i, cmd in enumerate(commands)
            ]
            return list(await asyncio.gather(*tasks))

        results: List[SandboxExecutionResult] = []
        for i, cmd in enumerate(commands):
            result = await self.execute(
                cmd,
                workspace_id=workspace_id,
                workspace=workspace,
                files=files if i == 0 else None,
                environment=environment,
                timeout=timeout,
            )
            results.append(result)

            if not result.success:
                self.logger.warning(
                    "Batch command failed, stopping batch",
                    command=cmd,
                    exit_code=result.exit_code,
                    remaining=len(commands) - i - 1,
                )
                # Fill remaining with skipped results
                for _ in range(i + 1, len(commands)):
                    results.append(
                        SandboxExecutionResult(
                            success=False,
                            exit_code=-1,
                            stderr="Skipped due to previous failure",
                        )
                    )
                break

        return results

    # ── Cleanup all ────────────────────────────────────────────

    async def cleanup_all(self) -> int:
        """
        Force-remove all tracked active containers.

        Returns
        -------
        int
            Number of containers removed.
        """
        removed = 0
        for cid, container in list(self._active_containers.items()):
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda c=container: c.remove(force=True),
                )
                removed += 1
            except Exception as exc:
                self.logger.warning(
                    "Failed to remove container during cleanup",
                    container_id=cid,
                    error=str(exc),
                )
        self._active_containers.clear()
        self.logger.info("Active containers cleaned up", removed=removed)
        return removed

    # ── Lifecycle ──────────────────────────────────────────────

    async def close(self) -> None:
        """Shut down the sandbox, removing all active containers and workspaces."""
        await self.cleanup_all()
        for workspace_id in list(self._persistent_workspaces):
            self.delete_workspace(workspace_id)
        self._client = None
        self.logger.info("Docker sandbox closed")

    # ── Context manager ────────────────────────────────────────

    async def __aenter__(self) -> DockerSandbox:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit — ensures cleanup."""
        await self.close()
