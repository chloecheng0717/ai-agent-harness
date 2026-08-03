"""
Python Tool

General-purpose Python execution inside a Docker sandbox.
This is the compute/decode tool the agent reaches for when no
specific tool fits.

The tool always returns a structured result (never a bare ``null``
output): ``stdout``, ``stderr``, ``exit_code``, ``timed_out``,
``exception``, ``execution_time`` and the list of files the run
produced/changed in the shared workspace (``generated_files``).
"""

import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import structlog

from app.models.tool import (
    ConfirmationRequirement,
    RiskLevel,
    SandboxCapability,
    ToolDefinition,
    ToolPermission,
    ToolResult,
)
from app.sandbox.docker_sandbox import DockerSandbox
from app.tools.base import BaseTool

if TYPE_CHECKING:
    from app.sandbox.workspace import Workspace

# Absolute ceiling for this tool (seconds) regardless of caller request.
_MAX_TIMEOUT: int = 120

#: The scaffold file name used to carry inline ``code`` into the container.
#: Excluded from ``generated_files`` because it is tool scaffolding, not
#: output the agent's program produced.
_SCRIPT_FILENAME = "_pt_script.py"


def _snapshot_workspace(host_path: Path) -> dict[str, tuple[int, int]]:
    """
    Fingerprint every file in *host_path* as ``relpath -> (size, mtime_ns)``.

    Used to compute ``generated_files``: files that are new or whose
    size/mtime changed while the program ran.
    """
    snapshot: dict[str, tuple[int, int]] = {}
    if not host_path.is_dir():
        return snapshot
    for path in host_path.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        snapshot[path.relative_to(host_path).as_posix()] = (
            stat.st_size,
            stat.st_mtime_ns,
        )
    return snapshot


class PythonTool(BaseTool):
    """
    Executes an arbitrary Python snippet/script inside a sandboxed
    container.

    Parameters
    ----------
    sandbox : DockerSandbox
        The sandbox to execute Python in.
    """

    def __init__(self, sandbox: DockerSandbox) -> None:
        definition = ToolDefinition(
            name="python",
            description=(
                "Execute a Python snippet or script inside an isolated "
                "Docker sandbox. Captures stdout, stderr, exit code, any "
                "exception, and the files produced in the workspace."
            ),
            version="1.0.0",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code to execute (mutually exclusive "
                            "with 'script_path')"
                        ),
                    },
                    "script_path": {
                        "type": "string",
                        "description": (
                            "Path to a Python script already in the "
                            "workspace to execute"
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            "Execution timeout in seconds "
                            "(max 120, default from sandbox config)"
                        ),
                    },
                },
            },
            returns={
                "type": "object",
                "properties": {
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                    "exit_code": {"type": "integer"},
                    "timed_out": {"type": "boolean"},
                    "exception": {"type": "string"},
                    "execution_time": {"type": "number"},
                    "generated_files": {"type": "array", "items": {"type": "string"}},
                },
            },
            tags=["ctf", "python", "execute", "decode", "compute"],
            permissions=[ToolPermission.EXECUTE],
            risk_level=RiskLevel.HIGH,
            confirmation_required=ConfirmationRequirement.NEVER,
            sandbox_capability=SandboxCapability.CONTAINER,
        )
        super().__init__(definition)
        self.sandbox = sandbox
        self.logger = structlog.get_logger().bind(component="python_tool")

    async def execute(
        self,
        parameters: Dict[str, Any],
        workspace: Optional["Workspace"] = None,
    ) -> ToolResult:
        code: Optional[str] = parameters.get("code")
        script_path: Optional[str] = parameters.get("script_path")
        timeout: Optional[int] = parameters.get("timeout")

        if not code and not script_path:
            return self._structured_result(
                success=False,
                error="Either 'code' or 'script_path' must be provided",
            )

        if code and script_path:
            return self._structured_result(
                success=False,
                error="Provide either 'code' or 'script_path', not both",
            )

        # Clamp timeout
        effective_timeout: Optional[int] = None
        if timeout is not None:
            effective_timeout = min(max(1, timeout), _MAX_TIMEOUT)

        files: Optional[Dict[str, str]] = None
        workspace_id: Optional[str] = None
        cmd: str = ""

        if code is not None:
            files = {_SCRIPT_FILENAME: code}
            # -u keeps stdout/stderr unbuffered so partial output is not lost
            # when a step times out or crashes mid-print.
            cmd = f"python -u /workspace/{_SCRIPT_FILENAME}"
            if workspace is None:
                workspace_id = "pythontool-temp"
        else:
            cmd = f"python -u {script_path}"

        # Fingerprint the shared workspace before execution so the files the
        # program actually produced can be reported.  Only meaningful when a
        # persistent workspace is attached (per-call temp workspaces are
        # removed by the sandbox before this method returns).
        before = _snapshot_workspace(workspace.host_path) if workspace else {}

        start = time.monotonic()
        try:
            result = await self.sandbox.execute(
                cmd,
                workspace_id=workspace_id,
                workspace=workspace,
                files=files,
                timeout=effective_timeout,
            )
        except Exception as exc:
            self.logger.error(
                "python execution failed",
                error=str(exc),
                error_type=type(exc).__name__,
                command=cmd,
                workspace_id=workspace.workspace_id if workspace else None,
                exc_info=True,
            )
            return self._structured_result(
                success=False,
                error=f"Sandbox execution failed: {exc}",
                exception=traceback.format_exc(),
                workspace=workspace,
                command=cmd,
            )

        after = _snapshot_workspace(workspace.host_path) if workspace else {}
        generated_files = sorted(
            rel for rel in after if before.get(rel) != after[rel]
        )
        if workspace:
            generated_files = [
                rel for rel in generated_files if rel != _SCRIPT_FILENAME
            ]

        return self._structured_result(
            success=result.success,
            error=result.stderr if not result.success else None,
            result=result,
            execution_time=result.execution_time,
            metadata={
                "container_id": result.container_id,
                "image": result.image,
                "timed_out": result.timed_out,
                "timeout_requested": effective_timeout,
                "command": cmd,
                "network_disabled": result.metadata.get("network_disabled"),
            },
            workspace=workspace,
            generated_files=generated_files,
            command=cmd,
        )

    @staticmethod
    def _structured_result(
        *,
        success: bool,
        error: Optional[str],
        result: Any = None,
        exception: Optional[str] = None,
        execution_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        workspace: Optional["Workspace"] = None,
        generated_files: Optional[list[str]] = None,
        command: str = "",
    ) -> ToolResult:
        """
        Build a ToolResult whose ``output`` is always a structured dict.

        Even on failure the output carries every captured field so the
        orchestrator never records ``output: null`` for a python step.
        """
        if result is not None:
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.exit_code
            timed_out = result.timed_out
        else:
            stdout = ""
            stderr = error or ""
            exit_code = None
            timed_out = False

        return ToolResult(
            success=success,
            output={
                "stdout": stdout or "",
                "stderr": stderr or "",
                "exit_code": exit_code,
                "timed_out": timed_out,
                "exception": exception or "",
                "execution_time": round((execution_time or 0.0), 3),
                "generated_files": generated_files or [],
                "workspace_id": workspace.workspace_id if workspace else None,
            },
            error=error,
            execution_time=execution_time or 0.0,
            metadata=metadata or {},
        )
