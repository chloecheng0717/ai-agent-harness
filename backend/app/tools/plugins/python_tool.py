"""
Python Tool

General-purpose Python execution inside a Docker sandbox.
This is the compute/decode tool the agent reaches for when no
specific tool fits.
"""

from typing import Any, Dict, Optional

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

# Absolute ceiling for this tool (seconds) regardless of caller request.
_MAX_TIMEOUT: int = 120


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
                "Docker sandbox. Captures stdout, stderr, and exit code."
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

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        code: Optional[str] = parameters.get("code")
        script_path: Optional[str] = parameters.get("script_path")
        timeout: Optional[int] = parameters.get("timeout")

        if not code and not script_path:
            return ToolResult(
                success=False,
                error="Either 'code' or 'script_path' must be provided",
            )

        if code and script_path:
            return ToolResult(
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
            workspace_id = "pythontool-temp"
            files = {"_pt_script.py": code}
            cmd = "python /workspace/_pt_script.py"
        else:
            cmd = f"python {script_path}"

        try:
            result = await self.sandbox.execute(
                cmd,
                workspace_id=workspace_id,
                files=files,
                timeout=effective_timeout,
            )
        except Exception as exc:
            self.logger.error("python execution failed", error=str(exc))
            return ToolResult(
                success=False,
                error=f"Sandbox execution failed: {exc}",
            )

        return ToolResult(
            success=result.success,
            output={
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
            },
            error=result.stderr if not result.success else None,
            execution_time=result.execution_time,
            metadata={
                "container_id": result.container_id,
                "image": result.image,
                "timed_out": result.timed_out,
                "timeout_requested": effective_timeout,
                "command": cmd,
            },
        )
