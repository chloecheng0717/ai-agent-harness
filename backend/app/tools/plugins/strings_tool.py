"""
Strings Tool

Extracts printable strings from a target file or buffer using the
``strings`` command inside a Docker sandbox.
"""

from typing import Any, Dict, List, Optional

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

# Maximum number of strings to return before truncation kicks in.
MAX_STRINGS: int = 5000


class StringsTool(BaseTool):
    """
    Runs ``strings`` on a target path or buffer.

    Parameters
    ----------
    sandbox : DockerSandbox
        The sandbox to execute ``strings`` in.
    """

    def __init__(self, sandbox: DockerSandbox) -> None:
        definition = ToolDefinition(
            name="strings",
            description=(
                "Extract printable strings from a file or buffer. "
                "Supports configurable minimum string length."
            ),
            version="1.0.0",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to extract strings from",
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Raw content to extract strings from "
                            "(written to a temp file first)"
                        ),
                    },
                    "min_length": {
                        "type": "integer",
                        "description": "Minimum string length (default 4)",
                        "default": 4,
                    },
                },
            },
            returns={
                "type": "object",
                "properties": {
                    "strings": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "count": {"type": "integer"},
                    "truncated": {"type": "boolean"},
                },
            },
            tags=["ctf", "strings", "extract", "binary"],
            permissions=[ToolPermission.READ, ToolPermission.EXECUTE],
            risk_level=RiskLevel.LOW,
            confirmation_required=ConfirmationRequirement.NEVER,
            sandbox_capability=SandboxCapability.CONTAINER,
        )
        super().__init__(definition)
        self.sandbox = sandbox
        self.logger = structlog.get_logger().bind(component="strings_tool")

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        path: Optional[str] = parameters.get("path")
        content: Optional[str] = parameters.get("content")
        min_length: int = parameters.get("min_length", 4)

        if not path and content is None:
            return ToolResult(
                success=False,
                error="Either 'path' or 'content' must be provided",
            )

        files: Optional[Dict[str, str]] = None
        workspace_id: Optional[str] = None
        target: str = ""

        if content is not None:
            workspace_id = "stringstool-temp"
            files = {"_st_target": content}
            target = "/workspace/_st_target"
        else:
            target = path  # type: ignore[assignment]

        cmd = f"strings -n {min_length} {target}"

        try:
            result = await self.sandbox.execute(
                cmd,
                workspace_id=workspace_id,
                files=files,
            )
        except Exception as exc:
            self.logger.error("strings command failed", error=str(exc))
            return ToolResult(
                success=False,
                error=f"Sandbox execution failed: {exc}",
            )

        # Parse stdout into individual strings
        raw_lines = result.stdout.splitlines()
        extracted: List[str] = raw_lines if result.success else []
        truncated = len(extracted) > MAX_STRINGS
        if truncated:
            extracted = extracted[:MAX_STRINGS]

        return ToolResult(
            success=result.success,
            output={
                "strings": extracted,
                "count": len(extracted),
                "truncated": truncated,
                "total_before_truncation": len(raw_lines) if truncated else len(extracted),
            },
            error=result.stderr if not result.success else None,
            execution_time=result.execution_time,
            metadata={
                "container_id": result.container_id,
                "image": result.image,
                "timed_out": result.timed_out,
                "min_length": min_length,
                "command": cmd,
            },
        )
