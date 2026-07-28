"""
File Tool

Identifies file types using the ``file`` command inside a Docker sandbox.
Returns structured type information including MIME type and description.
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


class FileTool(BaseTool):
    """
    Runs ``file`` on a target path or buffer to identify file type.

    Accepts either a ``path`` (already in the workspace) or raw
    ``content`` (base64-encoded) that is written to a temp file first.

    Parameters
    ----------
    sandbox : DockerSandbox
        The sandbox to execute the ``file`` command in.
    """

    def __init__(self, sandbox: DockerSandbox) -> None:
        definition = ToolDefinition(
            name="file",
            description=(
                "Identify the type of a file or buffer. "
                "Supply either 'path' (file path inside the workspace) "
                "or 'content' (raw bytes as a string)."
            ),
            version="1.0.0",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to identify",
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Raw file content to identify "
                            "(written to a temp file first)"
                        ),
                    },
                },
            },
            returns={
                "type": "object",
                "properties": {
                    "mime_type": {"type": "string"},
                    "description": {"type": "string"},
                    "exit_code": {"type": "integer"},
                },
            },
            tags=["ctf", "file", "identify", "magic"],
            permissions=[ToolPermission.READ, ToolPermission.EXECUTE],
            risk_level=RiskLevel.LOW,
            confirmation_required=ConfirmationRequirement.NEVER,
            sandbox_capability=SandboxCapability.CONTAINER,
        )
        super().__init__(definition)
        self.sandbox = sandbox
        self.logger = structlog.get_logger().bind(component="file_tool")

    # -- helpers ---------------------------------------------------

    def _build_command(self, path: Optional[str], content: Optional[str]) -> str:
        if path:
            return f"file --mime-type -b {path}"
        if content is not None:
            return "file --mime-type -b /tmp/_ft_target"
        raise ValueError("Either 'path' or 'content' must be provided")

    # -- execute ---------------------------------------------------

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        path: Optional[str] = parameters.get("path")
        content: Optional[str] = parameters.get("content")

        if not path and content is None:
            return ToolResult(
                success=False,
                error="Either 'path' or 'content' must be provided",
            )

        files: Optional[Dict[str, str]] = None
        workspace_id: Optional[str] = None
        cmd: str = ""

        if content is not None:
            workspace_id = "filetool-temp"
            files = {"_ft_target": content}
            cmd = "file --mime-type -b /workspace/_ft_target"
        else:
            cmd = f"file --mime-type -b {path}"

        try:
            result = await self.sandbox.execute(
                cmd,
                workspace_id=workspace_id,
                files=files,
            )
        except Exception as exc:
            self.logger.error("file command failed", error=str(exc))
            return ToolResult(
                success=False,
                error=f"Sandbox execution failed: {exc}",
            )

        output = result.stdout.strip()
        exit_code = result.exit_code

        # Parse MIME type from "file" output
        # Typical output: "text/plain" or "application/x-executable"
        mime_type = output if output else "application/octet-stream"

        description = self._describe_mime(mime_type)

        return ToolResult(
            success=result.success,
            output={
                "mime_type": mime_type,
                "description": description,
                "exit_code": exit_code,
                "raw_output": output,
            },
            error=result.stderr if not result.success else None,
            execution_time=result.execution_time,
            metadata={
                "container_id": result.container_id,
                "image": result.image,
                "timed_out": result.timed_out,
                "command": cmd,
            },
        )

    # -- static helpers --------------------------------------------

    @staticmethod
    def _describe_mime(mime: str) -> str:
        """Return a human-readable description for common MIME types."""
        descriptions = {
            "text/plain": "Plain text file",
            "text/html": "HTML document",
            "text/css": "CSS stylesheet",
            "text/x-python": "Python script",
            "text/x-c": "C source file",
            "application/json": "JSON data",
            "application/x-executable": "Executable binary",
            "application/x-sharedlib": "Shared library",
            "application/zip": "ZIP archive",
            "application/x-tar": "TAR archive",
            "application/gzip": "Gzip compressed archive",
            "image/png": "PNG image",
            "image/jpeg": "JPEG image",
            "image/gif": "GIF image",
            "image/bmp": "BMP image",
            "audio/mpeg": "MPEG audio",
            "video/mp4": "MP4 video",
            "application/pdf": "PDF document",
            "application/x-elf": "ELF binary",
        }
        return descriptions.get(mime, f"File of type {mime}")
