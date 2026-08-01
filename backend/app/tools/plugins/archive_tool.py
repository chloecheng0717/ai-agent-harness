"""
Archive Tool

Extracts ZIP, TAR, TAR.GZ, GZIP, and 7Z archives entirely inside the
Docker sandbox.  Extraction runs through a self-contained helper script
(``app.tools.archive_helpers``) that is shipped into the shared workspace,
so extracted files persist for later tool calls in the same run.

Returns structured results with the extracted file names, the extraction
destination, success/failure status, and error messages.  Corrupted and
password-protected archives are reported gracefully instead of crashing.
"""

import inspect
import json
from typing import TYPE_CHECKING, Any, Optional

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
from app.tools import archive_helpers
from app.tools.base import BaseTool

if TYPE_CHECKING:
    from app.sandbox.workspace import Workspace

_SCRIPT_NAME = "_archive_extract.py"
_PARAMS_NAME = "_archive_params.json"


class ArchiveTool(BaseTool):
    """
    Extracts an archive inside a Docker sandbox.

    Accepts either an ``archive_path`` (already in the shared workspace) or
    base64-encoded ``content`` that is written to ``archive_path`` first.

    Parameters
    ----------
    sandbox : DockerSandbox
        The sandbox to extract archives in.
    """

    def __init__(self, sandbox: DockerSandbox) -> None:
        definition = ToolDefinition(
            name="archive",
            description=(
                "Extract ZIP, TAR, TAR.GZ, GZIP, or 7Z archives inside the "
                "shared sandbox workspace. Supply either 'archive_path' (a "
                "file already in the workspace) or 'content' (base64-encoded "
                "archive bytes). Extracted files persist for later tool calls."
            ),
            version="1.0.0",
            parameters={
                "type": "object",
                "properties": {
                    "archive_path": {
                        "type": "string",
                        "description": (
                            "Path to the archive inside the workspace. "
                            "When 'content' is supplied this is the filename "
                            "the archive bytes are written to."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Base64-encoded archive bytes, written to "
                            "'archive_path' before extraction."
                        ),
                    },
                    "extract_dir": {
                        "type": "string",
                        "description": (
                            "Destination directory, relative to the workspace "
                            "root or an absolute container path. "
                            "Defaults to the archive's directory."
                        ),
                    },
                    "format": {
                        "type": "string",
                        "enum": list(archive_helpers.SUPPORTED_FORMATS),
                        "description": (
                            "Force the archive format. Auto-detected from the "
                            "file extension when omitted."
                        ),
                    },
                    "password": {
                        "type": "string",
                        "description": (
                            "Password for encrypted ZIP or 7Z archives."
                        ),
                    },
                },
                "required": ["archive_path"],
            },
            returns={
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extracted file names",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Extraction destination",
                    },
                    "success": {"type": "boolean"},
                    "archive_type": {"type": ["string", "null"]},
                    "file_count": {"type": "integer"},
                },
            },
            tags=["archive", "extract", "zip", "tar", "gzip", "7z", "ctf"],
            permissions=[
                ToolPermission.READ,
                ToolPermission.WRITE,
                ToolPermission.EXECUTE,
            ],
            risk_level=RiskLevel.MEDIUM,
            confirmation_required=ConfirmationRequirement.NEVER,
            sandbox_capability=SandboxCapability.CONTAINER,
        )
        super().__init__(definition)
        self.sandbox = sandbox
        self.logger = structlog.get_logger().bind(component="archive_tool")

    # -- helpers ---------------------------------------------------

    @staticmethod
    def _to_container_path(value: str, container_root: str) -> str:
        """Normalize a relative path to be container-absolute."""
        value = value.strip()
        if value.startswith("/"):
            return value
        return f"{container_root.rstrip('/')}/{value}"

    @staticmethod
    def _helper_source() -> str:
        """Return the source of the sandbox-side extraction helper."""
        return inspect.getsource(archive_helpers)

    def _build_params(
        self,
        parameters: dict[str, Any],
        container_root: str,
    ) -> dict[str, Any]:
        archive_path: str | None = parameters.get("archive_path")
        content: str | None = parameters.get("content")

        if not archive_path and content is None:
            raise ValueError("Either 'archive_path' or 'content' must be provided")

        if content is not None and not archive_path:
            archive_path = "_archive_content.bin"

        assert archive_path is not None
        params: dict[str, Any] = {
            "archive_path": self._to_container_path(archive_path, container_root),
        }
        if content is not None:
            params["content_b64"] = content
        extract_dir = parameters.get("extract_dir")
        if extract_dir:
            params["extract_dir"] = self._to_container_path(extract_dir, container_root)
        if parameters.get("format"):
            params["format"] = str(parameters["format"]).strip().lower()
        if parameters.get("password"):
            params["password"] = parameters["password"]
        return params

    # -- execute ---------------------------------------------------

    async def execute(
        self,
        parameters: dict[str, Any],
        workspace: Optional["Workspace"] = None,
    ) -> ToolResult:
        archive_path: str | None = parameters.get("archive_path")
        content: str | None = parameters.get("content")
        fmt: str | None = parameters.get("format")

        if not archive_path and content is None:
            return ToolResult(
                success=False,
                error="Either 'archive_path' or 'content' must be provided",
            )

        if fmt is not None and str(fmt).strip().lower() not in archive_helpers.SUPPORTED_FORMATS:
            return ToolResult(
                success=False,
                error=(
                    f"Unsupported archive format '{fmt}'. "
                    f"Supported: {', '.join(archive_helpers.SUPPORTED_FORMATS)}"
                ),
            )

        container_root = workspace.container_path if workspace else "/workspace"

        try:
            params = self._build_params(parameters, container_root)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

        files = {
            _SCRIPT_NAME: self._helper_source(),
            _PARAMS_NAME: json.dumps(params),
        }
        cmd = f"python {container_root}/{_SCRIPT_NAME}"
        workspace_id: str | None = None
        if workspace is None:
            workspace_id = "archivetool-temp"

        try:
            kwargs: dict[str, Any] = {"files": files}
            if workspace is not None:
                kwargs["workspace"] = workspace
            else:
                kwargs["workspace_id"] = workspace_id
            result = await self.sandbox.execute(cmd, **kwargs)
        except Exception as exc:
            self.logger.error("archive extraction failed", error=str(exc))
            return ToolResult(
                success=False,
                error=f"Sandbox execution failed: {exc}",
            )

        return self._build_result(result, cmd)

    def _build_result(self, result: Any, cmd: str) -> ToolResult:
        """Parse the helper's JSON output into a structured ToolResult."""
        metadata = {
            "container_id": result.container_id,
            "image": result.image,
            "timed_out": result.timed_out,
            "command": cmd,
        }

        payload: dict[str, Any]
        try:
            payload = json.loads(result.stdout.strip() or "{}")
        except json.JSONDecodeError:
            return ToolResult(
                success=False,
                output={
                    "files": [],
                    "destination": "",
                    "success": False,
                    "archive_type": None,
                    "file_count": 0,
                },
                error=result.stderr or "Archive extraction failed",
                execution_time=result.execution_time,
                metadata=metadata,
            )

        success = bool(payload.get("success"))
        files = payload.get("files", [])
        destination = payload.get("destination", "")
        archive_type = payload.get("archive_type")
        file_count = payload.get("file_count", len(files))
        error = payload.get("error")

        metadata["archive_type"] = archive_type
        if not success and not result.success and not error:
            error = result.stderr

        return ToolResult(
            success=success,
            output={
                "files": files,
                "destination": destination,
                "success": success,
                "archive_type": archive_type,
                "file_count": file_count,
            },
            error=error if not success else None,
            execution_time=result.execution_time,
            metadata=metadata,
        )
