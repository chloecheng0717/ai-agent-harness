"""
Encoding Tool

Encode / decode between base64, hex, and URL encodings.
Runs the actual work inside a Docker sandbox for safety.
"""

import base64
import re
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

# Supported encoding families
_VALID_ENCODINGS = {"base64", "hex", "url"}

# Regex patterns for auto-detecting encoding direction
# Padded base64 (its ``=`` suffix) is the least ambiguous signal.
_B64_RE = re.compile(
    r"^[A-Za-z0-9+/]{2,}={1,2}$", re.MULTILINE
)
_HEX_RE = re.compile(
    r"^[0-9a-fA-F]+$", re.MULTILINE
)
_URL_RE = re.compile(
    r"%[0-9a-fA-F]{2}", re.MULTILINE
)

#: Standard base64 alphabet (no ``=`` padding).
_BASE64_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
)


class EncodingTool(BaseTool):
    """
    Encode / decode data using base64, hex, or URL encoding.

    Can auto-detect whether input is encoded (decode) or plain text
    (encode), or the caller can force a direction.

    Parameters
    ----------
    sandbox : DockerSandbox
        The sandbox to run encoding operations in.
    """

    def __init__(self, sandbox: DockerSandbox) -> None:
        definition = ToolDefinition(
            name="encoding",
            description=(
                "Encode or decode data using base64, hex, or URL encoding. "
                "Supports auto-detection of direction."
            ),
            version="1.0.0",
            parameters={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Data to encode or decode",
                    },
                    "encoding": {
                        "type": "string",
                        "enum": ["base64", "hex", "url"],
                        "description": "Encoding scheme to use",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["encode", "decode", "auto"],
                        "description": (
                            "Direction: 'encode', 'decode', or "
                            "'auto' (default)"
                        ),
                        "default": "auto",
                    },
                },
                "required": ["input", "encoding"],
            },
            returns={
                "type": "object",
                "properties": {
                    "output": {"type": "string"},
                    "mode_used": {"type": "string"},
                    "valid_utf8": {"type": "boolean"},
                    "printable": {"type": "boolean"},
                },
            },
            tags=["ctf", "encoding", "base64", "hex", "url", "decode"],
            permissions=[ToolPermission.EXECUTE],
            risk_level=RiskLevel.LOW,
            confirmation_required=ConfirmationRequirement.NEVER,
            sandbox_capability=SandboxCapability.CONTAINER,
        )
        super().__init__(definition)
        self.sandbox = sandbox
        self.logger = structlog.get_logger().bind(component="encoding_tool")

    # -- auto-detection helpers ------------------------------------

    @staticmethod
    def _looks_like_paddingless_base64(data: str) -> bool:
        """Detect unpadded base64 that decodes to *meaningful* bytes.

        Conservative: must round-trip exactly after re-encoding and must
        contain at least one digit.  Plain mixed-case words such as
        "HelloWorldSomething" happen to be valid base64 alphabet, so the
        digit requirement keeps them out while real flag payloads (which
        almost always contain digits in their encoding) are caught.
        """
        token = data.strip()
        if len(token) < 8 or len(token) % 4 == 1:
            return False
        if "=" in token or not all(c in _BASE64_CHARS for c in token):
            return False
        if not any(c.isalpha() for c in token):
            return False
        if not any(c.isdigit() for c in token):
            return False
        padded = token + "=" * ((4 - len(token) % 4) % 4)
        try:
            decoded = base64.b64decode(padded, validate=True)
        except Exception:
            return False
        return base64.b64encode(decoded).decode().rstrip("=") == token

    @staticmethod
    def _detect_mode(data: str, encoding: str) -> str:
        """Heuristic to decide whether data looks encoded or plain."""
        stripped = data.strip()
        if not stripped:
            return "encode"

        if encoding == "base64":
            # Padding → likely encoded
            if _B64_RE.match(stripped):
                return "decode"
            # Contains base64-specific chars (+ or /) → likely encoded
            if "+" in stripped or "/" in stripped:
                return "decode"
            # Unpadded base64 → likely encoded
            if EncodingTool._looks_like_paddingless_base64(stripped):
                return "decode"
            return "encode"

        if encoding == "hex":
            clean = stripped.replace(" ", "")
            if _HEX_RE.match(clean) and len(clean) % 2 == 0:
                return "decode"
            return "encode"

        if encoding == "url":
            if _URL_RE.search(stripped):
                return "decode"
            return "encode"

        return "encode"

    @staticmethod
    def _is_printable(text: str) -> bool:
        """Check whether decoded text is printable ASCII / UTF-8."""
        try:
            if not isinstance(text, str):
                return False
            if len(text) == 0:
                return True
            printable_count = sum(
                1 for c in text if c.isprintable() or c in "\n\r\t"
            )
            return printable_count / len(text) > 0.9
        except Exception:
            return False

    @staticmethod
    def _is_valid_utf8(text: str) -> bool:
        """Check whether the output is valid UTF-8."""
        try:
            text.encode("utf-8").decode("utf-8")
            return True
        except (UnicodeDecodeError, UnicodeEncodeError):
            return False

    # -- Python snippet builders -----------------------------------

    @staticmethod
    def _build_python(encoding: str, mode: str, data: str) -> str:
        """Build a Python snippet that encodes or decodes *data*."""
        import base64
        import urllib.parse

        # We build a self-contained snippet; data is injected via
        # a triple-quoted string literal.
        safe_data = data.replace("\\", "\\\\").replace("'", "\\'")
        snippet = f"import sys, base64, urllib.parse\ndata = '{safe_data}'\n"

        if encoding == "base64":
            if mode == "decode":
                snippet += (
                    "raw = base64.b64decode(data)\n"
                    "sys.stdout.buffer.write(raw)\n"
                )
            else:
                snippet += (
                    "sys.stdout.write(base64.b64encode(data.encode()).decode())\n"
                )
        elif encoding == "hex":
            if mode == "decode":
                snippet += (
                    "sys.stdout.buffer.write(bytes.fromhex(data))\n"
                )
            else:
                snippet += (
                    "sys.stdout.write(data.encode().hex())\n"
                )
        elif encoding == "url":
            if mode == "decode":
                snippet += (
                    "sys.stdout.write(urllib.parse.unquote(data))\n"
                )
            else:
                snippet += (
                    "sys.stdout.write(urllib.parse.quote(data, safe=''))\n"
                )

        return snippet

    # -- execute ---------------------------------------------------

    async def execute(
        self,
        parameters: Dict[str, Any],
        workspace: Optional["Workspace"] = None,
    ) -> ToolResult:
        data: str = parameters.get("input", "")
        encoding: str = parameters.get("encoding", "")
        mode_param: str = parameters.get("mode", "auto")

        if not data:
            return ToolResult(success=False, error="No input data provided")

        if encoding not in _VALID_ENCODINGS:
            return ToolResult(
                success=False,
                error=f"Unsupported encoding '{encoding}'. "
                f"Choose from: {', '.join(sorted(_VALID_ENCODINGS))}",
            )

        if mode_param not in ("encode", "decode", "auto"):
            return ToolResult(
                success=False,
                error=f"Invalid mode '{mode_param}'. Use 'encode', 'decode', or 'auto'.",
            )

        # Resolve mode
        if mode_param == "auto":
            resolved_mode = self._detect_mode(data, encoding)
        else:
            resolved_mode = mode_param

        python_snippet = self._build_python(encoding, resolved_mode, data)

        try:
            result = await self.sandbox.execute(
                f"python -c '{python_snippet}'",
                workspace=workspace,
            )
        except Exception as exc:
            self.logger.error("encoding execution failed", error=str(exc))
            return ToolResult(
                success=False,
                error=f"Sandbox execution failed: {exc}",
            )

        output = result.stdout
        valid_utf8 = self._is_valid_utf8(output)
        printable = self._is_printable(output) if valid_utf8 else False

        return ToolResult(
            success=result.success,
            output={
                "output": output,
                "mode_used": resolved_mode,
                "valid_utf8": valid_utf8,
                "printable": printable,
            },
            error=result.stderr if not result.success else None,
            execution_time=result.execution_time,
            metadata={
                "container_id": result.container_id,
                "image": result.image,
                "timed_out": result.timed_out,
                "encoding": encoding,
                "requested_mode": mode_param,
                "resolved_mode": resolved_mode,
            },
        )
