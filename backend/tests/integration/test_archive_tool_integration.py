"""
Integration Tests for the Archive Tool

Verifies the full pipeline: tool -> sandbox -> ToolRegistry -> ExecutorAgent,
using the persistent shared workspace so extracted files are available to
later tool calls in the same run.

Docker interactions are simulated (same pattern as the existing sandbox
integration tests): the fake container "executes" tool commands against the
host-side workspace directory that a real container would mount at
``/workspace``, so archive extraction is exercised for real on the host.
"""

import base64
import gzip
import io
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from app.agents.specialized.executor import ExecutorAgent
from app.models.state import AgentState, PlanStep
from app.sandbox.docker_sandbox import (
    CleanupPolicy,
    DockerSandbox,
    SandboxConfig,
)
from app.tools.plugins.archive_tool import ArchiveTool
from app.tools.plugins.python_tool import PythonTool
from app.tools.plugins.strings_tool import StringsTool
from app.tools.registry import ToolRegistry


def _zip_bytes(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _gzip_bytes(content: bytes, name: str) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename=name, mode="wb", fileobj=buffer) as fh:
        fh.write(content)
    return buffer.getvalue()


# ── Simulated container ──────────────────────────────────────────


class _SimulatedContainer:
    """Simulates a Docker container against the host workspace directory.

    The real container would run tool commands with the shared workspace
    mounted at ``/workspace``.  This fake remaps ``/workspace`` paths to the
    workspace's host directory and executes Python scripts / reads files
    locally, so cross-tool persistence is exercised for real.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.short_id = "arc00001"
        self._stdout = ""
        self._stderr = ""
        self._workspace_source: str | None = None
        for mount in config.get("mounts", []):
            if mount.get("Target") == "/workspace":
                self._workspace_source = mount.get("Source")
                break

    def start(self) -> None:
        pass

    def remove(self, force: bool = False) -> None:
        pass

    def wait(self, timeout: int | None = None) -> dict[str, Any]:
        command: str = self.config["command"][2]
        if self._workspace_source:
            command = command.replace(
                "/workspace", self._workspace_source.replace("\\", "/")
            )

        if "_archive_extract.py" in command:
            self._prepare_archive_params()
            self._run_shell(command)
            self._remap_archive_result()
        elif command.startswith("python "):
            self._run_python(command)
        elif command.startswith("strings "):
            self._run_strings(command)
        else:
            self._run_shell(command)

        return {"StatusCode": 0, "Error": ""}

    def logs(self, stdout: bool = True, stderr: bool = False) -> bytes:
        if stdout:
            return self._stdout.encode("utf-8", errors="replace")
        if stderr:
            return self._stderr.encode("utf-8", errors="replace")
        return b""

    # -- command runners -----------------------------------------

    def _run_shell(self, command: str) -> None:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=self._workspace_source,
        )
        self._stdout = proc.stdout
        self._stderr = proc.stderr

    def _run_python(self, command: str) -> None:
        """Execute a python command, remapping /workspace inside scripts."""
        rest = command[len("python "):].strip()
        if not rest.startswith("-c"):
            script_path = rest.strip().strip('"\'')
            if self._workspace_source:
                script = Path(script_path)
                if script.exists():
                    original = script.read_text(encoding="utf-8")
                    rewritten = original.replace(
                        "/workspace", self._workspace_source.replace("\\", "/")
                    )
                    if rewritten != original:
                        script.write_text(rewritten, encoding="utf-8")
                        try:
                            self._run_shell(command)
                        finally:
                            script.write_text(original, encoding="utf-8")
                        return
        self._run_shell(command)

    def _run_strings(self, command: str) -> None:
        parts = command.split()
        min_length = 4
        for i, part in enumerate(parts):
            if part == "-n" and i + 1 < len(parts):
                try:
                    min_length = int(parts[i + 1])
                except ValueError:
                    pass
        target = self._remap_path(parts[-1])
        data = Path(target).read_bytes()
        sequences = re.findall(rb"[ -~]{%d,}" % min_length, data)
        self._stdout = "\n".join(
            s.decode("ascii", errors="replace") for s in sequences
        )
        if self._stdout:
            self._stdout += "\n"

    def _prepare_archive_params(self) -> None:
        """Rewrite container paths in the helper's params to host paths."""
        workspace = self._workspace_source
        if workspace is None:
            return
        params_file = Path(workspace) / "_archive_params.json"
        if params_file.exists():
            text = params_file.read_text(encoding="utf-8")
            text = text.replace(
                "/workspace", workspace.replace("\\", "/")
            )
            params_file.write_text(text, encoding="utf-8")

    def _remap_archive_result(self) -> None:
        """Map host paths in the helper's JSON result back to container paths.

        The remap runs on the parsed JSON because ``json.dumps`` escapes
        backslashes (``C:\\Users``) in the raw stdout, so text replacement
        on the raw string cannot match.
        """
        if not self._workspace_source:
            return
        try:
            data = json.loads(self._stdout)
        except (json.JSONDecodeError, ValueError):
            return
        workspace = self._workspace_source
        workspace_slash = workspace.replace("\\", "/")
        for key in ("destination", "archive_path"):
            value = data.get(key)
            if isinstance(value, str):
                data[key] = (
                    value.replace(workspace, "/workspace")
                    .replace(workspace_slash, "/workspace")
                    .replace("\\", "/")
                )
        self._stdout = json.dumps(data)

    def _remap_path(self, path: str) -> str:
        if self._workspace_source:
            return path.replace(
                "/workspace", self._workspace_source.replace("\\", "/")
            )
        return path


@contextmanager
def _patched_client(sandbox: DockerSandbox) -> Iterator[MagicMock]:
    """Patch the sandbox's client so containers are simulated."""
    client = MagicMock()
    client.ping.return_value = True
    client.images.get.return_value = MagicMock()
    client.containers.create.side_effect = lambda **kwargs: _SimulatedContainer(kwargs)
    with patch.object(sandbox, "_get_client", return_value=client):
        yield client


def _make_executor(workspace_dir: str, *tools: Any) -> tuple[ExecutorAgent, ToolRegistry]:
    config = SandboxConfig(
        workspace_dir=workspace_dir,
        pull_policy="never",
        cleanup_policy=CleanupPolicy.ALWAYS,
    )
    sandbox = DockerSandbox(config)
    registry = ToolRegistry(sandbox=sandbox)
    for tool in tools:
        registry.register_instance(tool)
    executor = ExecutorAgent(tool_registry=registry, sandbox=sandbox)
    return executor, registry


# ═══════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════


class TestArchiveToolIntegration:
    """End-to-end archive extraction through the executor and shared workspace."""

    @pytest.mark.asyncio
    async def test_zip_extraction_end_to_end(self) -> None:
        workspace_dir = tempfile.mkdtemp()
        try:
            config = SandboxConfig(
                workspace_dir=workspace_dir,
                pull_policy="never",
                cleanup_policy=CleanupPolicy.ALWAYS,
            )
            sandbox = DockerSandbox(config)
            registry = ToolRegistry(sandbox=sandbox)
            registry.register_instance(PythonTool(sandbox))
            registry.register_instance(ArchiveTool(sandbox))
            registry.register_instance(StringsTool(sandbox))
            executor = ExecutorAgent(tool_registry=registry, sandbox=sandbox)

            archive = _zip_bytes({"flag.txt": "FLAG{archive_persisted}"})
            encoded = base64.b64encode(archive).decode()

            state = AgentState()
            state.plan = [
                PlanStep(
                    index=0,
                    description="Write the archive into the workspace",
                    tool_name="python",
                    tool_parameters={
                        "code": (
                            "import base64\n"
                            f"with open('/workspace/challenge.zip', 'wb') as f:\n"
                            f"    f.write(base64.b64decode('{encoded}'))\n"
                            "print('written')\n"
                        ),
                    },
                ),
                PlanStep(
                    index=1,
                    description="Extract the archive",
                    tool_name="archive",
                    tool_parameters={
                        "archive_path": "/workspace/challenge.zip",
                        "extract_dir": "/workspace/extracted",
                    },
                    depends_on=[0],
                ),
                PlanStep(
                    index=2,
                    description="Read the extracted flag file",
                    tool_name="strings",
                    tool_parameters={"path": "/workspace/extracted/flag.txt"},
                    depends_on=[1],
                ),
            ]

            with _patched_client(sandbox):
                result = await executor.process({"state": state})

            assert result["success"] is True, result["errors"]
            assert len(result["results"]) == 3

            extract_step = result["results"][1]
            assert extract_step["tool_name"] == "archive"
            assert extract_step["success"] is True
            assert extract_step["output"]["archive_type"] == "zip"
            assert extract_step["output"]["file_count"] == 1
            assert extract_step["output"]["files"] == ["flag.txt"]
            assert extract_step["output"]["destination"] == "/workspace/extracted"

            strings_step = result["results"][2]
            assert "FLAG{archive_persisted}" in strings_step["output"]["strings"]

            # The shared workspace is removed once the run finishes.
            workspaces_root = Path(workspace_dir) / "sandbox_workspaces"
            assert list(workspaces_root.iterdir()) == []
        finally:
            shutil.rmtree(workspace_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_gzip_extraction_via_executor(self) -> None:
        workspace_dir = tempfile.mkdtemp()
        try:
            config = SandboxConfig(
                workspace_dir=workspace_dir,
                pull_policy="never",
            )
            sandbox = DockerSandbox(config)
            registry = ToolRegistry(sandbox=sandbox)
            registry.register_instance(PythonTool(sandbox))
            registry.register_instance(ArchiveTool(sandbox))
            executor = ExecutorAgent(tool_registry=registry, sandbox=sandbox)

            gz = _gzip_bytes(b"gzip payload", "message.txt")
            encoded = base64.b64encode(gz).decode()

            state = AgentState()
            state.plan = [
                PlanStep(
                    index=0,
                    description="Write the gzip file into the workspace",
                    tool_name="python",
                    tool_parameters={
                        "code": (
                            "import base64\n"
                            f"with open('/workspace/message.txt.gz', 'wb') as f:\n"
                            f"    f.write(base64.b64decode('{encoded}'))\n"
                            "print('written')\n"
                        ),
                    },
                ),
                PlanStep(
                    index=1,
                    description="Extract the gzip archive",
                    tool_name="archive",
                    tool_parameters={
                        "archive_path": "/workspace/message.txt.gz",
                        "extract_dir": "/workspace/out",
                    },
                    depends_on=[0],
                ),
                PlanStep(
                    index=2,
                    description="Read the decompressed file",
                    tool_name="python",
                    tool_parameters={
                        "code": (
                            "with open('/workspace/out/message.txt') as f:\n"
                            "    print(f.read())\n"
                        ),
                    },
                    depends_on=[1],
                ),
            ]

            with _patched_client(sandbox):
                result = await executor.process({"state": state})

            assert result["success"] is True, result["errors"]

            extract_step = result["results"][1]
            assert extract_step["output"]["archive_type"] == "gzip"
            assert extract_step["output"]["files"] == ["message.txt"]
            assert extract_step["output"]["destination"] == "/workspace/out"

            read_step = result["results"][2]
            assert "gzip payload" in read_step["output"]["stdout"]
        finally:
            shutil.rmtree(workspace_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_7z_unavailable_reports_gracefully(self) -> None:
        """7Z extraction without py7zr/7z support fails gracefully."""
        workspace_dir = tempfile.mkdtemp()
        try:
            config = SandboxConfig(
                workspace_dir=workspace_dir,
                pull_policy="never",
            )
            sandbox = DockerSandbox(config)
            registry = ToolRegistry(sandbox=sandbox)
            registry.register_instance(PythonTool(sandbox))
            registry.register_instance(ArchiveTool(sandbox))
            executor = ExecutorAgent(tool_registry=registry, sandbox=sandbox)

            encoded = base64.b64encode(b"not really 7z data").decode()

            state = AgentState()
            state.plan = [
                PlanStep(
                    index=0,
                    description="Write a 7z file into the workspace",
                    tool_name="python",
                    tool_parameters={
                        "code": (
                            "import base64\n"
                            f"with open('/workspace/data.7z', 'wb') as f:\n"
                            f"    f.write(base64.b64decode('{encoded}'))\n"
                            "print('written')\n"
                        ),
                    },
                ),
                PlanStep(
                    index=1,
                    description="Try to extract the 7z archive",
                    tool_name="archive",
                    tool_parameters={"archive_path": "/workspace/data.7z"},
                    depends_on=[0],
                ),
            ]

            with _patched_client(sandbox):
                result = await executor.process({"state": state})

            extract_step = result["results"][1]
            assert extract_step["success"] is False
            assert "7Z" in (extract_step["error"] or "")
        finally:
            shutil.rmtree(workspace_dir, ignore_errors=True)
