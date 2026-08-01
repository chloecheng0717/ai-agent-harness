"""
Integration Tests: Persistent Shared Workspace

These tests verify the persistent-workspace architecture end to end:

* ExecutorAgent creates one workspace per execution run.
* Every tool call within the run is handed the same workspace, so a file
  written by one tool is readable by a later tool call.
* The workspace is removed when the run finishes.
* Concurrent runs use distinct workspace IDs and do not interfere.

Docker interactions are simulated: the fake container "executes" tool
commands against the host-side workspace directory that a real container
would mount at ``/workspace``, so cross-tool persistence is exercised for
real without requiring a running Docker daemon.
"""

import re
import shutil
import subprocess
import tempfile
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
from app.tools.plugins.python_tool import PythonTool
from app.tools.plugins.strings_tool import StringsTool
from app.tools.registry import ToolRegistry

# ── Simulated container ──────────────────────────────────────────


class _SimulatedContainer:
    """Simulates a Docker container against the host workspace directory.

    The real container would run the tool command with the shared
    workspace mounted at ``/workspace``.  This fake does the same thing
    on the host: it remaps ``/workspace`` paths to the workspace's host
    directory and executes Python scripts / reads target files locally.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.short_id = "sim000001"
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

        if command.startswith("python "):
            self._run_python(command)
        elif command.startswith("strings "):
            self._run_strings(command)
        elif command.startswith("file "):
            self._run_file(command)
        else:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self._workspace_source,
            )
            self._stdout = proc.stdout
            self._stderr = proc.stderr
            return {"StatusCode": proc.returncode, "Error": ""}

        return {"StatusCode": 0, "Error": ""}

    def _run_python(self, command: str) -> None:
        """Execute a python command, remapping /workspace inside scripts."""
        rest = command[len("python "):].strip()
        if not rest.startswith("-c"):
            script_path = rest.strip().strip("'\"")
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
                            self._run_python_proc(command)
                        finally:
                            script.write_text(original, encoding="utf-8")
                        return
        self._run_python_proc(command)

    def _run_python_proc(self, command: str) -> None:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=self._workspace_source,
        )
        self._stdout = proc.stdout
        self._stderr = proc.stderr

    def _remap_path(self, path: str) -> str:
        if self._workspace_source:
            return path.replace(
                "/workspace", self._workspace_source.replace("\\", "/")
            )
        return path

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

    def _run_file(self, command: str) -> None:
        target = self._remap_path(command.split()[-1])
        data = Path(target).read_bytes()
        if data.startswith(b"\x50\x4b"):
            mimetype = "application/zip"
        elif data.startswith(b"\x89PNG"):
            mimetype = "image/png"
        else:
            mimetype = "text/plain"
        self._stdout = mimetype + "\n"

    def logs(self, stdout: bool = True, stderr: bool = False) -> bytes:
        if stdout:
            return self._stdout.encode("utf-8", errors="replace")
        if stderr:
            return self._stderr.encode("utf-8", errors="replace")
        return b""


@contextmanager
def _patched_client(sandbox: DockerSandbox) -> Iterator[MagicMock]:
    """Patch the sandbox's client so containers are simulated."""
    client = MagicMock()
    client.ping.return_value = True
    client.images.get.return_value = MagicMock()
    client.containers.create.side_effect = lambda **kwargs: _SimulatedContainer(kwargs)
    with patch.object(sandbox, "_get_client", return_value=client):
        yield client


@pytest.mark.asyncio
async def test_file_written_by_one_tool_readable_by_later_tool() -> None:
    """A file written by one tool persists for a later tool in the same run."""
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
        registry.register_instance(StringsTool(sandbox))

        executor = ExecutorAgent(tool_registry=registry, sandbox=sandbox)

        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Write the flag to a workspace file",
                tool_name="python",
                tool_parameters={
                    "code": (
                        "with open('/workspace/secret.txt', 'w') as f:\n"
                        "    f.write('FLAG{cross_tool_persistence}')\n"
                        "print('written')\n"
                    ),
                },
            ),
            PlanStep(
                index=1,
                description="Read strings from the file written earlier",
                tool_name="strings",
                tool_parameters={"path": "/workspace/secret.txt"},
                depends_on=[0],
            ),
        ]

        with _patched_client(sandbox):
            result = await executor.process({"state": state})

        assert result["success"] is True, result["errors"]
        assert len(result["results"]) == 2

        strings_output = result["results"][1]["output"]
        assert "FLAG{cross_tool_persistence}" in strings_output["strings"]

        # Workspace is removed once the run finishes
        workspaces_root = Path(workspace_dir) / "sandbox_workspaces"
        assert list(workspaces_root.iterdir()) == []
    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_each_run_gets_its_own_workspace() -> None:
    """Two sequential runs use distinct workspaces and stay isolated."""
    workspace_dir = tempfile.mkdtemp()
    try:
        config = SandboxConfig(
            workspace_dir=workspace_dir,
            pull_policy="never",
        )
        sandbox = DockerSandbox(config)

        registry = ToolRegistry(sandbox=sandbox)
        registry.register_instance(PythonTool(sandbox))
        registry.register_instance(StringsTool(sandbox))

        executor = ExecutorAgent(tool_registry=registry, sandbox=sandbox)

        def build_state(value: str) -> AgentState:
            state = AgentState()
            state.plan = [
                PlanStep(
                    index=0,
                    description="Write a value to a workspace file",
                    tool_name="python",
                    tool_parameters={
                        "code": (
                            f"with open('/workspace/data.txt', 'w') as f:\n"
                            f"    f.write('{value}')\n"
                            "print('written')\n"
                        ),
                    },
                ),
                PlanStep(
                    index=1,
                    description="Read the value back",
                    tool_name="strings",
                    tool_parameters={"path": "/workspace/data.txt"},
                    depends_on=[0],
                ),
            ]
            return state

        with _patched_client(sandbox):
            result_a = await executor.process(
                {"state": build_state("run_a_secret")}
            )
            result_b = await executor.process(
                {"state": build_state("run_b_secret")}
            )

        assert result_a["success"] is True
        assert result_b["success"] is True

        strings_a = result_a["results"][1]["output"]["strings"]
        strings_b = result_b["results"][1]["output"]["strings"]

        assert any("run_a_secret" in s for s in strings_a)
        assert any("run_b_secret" in s for s in strings_b)
        assert not any("run_b_secret" in s for s in strings_a)
        assert not any("run_a_secret" in s for s in strings_b)
    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_workspace_id_forwarded_to_tool_requests() -> None:
    """The same workspace_id is attached to every tool request in a run."""
    workspace_dir = tempfile.mkdtemp()
    try:
        config = SandboxConfig(
            workspace_dir=workspace_dir,
            pull_policy="never",
        )
        sandbox = DockerSandbox(config)

        registry = ToolRegistry(sandbox=sandbox)
        registry.register_instance(PythonTool(sandbox))
        registry.register_instance(StringsTool(sandbox))

        executor = ExecutorAgent(tool_registry=registry, sandbox=sandbox)

        seen_workspaces: list = []
        original_execute = registry.execute

        async def spy(request, **kwargs):
            seen_workspaces.append(request.workspace_id)
            return await original_execute(request, **kwargs)

        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Write the flag to a workspace file",
                tool_name="python",
                tool_parameters={
                    "code": (
                        "with open('/workspace/flag.txt', 'w') as f:\n"
                        "    f.write('FLAG{spy_check}')\n"
                        "print('written')\n"
                    ),
                },
            ),
            PlanStep(
                index=1,
                description="Read the flag back",
                tool_name="strings",
                tool_parameters={"path": "/workspace/flag.txt"},
                depends_on=[0],
            ),
        ]

        with _patched_client(sandbox):
            with patch.object(registry, "execute", new=spy):
                result = await executor.process({"state": state})

        assert result["success"] is True
        assert len(seen_workspaces) == 2
        assert seen_workspaces[0] is not None
        assert seen_workspaces[0] == seen_workspaces[1]
    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)
