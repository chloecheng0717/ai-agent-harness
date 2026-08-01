"""
Integration Tests for the CTF Benchmark Runner

Runs the full benchmark pipeline (orchestrator → tool registry → sandbox)
against real challenge files with a scripted LLM provider and a simulated
container that executes tool commands against the host workspace directory.

This validates, end to end, that:

* Challenge files are injected into the persistent run workspace.
* LLM-authored plans with real ``tool_parameters`` drive the existing CTF
  tools (``python``, ``archive``, ``strings``).
* The orchestrator captures a trace, recovers the flag into the final
  answer, and the runner reports pass/fail, failed milestone, tools used,
  iterations, and execution time.

No Docker daemon or API key is required.
"""

import json
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
from app.benchmarks.challenges import build_all_challenges, get_challenge
from app.benchmarks.models import BenchmarkReport, ChallengeResult, RunTrace
from app.benchmarks.runner import BenchmarkRunner
from app.models.llm import (
    BaseLLMProvider,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    LLMProviderConfig,
    ModelInfo,
    TokenUsageInfo,
)
from app.sandbox.docker_sandbox import (
    CleanupPolicy,
    DockerSandbox,
    SandboxConfig,
)

FLAG_PATTERN = re.compile(r"(?:flag|ctf)\{[^\}]+\}", re.IGNORECASE)


# ── Scripted LLM provider ─────────────────────────────────────────


class _StubProvider(BaseLLMProvider):
    """Deterministic provider that scripts the whole agent lifecycle.

    Dispatches on the system message of each lifecycle step: planning
    returns a canned JSON plan, evaluation reports COMPLETE when the flag
    is present in the results, and the final response extracts the flag
    from the results that were actually produced by the tools.
    """

    def __init__(self, plan: list[dict[str, Any]]) -> None:
        super().__init__(LLMProviderConfig(model="benchmark-stub"))
        self.plan = plan
        self.system_messages: list[str] = []

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        system = next(
            (m.content for m in request.messages if m.role == "system"),
            "",
        )
        user = next(
            (m.content for m in request.messages if m.role == "user"),
            "",
        )
        self.system_messages.append(system)

        if "task analyst" in system:
            content = user.strip()
        elif "planning agent" in system:
            content = json.dumps(self.plan)
        elif "execution evaluator" in system:
            content = (
                "COMPLETE: YES"
                if FLAG_PATTERN.search(user)
                else "COMPLETE: NO"
            )
        elif "reflective agent" in system:
            content = "Reflection recorded for the run."
        elif "helpful assistant" in system:
            match = FLAG_PATTERN.search(user)
            content = match.group(0) if match else "No flag recovered."
        else:
            content = ""

        return ChatCompletionResponse(
            model="benchmark-stub",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=content)
                )
            ],
            usage=TokenUsageInfo(prompt_tokens=1, completion_tokens=1),
        )

    async def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_name="benchmark-stub")


# ── Simulated container ───────────────────────────────────────────


class _SimulatedContainer:
    """Executes tool commands against the host workspace directory.

    Mirrors the pattern used by the existing archive integration tests: the
    command's ``/workspace`` mount is remapped to the workspace's host path
    and the tools run locally, so cross-tool file persistence is real.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.short_id = "bench0001"
        self._stdout = ""
        self._stderr = ""
        self._exit_code = 0
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
        elif command.startswith("file "):
            self._run_file(command)
        else:
            self._run_shell(command)

        return {"StatusCode": self._exit_code, "Error": ""}

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
        self._exit_code = proc.returncode

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
        match = re.search(r"-n\s+(\d+)\s+(.+)$", command)
        if match:
            min_length = int(match.group(1))
            target = match.group(2).strip().strip('"\'')
        else:
            min_length = 4
            target = command.split()[-1]
        data = Path(self._remap_path(target)).read_bytes()
        sequences = re.findall(rb"[ -~]{%d,}" % min_length, data)
        self._stdout = "\n".join(
            s.decode("ascii", errors="replace") for s in sequences
        )
        if self._stdout:
            self._stdout += "\n"
        self._stderr = ""

    def _run_file(self, command: str) -> None:
        match = re.search(r"-b\s+(.+)$", command)
        target = (match.group(1) if match else command.split()[-1]).strip().strip('"\'')
        data = Path(self._remap_path(target)).read_bytes()
        self._stdout = _sniff_mime(data) + "\n"
        self._stderr = ""

    def _prepare_archive_params(self) -> None:
        workspace = self._workspace_source
        if workspace is None:
            return
        params_file = Path(workspace) / "_archive_params.json"
        if params_file.exists():
            text = params_file.read_text(encoding="utf-8")
            text = text.replace("/workspace", workspace.replace("\\", "/"))
            params_file.write_text(text, encoding="utf-8")

    def _remap_archive_result(self) -> None:
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


def _sniff_mime(data: bytes) -> str:
    if data.startswith(b"PK\x03\x04"):
        return "application/zip"
    if data.startswith(b"\x1f\x8b"):
        return "application/gzip"
    try:
        data.decode("utf-8")
        return "text/plain"
    except UnicodeDecodeError:
        return "application/octet-stream"


@contextmanager
def _patched_client(sandbox: DockerSandbox) -> Iterator[MagicMock]:
    client = MagicMock()
    client.ping.return_value = True
    client.images.get.return_value = MagicMock()
    client.containers.create.side_effect = lambda **kwargs: _SimulatedContainer(kwargs)
    with patch.object(sandbox, "_get_client", return_value=client):
        yield client


# ── Fixtures / helpers ────────────────────────────────────────────


@pytest.fixture
def workspace_dir() -> Iterator[str]:
    base = tempfile.mkdtemp(prefix="ctf-bench-")
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _make_sandbox(base: str) -> DockerSandbox:
    return DockerSandbox(
        SandboxConfig(
            workspace_dir=base,
            pull_policy="never",
            cleanup_policy=CleanupPolicy.ALWAYS,
        )
    )


def _reference_flag_for(challenge_name: str) -> str:
    return get_challenge(challenge_name).expected_flag


# ═══════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════


class TestBenchmarkRunnerPassesChallenges:
    @pytest.mark.asyncio
    async def test_encoding_base64_passes(self, workspace_dir: str) -> None:
        challenge = get_challenge("encoding_base64")
        plan = [
            {
                "index": 0,
                "description": "Read the base64 message from the file",
                "tool_name": "python",
                "rationale": "The message is stored as text in the workspace.",
                "tool_parameters": {
                    "code": "print(open('/workspace/message.txt').read().strip())"
                },
            },
            {
                "index": 1,
                "description": "Decode the base64 text to recover the flag",
                "tool_name": "python",
                "rationale": "Base64-decode the file contents and print them.",
                "tool_parameters": {
                    "code": (
                        "import base64\n"
                        "data = open('/workspace/message.txt').read().strip()\n"
                        "print(base64.b64decode(data).decode())\n"
                    )
                },
            },
        ]
        sandbox = _make_sandbox(workspace_dir)
        runner = BenchmarkRunner(
            settings=None,
            sandbox=sandbox,
            llm_provider=_StubProvider(plan),
            max_iterations=3,
        )

        with _patched_client(sandbox):
            result: ChallengeResult = await runner.run_challenge(challenge)

        assert result.passed is True, result.errors
        assert result.flag_found is True
        assert result.status == "completed"
        assert result.failed_step is None
        assert [m["passed"] for m in result.milestones] == [True, True, True]
        assert result.iterations == 1
        assert "python" in result.tools_used
        assert result.final_answer == challenge.expected_flag
        assert result.expected_flag == challenge.expected_flag
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_multi_step_challenge_passes(self, workspace_dir: str) -> None:
        challenge = get_challenge("multi_step")
        plan = [
            {
                "index": 0,
                "description": "Extract the ZIP archive",
                "tool_name": "archive",
                "rationale": "The flag file is packed inside the archive.",
                "tool_parameters": {
                    "archive_path": "/workspace/challenge.zip",
                    "extract_dir": "/workspace/extracted",
                },
            },
            {
                "index": 1,
                "description": "Extract printable strings from the binary payload",
                "tool_name": "strings",
                "rationale": "The payload hides the encoded flag as a string.",
                "tool_parameters": {"path": "/workspace/extracted/payload.bin"},
            },
            {
                "index": 2,
                "description": "Decode the base64 string to recover the flag",
                "tool_name": "python",
                "rationale": "Decode the base64 token embedded in the payload.",
                "tool_parameters": {
                    "code": (
                        "import base64, re\n"
                        "data = open('/workspace/extracted/payload.bin', 'rb').read()\n"
                        "tokens = re.findall(rb'[A-Za-z0-9+/]{12,}={0,2}', data)\n"
                        "for token in tokens:\n"
                        "    try:\n"
                        "        decoded = base64.b64decode(token).decode()\n"
                        "        if decoded.startswith('flag{'):\n"
                        "            print(decoded)\n"
                        "            break\n"
                        "    except Exception:\n"
                        "        continue\n"
                    )
                },
            },
        ]
        sandbox = _make_sandbox(workspace_dir)
        runner = BenchmarkRunner(
            settings=None,
            sandbox=sandbox,
            llm_provider=_StubProvider(plan),
            max_iterations=3,
        )

        with _patched_client(sandbox):
            result: ChallengeResult = await runner.run_challenge(challenge)

        assert result.passed is True, result.errors
        assert result.status == "completed"
        assert result.failed_step is None
        assert [m["passed"] for m in result.milestones] == [True, True, True, True]
        assert result.iterations == 1
        assert set(["archive", "strings", "python"]).issubset(result.tools_used)
        assert result.final_answer == challenge.expected_flag

    @pytest.mark.asyncio
    async def test_flag_saved_to_workspace_file_passes(self, workspace_dir: str) -> None:
        challenge = get_challenge("encoding_base64")
        plan = [
            {
                "index": 0,
                "description": "Decode the base64 message and save the flag",
                "tool_name": "python",
                "rationale": "Decode the message and persist the result as a workspace file.",
                "tool_parameters": {
                    "code": (
                        "import base64\n"
                        "data = open('/workspace/message.txt').read().strip()\n"
                        "open('/workspace/flag.txt', 'w').write("
                        "base64.b64decode(data).decode())\n"
                        "print('saved')\n"
                    )
                },
            },
        ]
        sandbox = _make_sandbox(workspace_dir)
        runner = BenchmarkRunner(
            settings=None,
            sandbox=sandbox,
            llm_provider=_StubProvider(plan),
            max_iterations=1,
        )

        with _patched_client(sandbox):
            result: ChallengeResult = await runner.run_challenge(challenge)

        assert result.passed is True, result.errors
        assert result.flag_found is True
        assert result.status == "completed"
        assert result.failed_step is None
        assert [m["passed"] for m in result.milestones] == [True, True, True]
        assert "python" in result.tools_used

    @pytest.mark.asyncio
    async def test_failed_run_reports_first_failed_milestone(
        self, workspace_dir: str
    ) -> None:
        challenge = get_challenge("encoding_base64")
        plan = [
            {
                "index": 0,
                "description": "Try to decode the file",
                "tool_name": "python",
                "rationale": "Attempt a decode.",
                "tool_parameters": {
                    "code": "raise RuntimeError('decode attempt failed')\n"
                },
            }
        ]
        sandbox = _make_sandbox(workspace_dir)
        runner = BenchmarkRunner(
            settings=None,
            sandbox=sandbox,
            llm_provider=_StubProvider(plan),
            max_iterations=2,
        )

        with _patched_client(sandbox):
            result: ChallengeResult = await runner.run_challenge(challenge)

        assert result.passed is False
        assert result.flag_found is False
        assert result.status == "completed"
        assert result.failed_step == "Read the challenge file"
        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_run_challenges_aggregates(self, workspace_dir: str) -> None:
        challenges = build_all_challenges()
        plan = [
            {
                "index": 0,
                "description": "Recover the flag",
                "tool_name": "python",
                "rationale": "Decode and report the flag.",
                "tool_parameters": {
                    "code": (
                        "import base64\n"
                        "print(base64.b64decode("
                        "open('/workspace/message.txt').read().strip()).decode())\n"
                    )
                },
            }
        ]
        sandbox = _make_sandbox(workspace_dir)
        runner = BenchmarkRunner(
            settings=None,
            sandbox=sandbox,
            llm_provider=_StubProvider(plan),
            max_iterations=1,
        )

        with _patched_client(sandbox):
            results = await runner.run_challenges(challenges)

        report = BenchmarkReport(results=results)
        assert len(results) == 7
        assert 0.0 < report.success_rate <= 1.0
        assert report.total_time_ms >= 0
        for result in results:
            assert result.name in {c.name for c in challenges}
            assert result.failed_step is None if result.passed else True


class TestSimulatedContainer:
    """Unit-level checks of the simulated container helpers."""

    def test_sniff_mime(self) -> None:
        assert _sniff_mime(b"PK\x03\x04rest") == "application/zip"
        assert _sniff_mime(b"\x1f\x8brest") == "application/gzip"
        assert _sniff_mime(b"plain text") == "text/plain"
        assert _sniff_mime(b"\x00\x01\x02\xff") == "application/octet-stream"


class TestRunTraceAgainstBenchmark:
    def test_trace_from_result_fields(self) -> None:
        trace = RunTrace(
            tool_calls=[
                {
                    "step_index": 0,
                    "tool_name": "python",
                    "success": True,
                    "output": {"stdout": "flag{from_trace}"},
                }
            ],
            final_answer="flag{from_trace}",
            errors=[],
            status="completed",
            iterations=1,
            execution_time_ms=12.5,
        )
        assert trace.contains_flag("flag{from_trace}")
        assert trace.succeeded("python")
        assert trace.count_tool("python", success=True) == 1
