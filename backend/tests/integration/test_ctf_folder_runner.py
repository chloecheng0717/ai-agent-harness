"""
Integration Tests for the General CTF Folder Runner

Runs the full pipeline (folder loader → orchestrator → tool registry →
sandbox) against a real challenge folder with a scripted LLM provider
and a simulated container that executes tool commands against the host
workspace directory.

This validates, end to end, that:

* ``challenge.txt`` (or the folder's sole ``*.txt``) is parsed into
  Title / Category / Points / Description / Hints / Files.
* Sibling data files are shipped into the persistent run workspace.
* The run stops when a flag-like token appears in real tool output and
  reports it as the recovered flag — no expected flag is required.

No Docker daemon or API key is required.
"""

import base64
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
from app.ctf.runner import (
    extract_flags,
    load_challenge_folder,
    parse_challenge_text,
    run_challenge_folder,
    to_challenge,
)
from app.models.llm import (
    BaseLLMProvider,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    LLMProviderConfig,
    LLMUnavailableError,
    ModelInfo,
    TokenUsageInfo,
)
from app.sandbox.docker_sandbox import (
    CleanupPolicy,
    DockerSandbox,
    SandboxConfig,
)

FLAG_PATTERN = re.compile(r"(?:flag|ctf|academy)\{[^\}]+\}", re.IGNORECASE)


# ── Scripted LLM provider ─────────────────────────────────────────


class _StubProvider(BaseLLMProvider):
    """Deterministic provider that scripts the whole agent lifecycle."""

    def __init__(self, plan: list[dict[str, Any]]) -> None:
        super().__init__(LLMProviderConfig(model="runctf-stub"))
        self.plan = plan

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
            model="runctf-stub",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=content)
                )
            ],
            usage=TokenUsageInfo(prompt_tokens=1, completion_tokens=1),
        )

    async def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_name="runctf-stub")


class _UnavailableProvider(BaseLLMProvider):
    """Provider that fails the first LLM call with LLMUnavailableError."""

    def __init__(self) -> None:
        super().__init__(LLMProviderConfig(model="unavailable-stub"))

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        raise LLMUnavailableError(
            "LLM unavailable (daily quota exceeded 429)",
            provider=self.config.provider_type.value,
            status_code=429,
        )

    async def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_name="unavailable-stub")


# ── Simulated container ───────────────────────────────────────────


class _SimulatedContainer:
    """Executes tool commands against the host workspace directory."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.short_id = "runctf0001"
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
        if command.startswith("python -u "):
            self._run_python(command)
        else:
            self._run_shell(command)
        return {"StatusCode": self._exit_code, "Error": ""}

    def logs(self, stdout: bool = True, stderr: bool = False) -> bytes:
        if stdout:
            return self._stdout.encode("utf-8", errors="replace")
        if stderr:
            return self._stderr.encode("utf-8", errors="replace")
        return b""

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
        """Run a python script, remapping /workspace inside its source."""
        script_path = command[len("python -u "):].strip().strip('"\'')
        script = Path(script_path)
        if self._workspace_source and script.exists():
            host = self._workspace_source.replace("\\", "/")
            original = script.read_text(encoding="utf-8")
            rewritten = original.replace("/workspace", host)
            if rewritten != original:
                script.write_text(rewritten, encoding="utf-8")
                try:
                    self._run_shell(command)
                finally:
                    script.write_text(original, encoding="utf-8")
                return
        self._run_shell(command)


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
    base = tempfile.mkdtemp(prefix="runctf-")
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


def _write_challenge_folder(
    base: Path,
    *,
    flag: str = "flag{from_run_ctf}",
) -> Path:
    folder = base / "Hidden_Message"
    folder.mkdir(parents=True)
    (folder / "challenge.txt").write_text(
        "Title:\n"
        "Hidden Message\n"
        "\n"
        "Category:\n"
        "Encoding\n"
        "\n"
        "Points:\n"
        "100\n"
        "\n"
        "Description:\n"
        "A secret message hides the flag.\n"
        "\n"
        "Hints:\n"
        "- The payload is base64 encoded.\n"
        "\n"
        "Files:\n"
        "message.txt\n",
        encoding="utf-8",
    )
    (folder / "message.txt").write_text(
        base64.b64encode(flag.encode("utf-8")).decode(),
        encoding="utf-8",
    )
    return folder


def _decode_plan() -> list[dict[str, Any]]:
    return [
        {
            "index": 0,
            "description": "Decode the base64 message to recover the flag",
            "tool_name": "python",
            "rationale": "The message is base64 text in the workspace.",
            "tool_parameters": {
                "code": (
                    "import base64\n"
                    "data = open('/workspace/message.txt').read().strip()\n"
                    "print(base64.b64decode(data).decode())\n"
                )
            },
        }
    ]


# ═══════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════


class TestDescriptionParsing:
    def test_parse_template_format(self) -> None:
        text = (
            "Title:\n"
            "Secret of the Polyglot\n"
            "\n"
            "Category:\n"
            "Forensics\n"
            "\n"
            "Points:\n"
            "100\n"
            "\n"
            "Description:\n"
            "Can you find the flag?\n"
            "\n"
            "Hints:\n"
            "- Files can have more than one format.\n"
            "\n"
            "Challenge URL:\n"
            "https://play.picoctf.org/practice/challenge/123\n"
            "\n"
            "Files:\n"
            "polyglot.png\n"
        )
        sections = parse_challenge_text(text)
        assert sections["title"] == ["Secret of the Polyglot"]
        assert sections["category"] == ["Forensics"]
        assert sections["description"] == ["Can you find the flag?"]
        assert sections["files"] == ["polyglot.png"]
        assert sections["challenge url"] == [
            "https://play.picoctf.org/practice/challenge/123"
        ]

    def test_parse_network_link_header(self) -> None:
        text = (
            "Title:\nCrack the Gate 1\n"
            "Description:\nCan you figure it out?\n"
            "Category:\nWeb Exploitation\n"
            "Link:\nhttp://amiable-citadel.picoctf.net:53877/\n"
        )
        sections = parse_challenge_text(text)
        assert sections["link"] == ["http://amiable-citadel.picoctf.net:53877/"]

    def test_description_carries_endpoint_line(self) -> None:
        text = (
            "Title:\nPerceptron Play Naught\n"
            "Description:\nSeparate the labeled points with a single line.\n\n"
            "$ nc aureolin-pixie.cylabacademy.net 63576\n"
        )
        sections = parse_challenge_text(text)
        assert any(
            "nc aureolin-pixie.cylabacademy.net 63576" in line
            for line in sections["description"]
        )


class TestFolderLoading:
    def test_load_challenge_folder(self, tmp_path: Path) -> None:
        folder = _write_challenge_folder(tmp_path)
        parsed = load_challenge_folder(folder)

        assert parsed.title == "Hidden Message"
        assert parsed.category == "Encoding"
        assert parsed.points == "100"
        assert "A secret message hides the flag." in parsed.description
        assert "- The payload is base64 encoded." in parsed.hints
        assert parsed.declared_files == ["message.txt"]
        assert sorted(parsed.payload_files) == ["message.txt"]

    def test_load_uses_sole_txt_as_description(self, tmp_path: Path) -> None:
        folder = tmp_path / "Perceptron_Play_Naught"
        folder.mkdir()
        (folder / "PerceptronPlayNaught.txt").write_text(
            "Title:\nPerceptron Play Naught\n"
            "Description:\nSeparate the labeled points.\n\n"
            "$ nc host.example 12345\n",
            encoding="utf-8",
        )
        parsed = load_challenge_folder(folder)
        assert parsed.title == "Perceptron Play Naught"
        assert "nc host.example 12345" in parsed.description
        assert parsed.payload_files == {}

    def test_to_challenge_goal(self, tmp_path: Path) -> None:
        folder = _write_challenge_folder(tmp_path)
        parsed = load_challenge_folder(folder)
        challenge = to_challenge(folder, parsed)

        assert challenge.name == "Hidden_Message"
        assert challenge.category == "Encoding"
        assert challenge.difficulty == "easy"
        assert challenge.expected_flag == ""
        assert "Solve the CTF challenge: Hidden Message" in challenge.goal
        assert "base64 encoded" in challenge.goal
        assert "Files available in the workspace: message.txt" in challenge.goal
        assert set(challenge.files) == {"message.txt"}
        assert "python" in challenge.required_capabilities

    def test_extract_flags(self) -> None:
        assert extract_flags("no flags here") == []
        assert extract_flags("flag{a} and ctf{b}") == ["flag{a}", "ctf{b}"]
        assert extract_flags("flag{a} flag{a}") == ["flag{a}"]


class TestRunChallengeFolder:
    @pytest.mark.asyncio
    async def test_recovers_flag_without_expected_flag(
        self, tmp_path: Path, workspace_dir: str
    ) -> None:
        folder = _write_challenge_folder(tmp_path)
        sandbox = _make_sandbox(workspace_dir)
        runner_kwargs: dict[str, Any] = {}

        with _patched_client(sandbox):
            result = await run_challenge_folder(
                folder,
                sandbox=sandbox,
                llm_provider=_StubProvider(_decode_plan()),
                max_iterations=2,
                **runner_kwargs,
            )

        assert result["flag_found"] is True, result["errors"]
        assert result["flags"] == ["flag{from_run_ctf}"]
        assert result["status"] == "completed"
        assert result["iterations"] == 1
        assert "python" in result["tools_used"]
        assert "flag{from_run_ctf}" in result["final_answer"]
        assert result["llm_unavailable"] is False

    @pytest.mark.asyncio
    async def test_reports_llm_unavailable(
        self, tmp_path: Path, workspace_dir: str
    ) -> None:
        folder = _write_challenge_folder(tmp_path)
        sandbox = _make_sandbox(workspace_dir)

        with _patched_client(sandbox):
            result = await run_challenge_folder(
                folder,
                sandbox=sandbox,
                llm_provider=_UnavailableProvider(),
                max_iterations=2,
            )

        assert result["llm_unavailable"] is True
        assert result["flag_found"] is False
        assert any("quota" in e for e in result["errors"])
