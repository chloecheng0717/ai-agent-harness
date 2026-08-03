"""
General CTF Folder Runner

Loads a CTF challenge from a plain-text description folder
(``challenge.txt`` plus sibling data files) and drives the full agent
pipeline (orchestrator → planner → tool registry → Docker sandbox) until
a flag-like token (``flag{...}`` / ``ctf{...}`` / ``academy{...}``)
appears in real tool output or the iteration cap is reached.

The folder format is intentionally loose so real-world challenge folders
work without editing:

* The description file is ``challenge.txt`` or the folder's only
  ``*.txt`` file.
* Recognised headers (``Title:``, ``Category:``, ``Points:``,
  ``Description:``, ``Hints:``, ``Challenge URL:``/``Link:``,
  ``Files:``) are lines ending with ``:``.
* Every sibling file (anything but the description file) is shipped
  into the agent's workspace as challenge payload.

No expected flag is required: completion is decided by the orchestrator's
deterministic guard, which stops as soon as a flag-like token appears in
real tool output.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.benchmarks.models import Challenge, RunTrace
from app.benchmarks.orchestrator import BenchmarkOrchestrator
from app.config import get_settings
from app.models.llm import BaseLLMProvider
from app.sandbox.docker_sandbox import DockerSandbox, SandboxConfig
from app.tools.plugins.archive_tool import ArchiveTool
from app.tools.plugins.encoding_tool import EncodingTool
from app.tools.plugins.file_tool import FileTool
from app.tools.plugins.python_tool import PythonTool
from app.tools.plugins.strings_tool import StringsTool
from app.tools.registry import ToolRegistry

#: Flag-like tokens recognised as a completed solve.  Mirrors the
#: deterministic completion guard in the orchestrator so the two can never
#: disagree about what a recovered flag looks like.
FLAG_PATTERN = re.compile(r"(?:flag|ctf|academy)\{[^\}]+\}", re.IGNORECASE)

#: The complete CTF toolchain surfaced to the planner for folder challenges.
_CTF_TOOL_CLASSES = (
    FileTool,
    StringsTool,
    PythonTool,
    EncodingTool,
    ArchiveTool,
)

#: Tool names exposed to the planner for every folder challenge.
CTF_TOOL_NAMES = ["file", "strings", "python", "encoding", "archive"]

#: Recognised section headers in a challenge description file.
_HEADER_KEYS = frozenset(
    {
        "title",
        "category",
        "points",
        "description",
        "hints",
        "challenge url",
        "link",
        "files",
    }
)


@dataclass
class ParsedChallenge:
    """Structured contents of a challenge folder."""

    title: str = ""
    category: str = ""
    points: str = ""
    description: str = ""
    hints: str = ""
    link: str = ""
    declared_files: list[str] = field(default_factory=list)
    payload_files: dict[str, bytes] = field(default_factory=dict)


# ── Description file parsing ───────────────────────────────────────


def _header_key(line: str) -> str | None:
    """Return the canonical section key if *line* is a bare ``Key:`` line."""
    if not line.endswith(":") or ":" not in line:
        return None
    key = line[:-1].strip().lower()
    return key if key in _HEADER_KEYS else None


def parse_challenge_text(text: str) -> dict[str, list[str]]:
    """
    Parse a challenge description file into section → lines.

    A section starts at a recognised ``Key:`` line and runs until the next
    recognised key.  Header values are case-insensitive; unknown ``Key:``
    lines are treated as ordinary content so description prose is not lost.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        key = _header_key(line)
        if key is not None:
            current = key
            sections.setdefault(key, [])
        elif current is not None and line:
            sections[current].append(line)
    return sections


def _first(sections: dict[str, list[str]], key: str) -> str:
    for line in sections.get(key, []):
        if line.strip():
            return line.strip()
    return ""


def _joined(sections: dict[str, list[str]], key: str) -> str:
    return "\n".join(line for line in sections.get(key, []) if line.strip()).strip()


# ── Folder loading ─────────────────────────────────────────────────


def find_description_file(folder: Path) -> Path | None:
    """Locate the description file: ``challenge.txt`` or the sole ``*.txt``."""
    candidate = folder / "challenge.txt"
    if candidate.is_file():
        return candidate
    txts = sorted(folder.glob("*.txt"))
    if len(txts) == 1:
        return txts[0]
    return None


def load_challenge_folder(folder: Path) -> ParsedChallenge:
    """Read and parse a challenge folder (description + payload files)."""
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    desc_file = find_description_file(folder)
    if desc_file is None:
        raise FileNotFoundError(
            f"No description file found in {folder}: expected 'challenge.txt' "
            "or a single '*.txt' file."
        )

    sections = parse_challenge_text(
        desc_file.read_text(encoding="utf-8", errors="replace")
    )

    payload_files: dict[str, bytes] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path == desc_file:
            continue
        try:
            payload_files[path.name] = path.read_bytes()
        except OSError:
            continue

    return ParsedChallenge(
        title=_first(sections, "title"),
        category=_first(sections, "category"),
        points=_first(sections, "points"),
        description=_joined(sections, "description"),
        hints=_joined(sections, "hints"),
        link=_first(sections, "link") or _first(sections, "challenge url"),
        declared_files=[line for line in sections.get("files", []) if line.strip()],
        payload_files=payload_files,
    )


# ── Challenge building ─────────────────────────────────────────────


def build_goal(parsed: ParsedChallenge) -> str:
    """Synthesise the task description handed to the agent."""
    parts: list[str] = []
    if parsed.title:
        parts.append(f"Solve the CTF challenge: {parsed.title}")
    if parsed.description:
        parts.append(parsed.description)
    if parsed.hints:
        parts.append(f"Hints: {parsed.hints}")
    if parsed.link:
        parts.append(f"Target: {parsed.link}")
    if parsed.payload_files:
        parts.append(
            "Files available in the workspace: "
            + ", ".join(sorted(parsed.payload_files))
        )
    parts.append(
        "The recovered answer will be a CTF flag: the word 'flag', 'ctf', "
        "or 'academy' followed by curly braces enclosing the secret text. "
        "Recover it using the available tools."
    )
    return "\n".join(parts)


def _difficulty_from_points(points: str) -> str:
    try:
        value = int(points)
    except ValueError:
        return "unknown"
    if value <= 100:
        return "easy"
    if value <= 300:
        return "medium"
    return "hard"


def to_challenge(folder: Path, parsed: ParsedChallenge) -> Challenge:
    """Build a :class:`Challenge` from a parsed folder."""
    return Challenge(
        name=folder.name,
        category=parsed.category or "unknown",
        description=parsed.title or folder.name,
        difficulty=_difficulty_from_points(parsed.points),
        goal=build_goal(parsed),
        expected_flag="",
        required_capabilities=list(CTF_TOOL_NAMES),
        files=parsed.payload_files,
    )


# ── Flag extraction ────────────────────────────────────────────────


def extract_flags(text: str) -> list[str]:
    """Return every distinct flag-like token found in *text*."""
    seen: list[str] = []
    for match in FLAG_PATTERN.finditer(text or ""):
        if match.group(0) not in seen:
            seen.append(match.group(0))
    return seen


# ── Running ─────────────────────────────────────────────────────────


def build_sandbox(settings: Any) -> DockerSandbox:
    """Build a :class:`DockerSandbox` from application settings."""
    return DockerSandbox(
        SandboxConfig(
            image=settings.sandbox_image,
            memory_limit=settings.sandbox_memory_limit,
            cpu_limit=settings.sandbox_cpu_limit,
            timeout=settings.sandbox_timeout,
            network_disabled=settings.sandbox_network_disabled,
        )
    )


def build_registry(sandbox: DockerSandbox) -> ToolRegistry:
    """Register the full CTF toolchain against *sandbox*."""
    registry = ToolRegistry(sandbox=sandbox)
    for tool_class in _CTF_TOOL_CLASSES:
        registry.register_instance(tool_class(sandbox))
    return registry


async def run_ctf_challenge(
    challenge: Challenge,
    *,
    settings: Any = None,
    sandbox: DockerSandbox | None = None,
    llm_provider: BaseLLMProvider | None = None,
    max_iterations: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run one challenge through the full agent pipeline.

    Parameters
    ----------
    challenge : Challenge
        The challenge to solve (files are injected into the workspace).
    settings : Settings | None
        App settings; defaults to the cached singleton.
    sandbox : DockerSandbox | None
        A pre-built sandbox (used by tests with a simulated client).  When
        omitted one is built from the settings.
    llm_provider : BaseLLMProvider | None
        Provider to drive the orchestrator.  When omitted a provider is
        built from the settings.

    Returns
    -------
    dict
        A serialisable summary: status, iterations, tools used, errors,
        execution time, recovered flags, and the final answer.
    """
    settings = settings or get_settings()
    if sandbox is None:
        sandbox = build_sandbox(settings)
    if llm_provider is None:
        from app.models.providers.openai_provider import OpenAIProvider

        llm_provider = OpenAIProvider()

    orchestrator = BenchmarkOrchestrator(
        llm_provider=llm_provider,
        tool_registry=build_registry(sandbox),
        sandbox=sandbox,
        challenge_files=challenge.files,
        verbose=verbose,
        max_iterations=max_iterations,
    )

    start = time.monotonic()
    try:
        raw = await orchestrator.run(
            task_id=uuid4(),
            task_data={"description": challenge.goal},
            tools_available=challenge.required_capabilities,
        )
    except Exception as exc:
        raw = {
            "status": "crashed",
            "final_answer": "",
            "errors": [str(exc)],
            "tools_used": [],
            "benchmark_trace": [],
            "benchmark_workspace_files": [],
            "iterations": 0,
        }
    duration_ms = (time.monotonic() - start) * 1000

    trace = RunTrace(
        tool_calls=raw.get("benchmark_trace", []),
        workspace_files=raw.get("benchmark_workspace_files", []),
        final_answer=raw.get("final_answer") or "",
        errors=raw.get("errors", []),
        status=raw.get("status", ""),
        iterations=raw.get("iterations", 0),
        execution_time_ms=duration_ms,
    )
    flags = extract_flags(trace.all_text())

    return {
        "name": challenge.name,
        "category": challenge.category,
        "difficulty": challenge.difficulty,
        "goal": challenge.goal,
        "files": sorted(challenge.files),
        "status": trace.status,
        "iterations": trace.iterations,
        "tools_used": list(raw.get("tools_used", [])),
        "errors": list(trace.errors),
        "execution_time_ms": round(duration_ms, 2),
        "flags": flags,
        "flag_found": bool(flags),
        "final_answer": trace.final_answer,
        "llm_unavailable": bool(raw.get("llm_unavailable")),
    }


async def run_challenge_folder(
    folder: Path,
    *,
    settings: Any = None,
    sandbox: DockerSandbox | None = None,
    llm_provider: BaseLLMProvider | None = None,
    max_iterations: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Load a challenge folder and run it end to end."""
    parsed = load_challenge_folder(folder)
    result = await run_ctf_challenge(
        to_challenge(folder, parsed),
        settings=settings,
        sandbox=sandbox,
        llm_provider=llm_provider,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    result["folder"] = str(folder)
    result["title"] = parsed.title
    result["points"] = parsed.points
    result["link"] = parsed.link
    return result
