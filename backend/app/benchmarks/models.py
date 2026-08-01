"""
CTF Benchmark Data Models

Shared dataclasses describing a CTF challenge, the evidence an agent
run produces, and the per-challenge / aggregate results the benchmark
reporter aggregates.

Everything here is framework-agnostic (no dependencies on the sandbox,
tools, or LLM providers) so the milestone checks and report logic can be
unit-tested without Docker or an API key.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

#: Flag-like token used for display, not for pass/fail (exact match wins).
DEFAULT_FLAG_PATTERN = re.compile(r"(?:flag|ctf)\{[^\}]+\}", re.IGNORECASE)


# ── Evidence ──────────────────────────────────────────────────────


@dataclass
class RunTrace:
    """
    The observable evidence produced by a single agent run.

    Parameters
    ----------
    tool_calls : list[dict]
        One entry per executed plan step: ``step_index``, ``description``,
        ``tool_name``, ``success``, ``output``, ``error``.
    workspace_files : list[dict]
        Verified workspace files produced during execution (``path``,
        ``bytes``, ``content``).  Challenge seed files injected before the
        run are excluded.  See ``BenchmarkOrchestrator._snapshot_workspace``.
    final_answer : str
        The agent's final synthesized answer.  Deliberately NOT treated as
        pass/fail evidence — it is LLM-authored text.
    errors : list[str]
        Errors recorded during the run.
    status : str
        Serialized lifecycle status (e.g. ``"completed"``).
    iterations : int
        Number of plan-execute-evaluate iterations the orchestrator ran.
    execution_time_ms : float
        Wall-clock duration of the whole run.
    """

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    workspace_files: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    errors: list[str] = field(default_factory=list)
    status: str = ""
    iterations: int = 0
    execution_time_ms: float = 0.0

    def tool_output_text(self) -> str:
        """Concatenate every real tool-execution output/error string."""
        parts: list[str] = []
        for call in self.tool_calls:
            output = call.get("output")
            if isinstance(output, str):
                parts.append(output)
            elif output is not None:
                parts.append(json.dumps(output, default=str))
            if call.get("error"):
                parts.append(str(call["error"]))
        return "\n".join(parts)

    def workspace_evidence_text(self) -> str:
        """Concatenate the contents of verified execution workspace files."""
        parts: list[str] = []
        for entry in self.workspace_files:
            content = entry.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif content is not None:
                parts.append(json.dumps(content, default=str))
        return "\n".join(parts)

    def execution_text(self) -> str:
        """
        Concatenate all REAL execution evidence: tool execution output and
        errors, run errors, and verified workspace file contents.

        LLM-authored text (``final_answer``, planner output, reasoning, or
        explanations) is deliberately excluded from this string.
        """
        parts = [self.tool_output_text(), self.workspace_evidence_text()]
        parts.extend(self.errors or [])
        return "\n".join(parts)

    def all_text(self) -> str:
        """Concatenate every string-bearing piece of evidence."""
        parts = [self.execution_text()]
        if self.final_answer:
            parts.append(self.final_answer)
        return "\n".join(parts)

    def contains_flag(self, expected_flag: str) -> bool:
        """
        Whether *expected_flag* appears in REAL execution evidence: tool
        execution output, verified workspace files, or execution errors.

        A flag that only appears in the ``final_answer`` (or any other
        LLM-authored text) does NOT count as discovered.
        """
        if not expected_flag:
            return False
        return expected_flag.lower() in self.execution_text().lower()

    def has_tool(self, *names: str, success: bool | None = None) -> bool:
        """Whether any call used one of *names* (optionally filtering by success)."""
        return any(
            call.get("tool_name") in names
            and (success is None or call.get("success") == success)
            for call in self.tool_calls
        )

    def count_tool(self, *names: str, success: bool | None = None) -> int:
        """Count calls that used one of *names* (optionally filtered by success)."""
        return sum(
            1
            for call in self.tool_calls
            if call.get("tool_name") in names
            and (success is None or call.get("success") == success)
        )

    def succeeded(self, *names: str) -> bool:
        """Whether a successful call used one of *names*."""
        return self.has_tool(*names, success=True)


# ── Challenge definition ──────────────────────────────────────────


@dataclass
class Milestone:
    """
    One expected stage of a challenge's solution path.

    ``check`` receives the :class:`RunTrace` and returns whether the agent
    demonstrated that stage.  Milestones are diagnostic: they pin down the
    *first* intended capability the agent failed to show, which becomes the
    reported ``failed_step``.
    """

    name: str
    check: Callable[[RunTrace], bool]


@dataclass
class Challenge:
    """
    A fully specified CTF-style benchmark challenge.

    Parameters
    ----------
    name : str
        Unique challenge identifier.
    category : str
        Challenge family (``"encoding"``, ``"archive"``,
        ``"file_analysis"``, ``"multi_step"``).
    description : str
        Human-readable summary of the challenge.
    difficulty : str
        ``"easy"``, ``"medium"`` or ``"hard"``.
    goal : str
        The task description handed to the agent verbatim.
    expected_flag : str
        Ground-truth flag.  A run passes iff this appears in real
        execution evidence (tool output, verified workspace files, or
        execution errors) — never in the final answer or other LLM text.
    required_capabilities : list[str]
        Tool names the agent is allowed to use.
    files : dict[str, bytes]
        Challenge payloads, keyed by workspace-relative path.
    milestones : list[Milestone]
        Ordered expected solution stages, used for ``failed_step``.
    """

    name: str
    category: str
    description: str
    difficulty: str
    goal: str
    expected_flag: str
    required_capabilities: list[str] = field(default_factory=list)
    files: dict[str, bytes] = field(default_factory=dict)
    milestones: list[Milestone] = field(default_factory=list)

    @property
    def milestone_names(self) -> list[str]:
        return [m.name for m in self.milestones]


# ── Per-challenge result ──────────────────────────────────────────


@dataclass
class ChallengeResult:
    """Outcome of a single challenge run."""

    name: str
    category: str
    difficulty: str
    passed: bool
    flag_found: bool
    status: str
    failed_step: str | None
    milestones: list[dict[str, Any]]
    tools_used: list[str]
    execution_time_ms: float
    iterations: int
    errors: list[str]
    expected_flag: str
    final_answer: str
    llm_unavailable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "difficulty": self.difficulty,
            "passed": self.passed,
            "flag_found": self.flag_found,
            "status": self.status,
            "failed_step": self.failed_step,
            "milestones": self.milestones,
            "tools_used": self.tools_used,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "iterations": self.iterations,
            "errors": self.errors,
            "expected_flag": self.expected_flag,
            "final_answer": self.final_answer[:500],
            "llm_unavailable": self.llm_unavailable,
        }


# ── Evaluation ────────────────────────────────────────────────────


def evaluate_challenge(
    challenge: Challenge,
    trace: RunTrace,
) -> tuple[bool, bool, list[dict[str, Any]], str | None]:
    """
    Evaluate a run against a challenge.

    Returns
    -------
    (passed, flag_found, milestone_results, failed_step)

    * ``passed`` — the expected flag appeared in REAL execution evidence
      (tool output, verified workspace files, or execution errors).  The
      ``final_answer`` is never considered evidence.
    * ``flag_found`` — same as ``passed`` but kept explicit.
    * ``milestone_results`` — ``[{"name", "passed"}]`` for diagnostics.
    * ``failed_step`` — first milestone not demonstrated, or ``None`` when
      the flag was recovered (passing always clears the failure report).
    """
    flag_found = trace.contains_flag(challenge.expected_flag)

    milestone_results: list[dict[str, Any]] = []
    failed_step: str | None = None
    for milestone in challenge.milestones:
        passed = milestone.check(trace)
        milestone_results.append({"name": milestone.name, "passed": passed})
        if failed_step is None and not passed:
            failed_step = milestone.name

    if flag_found:
        failed_step = None

    return flag_found, flag_found, milestone_results, failed_step


# ── Aggregate report ──────────────────────────────────────────────


@dataclass
class BenchmarkReport:
    """Aggregated results across every challenge in a benchmark run."""

    results: list[ChallengeResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def llm_unavailable_count(self) -> int:
        return sum(1 for r in self.results if r.llm_unavailable)

    @property
    def total_time_ms(self) -> float:
        return sum(r.execution_time_ms for r in self.results)

    @property
    def total_iterations(self) -> int:
        return sum(r.iterations for r in self.results)

    @property
    def all_tools_used(self) -> list[str]:
        """Every distinct tool used across all challenges, in first-use order."""
        seen: list[str] = []
        for result in self.results:
            for tool in result.tools_used:
                if tool not in seen:
                    seen.append(tool)
        return seen

    def tool_frequency(self) -> list[dict[str, Any]]:
        """Count of challenge-level uses per tool."""
        counts: dict[str, int] = {}
        for result in self.results:
            for tool in set(result.tools_used):
                counts[tool] = counts.get(tool, 0) + 1
        return [
            {"tool": tool, "challenges": count}
            for tool, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_challenges": len(self.results),
            "passed": self.passed_count,
            "failed": len(self.results) - self.passed_count - self.llm_unavailable_count,
            "llm_unavailable": self.llm_unavailable_count,
            "success_rate": round(self.success_rate, 3),
            "total_time_ms": round(self.total_time_ms, 2),
            "total_iterations": self.total_iterations,
            "tools_used": self.all_tools_used,
            "tool_frequency": self.tool_frequency(),
            "challenges": [r.to_dict() for r in self.results],
        }
