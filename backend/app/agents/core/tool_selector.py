"""
Deterministic Tool Selector

Selects the best available tool for a plan step without invoking an LLM.
Matching is keyword-based and fully deterministic so that identical inputs
always produce identical selections.

Design rules
------------
* An explicit ``tool_name`` already present on a step is honoured whenever
  it is available.
* Otherwise ``estimated_tools`` are consulted before keyword matching.
* Dedicated tools (archive, encoding, strings, file, calculator) are
  preferred over generic fallbacks (python, web_search).
* Ties are broken by a fixed priority ordering, never by hash order.
"""

from collections.abc import Sequence
from typing import Any

import structlog

# Keyword → tool name.  Matching is a lowercase substring test against the
# step's description / rationale / expected_output combined text.
_TOOL_KEYWORDS: dict[str, list[str]] = {
    "archive": [
        "archive",
        "extract",
        "unzip",
        "zip",
        "tar",
        "gzip",
        "gz",
        "7z",
        "decompress",
        "compressed",
    ],
    "encoding": [
        "base64",
        "hex",
        "url",
        "encode",
        "decode",
        "encoding",
        "encoded",
        "decoded",
        "cipher",
        "rot13",
    ],
    "strings": [
        "strings",
        "binary",
        "printable",
        "ascii",
        "bytes",
        "dump",
    ],
    "file": [
        "file type",
        "identify",
        "mime",
        "filetype",
        "magic bytes",
        "header",
        "signature",
    ],
    "calculator": [
        "calculate",
        "calculator",
        "arithmetic",
        "math",
        "expression",
        "compute",
        "sum",
    ],
    "web_search": [
        "search",
        "web",
        "internet",
        "google",
        "look up",
        "query",
    ],
    "python": [
        "python",
        "script",
        "write code",
        "run code",
        "execute code",
        "custom logic",
    ],
}

# Tools that should never be the first choice when a dedicated tool matches.
# Keeping this small ensures dedicated tools win keyword ties.
_GENERIC_TOOLS = {"python", "web_search"}

# Fixed priority ordering used to break score ties deterministically.
_TOOL_PRIORITY: list[str] = [
    "archive",
    "encoding",
    "strings",
    "file",
    "calculator",
    "web_search",
    "python",
]


class ToolSelector:
    """Select the best available tool for a plan step, deterministically."""

    def __init__(self) -> None:
        self.logger = structlog.get_logger().bind(component="tool_selector")

    def select_for_step(
        self,
        step: Any,
        available_tools: Sequence[str],
    ) -> str | None:
        """
        Return the best tool name for *step*, or ``None``.

        Parameters
        ----------
        step : PlanStep | dict
            A plan step carrying ``tool_name``, ``estimated_tools``,
            ``description``, ``rationale`` and ``expected_output``.
        available_tools : sequence[str]
            Tool names that may be selected.

        Returns
        -------
        str | None
            The best matching tool name, or ``None`` when no tool fits.
        """
        available = list(available_tools)

        # 1. Honour an explicit, available tool name.
        explicit = self._step_field(step, "tool_name")
        if isinstance(explicit, str) and explicit in available:
            return explicit

        # 2. Honour an estimated tool that is actually available.
        for estimated in self._step_field(step, "estimated_tools", default=[]):
            if isinstance(estimated, str) and estimated in available:
                return estimated

        # 3. Keyword scoring against the step's descriptive text.
        text = self._step_text(step).lower()
        best: str | None = None
        best_score = 0.0
        for tool in _TOOL_PRIORITY:
            if tool not in available:
                continue
            keywords = _TOOL_KEYWORDS.get(tool, [])
            score = sum(1 for kw in keywords if kw in text)
            weighted = score * 0.5 if tool in _GENERIC_TOOLS else float(score)
            if weighted > best_score:
                best = tool
                best_score = weighted

        if best is not None and best_score > 0:
            self.logger.debug(
                "Selected tool for step",
                tool_name=best,
                score=best_score,
            )
            return best

        return None

    def select_for_plan(
        self,
        steps: Sequence[Any],
        available_tools: Sequence[str],
    ) -> list[str | None]:
        """Select a tool for every step in a plan, preserving order."""
        return [self.select_for_step(step, available_tools) for step in steps]

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _step_field(step: Any, field: str, default: Any = None) -> Any:
        if isinstance(step, dict):
            return step.get(field, default)
        return getattr(step, field, default)

    @classmethod
    def _step_text(cls, step: Any) -> str:
        description = cls._step_field(step, "description", "") or ""
        rationale = cls._step_field(step, "rationale", "") or ""
        expected_output = cls._step_field(step, "expected_output", "") or ""
        return " ".join(str(part) for part in (description, rationale, expected_output))
