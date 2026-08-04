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
        "hidden",
        "embedded",
        "secret",
        "log",
        "logs",
    ],
    "file": [
        "file type",
        "filetype",
        "mime",
        "magic bytes",
        "magic",
        "header",
        "signature",
        "type of the file",
        "sniff",
        "identify the file",
        "identify file type",
        "identify filetype",
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
        "socket",
        "network",
        "netcat",
        "tcp",
        "connect",
        "remote service",
        "host:port",
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

#: Tools whose job is metadata inspection or generic execution rather than
#: extracting hidden content.  These yield to ``strings`` when a step's
#: wording shows clear string-extraction intent.
_METADATA_OR_GENERIC_TOOLS = {"file", "python", "web_search"}

#: Wording that marks a step as "pull hidden text out of a file".
_STRING_EXTRACTION_ACTIONS = (
    "strings",
    "printable",
    "ascii",
    "extract",
    "find",
    "recover",
    "locate",
    "search",
    "look",
    "reveal",
    "flag",
)
_STRING_HIDDEN_SIGNALS = (
    "hidden",
    "hide",
    "embed",
    "embedded",
    "secret",
    "obfuscated",
    "concealed",
    "inside",
    "among",
)
_STRING_FILE_SIGNALS = (
    "file",
    "binary",
    "log",
    "logs",
    "blob",
    "bytes",
    "dump",
)


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
        text = self._step_text(step).lower()

        # 0. String-extraction intent picks ``strings`` for steps the LLM
        #    left without a tool.  An explicit ``tool_name`` already present
        #    on the step is always honoured (rule 1) and never overridden.
        if "strings" in available and self._string_extraction_intent(text):
            if self._step_field(step, "tool_name") is None:
                self.logger.debug(
                    "String-extraction intent selected strings tool",
                    step_text=text[:120],
                )
                return "strings"

        # 1. Honour an explicit, available tool name.
        explicit = self._step_field(step, "tool_name")
        if isinstance(explicit, str) and explicit in available:
            return explicit

        # 2. Honour an estimated tool that is actually available.
        for estimated in self._step_field(step, "estimated_tools", default=[]):
            if isinstance(estimated, str) and estimated in available:
                return estimated

        # 3. Keyword scoring against the step's descriptive text.
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

        # 4. Re-check intent against the keyword winner: ``file``/``python``
        #    should not win a step that is really about extracting content.
        if (
            best_score > 0
            and best in _METADATA_OR_GENERIC_TOOLS
            and self._string_extraction_intent(text)
            and "strings" in available
        ):
            return "strings"

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

    @classmethod
    def _string_extraction_intent(cls, text: str) -> bool:
        """
        Whether a step's wording asks to pull hidden text out of a file.

        All three signals must be present so ordinary steps (decoding,
        archiving, type inspection) are never misrouted:
        an extraction action *and* a hidden-content hint *and* a
        file-like target.
        """
        action = any(kw in text for kw in _STRING_EXTRACTION_ACTIONS)
        hidden = any(kw in text for kw in _STRING_HIDDEN_SIGNALS)
        file_like = any(kw in text for kw in _STRING_FILE_SIGNALS)
        return action and hidden and file_like
