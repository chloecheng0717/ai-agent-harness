"""
Unit Tests for Deterministic Tool Selection

Covers the string-extraction intent rules added for the hidden-text
benchmarks:

- ``strings`` is preferred for steps that extract hidden/embedded text
  from files, logs, or binaries, but only when the LLM plan left the step
  without an explicit tool.
- An explicit ``tool_name`` on a plan step is always honoured and is
  never overridden by keyword-based intent.
- ``file`` stays the choice for metadata/type inspection.
- Ordinary steps (decoding, archiving, computation) are never misrouted.
"""

from app.agents.core.tool_selector import ToolSelector
from app.models.state import PlanStep

_AVAILABLE = ["archive", "encoding", "strings", "file", "calculator", "python"]


class TestStringExtractionSelection:
    def test_selects_strings_for_hidden_text_in_binary(self) -> None:
        step = PlanStep(
            index=0,
            description="Extract the hidden text from the binary file",
            rationale="The flag is embedded in the binary.",
        )
        assert ToolSelector().select_for_step(step, _AVAILABLE) == "strings"

    def test_explicit_file_tool_is_kept_for_string_extraction(self) -> None:
        """An explicit 'file' tool survives a hidden-string-extraction step."""
        step = PlanStep(
            index=0,
            description="Find the flag hidden among the log lines",
            tool_name="file",
            rationale="Scan the log for embedded strings.",
        )
        assert ToolSelector().select_for_step(step, _AVAILABLE) == "file"

    def test_explicit_python_tool_is_kept_for_string_extraction(self) -> None:
        step = PlanStep(
            index=0,
            description="Locate the hidden secret inside the file",
            tool_name="python",
            rationale="Search the file for embedded printable strings.",
        )
        assert ToolSelector().select_for_step(step, _AVAILABLE) == "python"

    def test_explicit_python_tool_kept_with_parameters(self) -> None:
        """Regression: intent must never discard an explicit tool + params."""
        step = PlanStep(
            index=0,
            description="Recover the flag hidden inside the binary file",
            tool_name="python",
            tool_parameters={"code": "print(open('f','rb').read())"},
            rationale="Extract hidden printable strings from the binary.",
        )
        assert ToolSelector().select_for_step(step, _AVAILABLE) == "python"

    def test_strings_still_chosen_when_no_explicit_tool(self) -> None:
        """Without an explicit tool, string-extraction intent wins."""
        step = PlanStep(
            index=0,
            description="Find the flag hidden among the log lines",
            rationale="Scan the log for embedded strings.",
        )
        assert ToolSelector().select_for_step(step, _AVAILABLE) == "strings"

    def test_selects_strings_for_hidden_flag_extraction(self) -> None:
        step = PlanStep(
            index=0,
            description="Recover the flag hidden inside the log",
        )
        assert ToolSelector().select_for_step(step, _AVAILABLE) == "strings"

    def test_selects_strings_for_printable_ascii_dump(self) -> None:
        step = PlanStep(
            index=0,
            description="Dump the printable ascii bytes of the file",
        )
        assert ToolSelector().select_for_step(step, _AVAILABLE) == "strings"


class TestFileInspectionSelection:
    def test_selects_file_for_type_inspection(self) -> None:
        step = PlanStep(
            index=0,
            description="Identify the file type of payload.bin",
        )
        assert ToolSelector().select_for_step(step, _AVAILABLE) == "file"

    def test_selects_file_for_mime_inspection(self) -> None:
        step = PlanStep(
            index=0,
            description="Check the mime type and magic bytes of the file",
        )
        assert ToolSelector().select_for_step(step, _AVAILABLE) == "file"

    def test_keeps_explicit_file_tool_for_metadata(self) -> None:
        step = PlanStep(
            index=0,
            description="Sniff the file header signature",
            tool_name="file",
        )
        assert ToolSelector().select_for_step(step, _AVAILABLE) == "file"


class TestNoMisrouting:
    def test_decoding_steps_still_use_encoding(self) -> None:
        step = PlanStep(
            index=0,
            description="Decode the base64 data layer by layer",
            rationale="Unwrap the encoding.",
        )
        assert ToolSelector().select_for_step(step, _AVAILABLE) == "encoding"

    def test_archive_steps_still_use_archive(self) -> None:
        step = PlanStep(
            index=0,
            description="Extract the compressed zip archive",
        )
        assert ToolSelector().select_for_step(step, _AVAILABLE) == "archive"

    def test_explicit_dedicated_tool_is_honoured(self) -> None:
        step = PlanStep(
            index=0,
            description="Extract the zip archive",
            tool_name="archive",
        )
        assert ToolSelector().select_for_step(step, _AVAILABLE) == "archive"
