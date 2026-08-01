"""
File Analysis Challenges

Test the agent's ability to sniff a file's type with the ``file`` tool
and extract hidden printable strings with the ``strings`` tool to uncover
a flag embedded in binary content.
"""

from __future__ import annotations

from app.benchmarks.challenges.helpers import binary_blob, log_blob
from app.benchmarks.models import Challenge, Milestone


def build_file_analysis_strings() -> Challenge:
    """Recover a flag embedded as a printable string inside a binary blob."""
    flag = "flag{needle_in_haystack}"
    blob = binary_blob(
        entries=[
            b"project_alpha",
            b"0xdeadbeef",
            b"SECRET_KEY=7f9c2b",
            flag.encode("ascii"),
            b"https://example.local/endpoint",
        ]
    )
    return Challenge(
        name="file_analysis_strings",
        category="file_analysis",
        description=(
            "A binary file with a flag embedded among other printable "
            "strings. The agent must identify the file and extract strings."
        ),
        difficulty="easy",
        goal=(
            "A flag is hidden inside the binary file /workspace/payload.bin. "
            "Analyse the file and extract the flag from its strings."
        ),
        expected_flag=flag,
        required_capabilities=["file", "strings", "python", "encoding"],
        files={"payload.bin": blob},
        milestones=[
            Milestone(
                "Inspect the file",
                lambda t: t.succeeded("file", "strings", "python")
                or t.contains_flag(flag),
            ),
            Milestone(
                "Extract printable strings",
                lambda t: t.count_tool("strings", success=True) >= 1
                or t.contains_flag(flag),
            ),
            Milestone(
                "Recover the flag",
                lambda t: t.contains_flag(flag),
            ),
        ],
    )


def build_file_analysis_hidden() -> Challenge:
    """Find a flag hidden among decoy tokens in a binary log file."""
    flag = "flag{hidden_in_plain_sight}"
    blob = log_blob(
        [
            "2026-07-21 08:11:02 [INFO] scheduler tick 4521",
            "2026-07-21 08:11:03 [INFO] worker pool healthy",
            "SESSION_ID=9a81f7c3-2d54-4b0e-8f1a-7c9d0e2f5b41",
            "TOKEN=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzZXJ2aWNlIn0",
            "credential_backup=flag{hidden_in_plain_sight}",
            "2026-07-21 08:11:04 [INFO] sync complete",
            "PING 1.2.3.4 seq=7 ttl=64 time=12.3 ms",
        ]
    )
    return Challenge(
        name="file_analysis_hidden",
        category="file_analysis",
        description=(
            "A binary log file where the flag is hidden among decoy tokens. "
            "The agent must scan the file and pick the real flag out of the noise."
        ),
        difficulty="medium",
        goal=(
            "A flag is hidden somewhere inside /workspace/logs.bin among many "
            "decoy tokens. Analyse the file, extract its strings, and report "
            "the real flag."
        ),
        expected_flag=flag,
        required_capabilities=["file", "strings", "python", "encoding"],
        files={"logs.bin": blob},
        milestones=[
            Milestone(
                "Inspect the file",
                lambda t: t.succeeded("file", "strings", "python")
                or t.contains_flag(flag),
            ),
            Milestone(
                "Extract strings from the log",
                lambda t: t.count_tool("strings", success=True) >= 1
                or t.contains_flag(flag),
            ),
            Milestone(
                "Recover the flag",
                lambda t: t.contains_flag(flag),
            ),
        ],
    )
