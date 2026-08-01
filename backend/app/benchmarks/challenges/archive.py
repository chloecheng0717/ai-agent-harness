"""
Archive Challenges

Test the agent's ability to inspect and extract archives (ZIP, and a
tar.gz nesting a ZIP) with the ``archive`` tool, then read the extracted
flag file.
"""

from __future__ import annotations

from app.benchmarks.challenges.helpers import make_tar_gz, make_zip
from app.benchmarks.models import Challenge, Milestone


def build_archive_zip() -> Challenge:
    """Extract a ZIP archive and read the flag file inside it."""
    flag = "flag{zip_me_up}"
    return Challenge(
        name="archive_zip",
        category="archive",
        description=(
            "A ZIP archive containing a hint and a flag file. The agent must "
            "identify the archive, extract it, and read the extracted flag."
        ),
        difficulty="easy",
        goal=(
            "The file /workspace/challenge.zip is an archive containing a "
            "hidden flag. Extract it and report the flag you find inside."
        ),
        expected_flag=flag,
        required_capabilities=["file", "archive", "strings", "python", "encoding"],
        files={
            "challenge.zip": make_zip(
                {
                    "readme.txt": "The flag you seek is in flag.txt.",
                    "flag.txt": flag,
                }
            )
        },
        milestones=[
            Milestone(
                "Inspect the challenge file",
                lambda t: t.succeeded("file", "strings", "python", "archive")
                or t.contains_flag(flag),
            ),
            Milestone(
                "Extract the archive",
                lambda t: t.count_tool("archive", success=True) >= 1
                or t.contains_flag(flag),
            ),
            Milestone(
                "Read the extracted flag file",
                lambda t: t.contains_flag(flag),
            ),
        ],
    )


def build_archive_nested() -> Challenge:
    """Extract a nested archive (tar.gz containing a ZIP) to reach the flag."""
    flag = "flag{twice_wrapped}"
    inner_zip = make_zip({"flag.txt": flag})
    return Challenge(
        name="archive_nested",
        category="archive",
        description=(
            "A tar.gz that contains a nested ZIP archive. The agent must "
            "extract the outer archive, then the inner ZIP, to reach the flag."
        ),
        difficulty="medium",
        goal=(
            "The file /workspace/challenge.tar.gz contains a nested archive. "
            "Extract every archive layer until you reach the flag file, then "
            "report the flag."
        ),
        expected_flag=flag,
        required_capabilities=["file", "archive", "strings", "python", "encoding"],
        files={
            "challenge.tar.gz": make_tar_gz(
                {
                    "hint.txt": "The payload is packed inside inner/package.zip.",
                    "inner/package.zip": inner_zip,
                }
            )
        },
        milestones=[
            Milestone(
                "Inspect the challenge file",
                lambda t: t.succeeded("file", "strings", "python", "archive")
                or t.contains_flag(flag),
            ),
            Milestone(
                "Extract the outer archive",
                lambda t: t.count_tool("archive", success=True) >= 1
                or t.contains_flag(flag),
            ),
            Milestone(
                "Extract the nested archive",
                lambda t: t.count_tool("archive", success=True) >= 2
                or t.contains_flag(flag),
            ),
            Milestone(
                "Recover the flag",
                lambda t: t.contains_flag(flag),
            ),
        ],
    )
