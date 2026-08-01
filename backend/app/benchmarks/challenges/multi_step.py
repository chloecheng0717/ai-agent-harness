"""
Multi-Step Challenge

A full workflow that exercises the whole agent pipeline end to end:
extract an archive, analyse the extracted payload, decode its encoded
content, and recover the flag.
"""

from __future__ import annotations

from app.benchmarks.challenges.helpers import b64, binary_blob, make_zip
from app.benchmarks.models import Challenge, Milestone


def build_multi_step() -> Challenge:
    """
    Flag = archive → binary payload → base64-encoded text → flag.

    Steps the agent must complete:
    1. Extract the ZIP archive.
    2. Analyse the extracted ``payload.bin`` (binary) and extract its strings.
    3. Decode the base64 string found in the payload.
    4. Recover the flag.
    """
    flag = "flag{multi_stage_pipeline}"
    encoded = b64(flag)
    payload = binary_blob(
        entries=[
            b"version=1.4.2",
            b"build_id=alpha-77",
            encoded.encode("ascii"),
            b"checksum=0x1f2e3d",
        ]
    )
    return Challenge(
        name="multi_step",
        category="multi_step",
        description=(
            "A multi-stage workflow: a ZIP archive contains a binary payload "
            "whose strings hide a base64-encoded flag. The agent must extract, "
            "analyse, decode, and report the flag."
        ),
        difficulty="hard",
        goal=(
            "Recover the flag from /workspace/challenge.zip. The archive holds "
            "a binary payload; one of its strings is a base64-encoded flag. "
            "Extract the archive, analyse the payload, decode the encoded "
            "content, and report the flag."
        ),
        expected_flag=flag,
        required_capabilities=[
            "file",
            "archive",
            "strings",
            "python",
            "encoding",
        ],
        files={
            "challenge.zip": make_zip(
                {
                    "hint.txt": (
                        "The flag lives inside payload.bin, encoded in base64."
                    ),
                    "payload.bin": payload,
                }
            )
        },
        milestones=[
            Milestone(
                "Extract the archive",
                lambda t: t.count_tool("archive", success=True) >= 1
                or t.contains_flag(flag),
            ),
            Milestone(
                "Analyse the payload file",
                lambda t: t.succeeded("file", "strings", "python")
                or t.contains_flag(flag),
            ),
            Milestone(
                "Decode the encoded content",
                lambda t: t.count_tool("encoding", "python", success=True) >= 1
                or t.contains_flag(flag),
            ),
            Milestone(
                "Recover the flag",
                lambda t: t.contains_flag(flag),
            ),
        ],
    )
