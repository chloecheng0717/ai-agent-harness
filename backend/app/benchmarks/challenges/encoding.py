"""
Encoding Challenges

Test the agent's ability to detect and reverse encoding layers using the
``encoding`` tool (or ``python``): single base64 and stacked base64+hex.
"""

from __future__ import annotations

from app.benchmarks.challenges.helpers import b64, hexstr
from app.benchmarks.models import Challenge, Milestone, RunTrace


def _decode_tool_calls(trace: RunTrace) -> int:
    """Count successful decoding-capable tool calls."""
    return trace.count_tool("encoding", "python", success=True)


def build_encoding_base64() -> Challenge:
    """Base64-decode a single-layer message to recover the flag."""
    flag = "flag{base64_decode_warmup}"
    return Challenge(
        name="encoding_base64",
        category="encoding",
        description=(
            "A single-layer base64 message. The agent must read the file, "
            "recognise base64, decode it, and report the flag."
        ),
        difficulty="easy",
        goal=(
            "The file /workspace/message.txt contains a base64-encoded flag. "
            "Decode it and report the recovered flag."
        ),
        expected_flag=flag,
        required_capabilities=["file", "strings", "python", "encoding"],
        files={"message.txt": b64(flag).encode("ascii")},
        milestones=[
            Milestone(
                "Read the challenge file",
                lambda t: t.succeeded("python", "strings", "file")
                or t.contains_flag(flag),
            ),
            Milestone(
                "Decode base64",
                lambda t: _decode_tool_calls(t) >= 1 or t.contains_flag(flag),
            ),
            Milestone(
                "Recover the flag",
                lambda t: t.contains_flag(flag),
            ),
        ],
    )


def build_encoding_multilayer() -> Challenge:
    """Recover a flag wrapped in two stacked encodings (base64 then hex)."""
    flag = "flag{layers_of_encoding}"
    # outer base64(text) of the hex(text) representation of the flag.
    outer = b64(hexstr(flag))
    return Challenge(
        name="encoding_multilayer",
        category="encoding",
        description=(
            "A two-layer encoding (base64 of hex). The agent must detect and "
            "reverse each layer in order before the flag is revealed."
        ),
        difficulty="medium",
        goal=(
            "The file /workspace/deep.txt contains a flag wrapped in multiple "
            "encoding layers. Decode every layer, one at a time, until you "
            "recover the flag, then report it."
        ),
        expected_flag=flag,
        required_capabilities=["file", "strings", "python", "encoding"],
        files={"deep.txt": outer.encode("ascii")},
        milestones=[
            Milestone(
                "Read the challenge file",
                lambda t: t.succeeded("python", "strings", "file")
                or t.contains_flag(flag),
            ),
            Milestone(
                "Decode the first layer",
                lambda t: _decode_tool_calls(t) >= 1 or t.contains_flag(flag),
            ),
            Milestone(
                "Decode the second layer",
                lambda t: _decode_tool_calls(t) >= 2 or t.contains_flag(flag),
            ),
            Milestone(
                "Recover the flag",
                lambda t: t.contains_flag(flag),
            ),
        ],
    )
