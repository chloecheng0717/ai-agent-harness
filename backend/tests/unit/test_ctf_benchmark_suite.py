"""
Unit Tests for the CTF Benchmark Suite

Covers the framework-agnostic layers of the benchmark:

- Challenge file builders (helpers) and reference solvability of every
  challenge in the registry.
- ``evaluate_challenge`` pass/fail and "failed step" milestone logic.
- ``RunTrace`` evidence helpers.
- ``ChallengeResult`` / ``BenchmarkReport`` aggregation and rendering.

No sandbox, tools, or LLM provider is required here.
"""

import base64
import io
import json
import re
import tarfile
import zipfile

import pytest
from app.benchmarks.challenges import (
    build_all_challenges,
    challenge_categories,
    get_challenge,
)
from app.benchmarks.challenges.helpers import (
    b64,
    binary_blob,
    from_b64,
    from_hex,
    hexstr,
    log_blob,
    make_tar_gz,
    make_zip,
)
from app.benchmarks.models import (
    BenchmarkReport,
    Challenge,
    ChallengeResult,
    Milestone,
    RunTrace,
    evaluate_challenge,
)

FLAG_PATTERN = re.compile(r"(?:flag|ctf)\{[^\}]+\}", re.IGNORECASE)


# ── Reference solvers ─────────────────────────────────────────────


def _printable_strings(data: bytes) -> list[str]:
    return [s.decode("ascii", errors="replace") for s in re.findall(rb"[ -~]{4,}", data)]


def _reference_flag(challenge: Challenge) -> str:
    """Recover a challenge's flag from its files using a hand-written solver."""
    if challenge.name == "encoding_base64":
        return from_b64(challenge.files["message.txt"].decode("ascii"))
    if challenge.name == "encoding_multilayer":
        first = from_b64(challenge.files["deep.txt"].decode("ascii"))
        return from_hex(first)
    if challenge.name == "archive_zip":
        with zipfile.ZipFile(io.BytesIO(challenge.files["challenge.zip"])) as z:
            return z.read("flag.txt").decode("utf-8")
    if challenge.name == "archive_nested":
        with tarfile.open(fileobj=io.BytesIO(challenge.files["challenge.tar.gz"]), mode="r:gz") as t:
            inner = t.extractfile("inner/package.zip").read()
        with zipfile.ZipFile(io.BytesIO(inner)) as z:
            return z.read("flag.txt").decode("utf-8")
    if challenge.name == "file_analysis_strings":
        joined = "\n".join(_printable_strings(challenge.files["payload.bin"]))
        return FLAG_PATTERN.search(joined).group(0)  # type: ignore[union-attr]
    if challenge.name == "file_analysis_hidden":
        joined = "\n".join(_printable_strings(challenge.files["logs.bin"]))
        return FLAG_PATTERN.search(joined).group(0)  # type: ignore[union-attr]
    if challenge.name == "multi_step":
        with zipfile.ZipFile(io.BytesIO(challenge.files["challenge.zip"])) as z:
            payload = z.read("payload.bin")
        tokens = re.findall(rb"[A-Za-z0-9+/]{12,}={0,2}", payload)
        for token in tokens:
            try:
                decoded = base64.b64decode(token).decode("utf-8")
            except Exception:
                continue
            if "flag{" in decoded:
                return decoded
        raise AssertionError("multi_step flag not recoverable from payload strings")
    raise AssertionError(f"no reference solver for {challenge.name}")


# ── Helper builders ───────────────────────────────────────────────


class TestHelpers:
    def test_b64_hex_round_trip(self) -> None:
        payload = "flag{roundtrip}"
        assert from_b64(b64(payload)) == payload
        assert from_hex(hexstr(payload)) == payload

    def test_make_zip_round_trip(self) -> None:
        raw = make_zip({"a.txt": "hello", "b.bin": b"\x00\x01\x02"})
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            assert z.read("a.txt") == b"hello"
            assert z.read("b.bin") == b"\x00\x01\x02"

    def test_make_tar_gz_round_trip(self) -> None:
        raw = make_tar_gz({"nested/flag.txt": "flag{tar}"})
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as t:
            member = t.extractfile("nested/flag.txt")
            assert member is not None
            assert member.read() == b"flag{tar}"

    def test_binary_blob_embeds_entries_deterministically(self) -> None:
        entries = [b"version=1.0", b"flag{blob}"]
        first = binary_blob(entries, filler_seed=0)
        second = binary_blob(entries, filler_seed=0)
        assert first == second
        assert first.startswith(b"\x7fBENCH\x01")
        joined = "\n".join(_printable_strings(first))
        assert "version=1.0" in joined
        assert "flag{blob}" in joined

    def test_log_blob_shape(self) -> None:
        raw = log_blob(["line one", "line two"])
        assert raw.startswith(b"\x42\x45\x4e\x43\n")
        joined = "\n".join(_printable_strings(raw))
        assert "line one" in joined
        assert "line two" in joined


# ── Challenge registry & solvability ──────────────────────────────


class TestChallengeRegistry:
    def test_build_all_returns_seven_unique_challenges(self) -> None:
        challenges = build_all_challenges()
        assert len(challenges) == 7
        names = [c.name for c in challenges]
        assert len(set(names)) == 7
        for challenge in challenges:
            assert challenge.name
            assert challenge.category
            assert challenge.goal
            assert challenge.difficulty in ("easy", "medium", "hard")
            assert challenge.expected_flag
            assert challenge.files
            assert challenge.milestones

    def test_all_flags_match_flag_pattern(self) -> None:
        for challenge in build_all_challenges():
            assert FLAG_PATTERN.fullmatch(challenge.expected_flag), challenge.name

    def test_challenge_categories(self) -> None:
        assert challenge_categories() == ["encoding", "archive", "file_analysis", "multi_step"]

    def test_get_challenge_lookup(self) -> None:
        assert get_challenge("encoding_base64").name == "encoding_base64"
        with pytest.raises(KeyError):
            get_challenge("does_not_exist")

    @pytest.mark.parametrize(
        "name",
        [c.name for c in build_all_challenges()],
    )
    def test_reference_solver_recovers_flag(self, name: str) -> None:
        challenge = get_challenge(name)
        assert _reference_flag(challenge) == challenge.expected_flag


# ── RunTrace evidence helpers ─────────────────────────────────────


class TestRunTrace:
    def test_contains_flag_case_insensitive(self) -> None:
        trace = RunTrace(final_answer="the answer is FLAG{Upper}")
        assert trace.contains_flag("flag{upper}")
        assert not trace.contains_flag("flag{nope}")

    def test_all_text_collects_outputs_and_errors(self) -> None:
        trace = RunTrace(
            tool_calls=[
                {"tool_name": "python", "success": True, "output": "hello"},
                {"tool_name": "strings", "success": False, "error": "boom"},
            ],
            errors=["top-level error"],
        )
        text = trace.all_text()
        assert "hello" in text
        assert "boom" in text
        assert "top-level error" in text

    def test_has_and_count_tool_with_success_filter(self) -> None:
        trace = RunTrace(
            tool_calls=[
                {"tool_name": "python", "success": True},
                {"tool_name": "python", "success": False},
                {"tool_name": "strings", "success": True},
            ]
        )
        assert trace.has_tool("python", "strings")
        assert trace.has_tool("python", success=True)
        assert not trace.has_tool("archive")
        assert trace.count_tool("python") == 2
        assert trace.count_tool("python", success=True) == 1
        assert trace.succeeded("python", "strings")


# ── evaluate_challenge ────────────────────────────────────────────


class TestEvaluateChallenge:
    def test_flag_in_final_answer_passes_and_clears_failed_step(self) -> None:
        challenge = get_challenge("encoding_base64")
        trace = RunTrace(final_answer=challenge.expected_flag)
        passed, flag_found, milestones, failed_step = evaluate_challenge(challenge, trace)
        assert passed is True
        assert flag_found is True
        assert all(m["passed"] for m in milestones)
        assert failed_step is None

    def test_flag_in_tool_output_alone_passes(self) -> None:
        challenge = get_challenge("encoding_base64")
        trace = RunTrace(
            tool_calls=[
                {
                    "step_index": 1,
                    "tool_name": "python",
                    "success": True,
                    "output": {"stdout": challenge.expected_flag},
                }
            ]
        )
        passed, flag_found, _, failed_step = evaluate_challenge(challenge, trace)
        assert passed is True
        assert failed_step is None

    def test_partial_evidence_reports_first_failed_milestone(self) -> None:
        challenge = get_challenge("encoding_base64")
        trace = RunTrace(
            tool_calls=[
                {"tool_name": "strings", "success": True},
                {"tool_name": "python", "success": True, "output": "still encoded"},
            ]
        )
        passed, flag_found, milestones, failed_step = evaluate_challenge(challenge, trace)
        assert passed is False
        assert flag_found is False
        assert [m["passed"] for m in milestones] == [True, True, False]
        assert failed_step == "Recover the flag"

    def test_empty_trace_fails_on_first_milestone(self) -> None:
        challenge = get_challenge("archive_zip")
        passed, _, _, failed_step = evaluate_challenge(challenge, RunTrace())
        assert passed is False
        assert failed_step == "Inspect the challenge file"


# ── ChallengeResult / BenchmarkReport ─────────────────────────────


def _make_result(
    name: str,
    passed: bool,
    tools: list[str] | None = None,
    time_ms: float = 100.0,
) -> ChallengeResult:
    return ChallengeResult(
        name=name,
        category="encoding",
        difficulty="easy",
        passed=passed,
        flag_found=passed,
        status="completed",
        failed_step=None,
        milestones=[],
        tools_used=tools or [],
        execution_time_ms=time_ms,
        iterations=1,
        errors=[],
        expected_flag="flag{x}",
        final_answer="",
    )


class TestChallengeResult:
    def test_to_dict_shape(self) -> None:
        data = _make_result("encoding_base64", True, ["python"], 123.456).to_dict()
        assert data["name"] == "encoding_base64"
        assert data["passed"] is True
        assert data["execution_time_ms"] == 123.46
        assert data["tools_used"] == ["python"]


class TestBenchmarkReport:
    def test_empty_report(self) -> None:
        report = BenchmarkReport()
        assert report.success_rate == 0.0
        assert report.passed_count == 0
        assert report.total_time_ms == 0.0
        assert report.total_iterations == 0
        assert report.all_tools_used == []

    def test_aggregation(self) -> None:
        report = BenchmarkReport(
            results=[
                _make_result("a", True, ["python"], 10.0),
                _make_result("b", False, ["strings"], 20.0),
                _make_result("c", True, ["archive", "python"], 30.0),
            ]
        )
        assert report.success_rate == 2 / 3
        assert report.passed_count == 2
        assert report.total_time_ms == 60.0
        assert report.total_iterations == 3
        assert report.all_tools_used == ["python", "strings", "archive"]

    def test_tool_frequency_orders_and_dedupes(self) -> None:
        report = BenchmarkReport(
            results=[
                _make_result("a", True, ["python", "python", "archive"]),
                _make_result("b", True, ["python"]),
                _make_result("c", False, ["strings"]),
            ]
        )
        freq = report.tool_frequency()
        assert freq[0] == {"tool": "python", "challenges": 2}
        assert freq[1] == {"tool": "archive", "challenges": 1}
        assert freq[2] == {"tool": "strings", "challenges": 1}

    def test_to_dict_aggregate(self) -> None:
        report = BenchmarkReport(results=[_make_result("a", True), _make_result("b", False)])
        data = report.to_dict()
        assert data["total_challenges"] == 2
        assert data["passed"] == 1
        assert data["failed"] == 1
        assert data["success_rate"] == 0.5
        assert len(data["challenges"]) == 2


# ── Milestone plumbing (diagnostic) ───────────────────────────────


class TestMilestone:
    def test_milestone_check_receives_trace(self) -> None:
        seen: list[RunTrace] = []
        milestone = Milestone("probe", lambda t: seen.append(t) or True)
        trace = RunTrace()
        assert milestone.check(trace) is True
        assert seen == [trace]


# ── report.py rendering ───────────────────────────────────────────


class TestReportRendering:
    def test_render_result_table_lists_rows(self) -> None:
        from app.benchmarks.report import render_result_table

        table = render_result_table(
            [
                _make_result("encoding_base64", True, ["python"], 123.0),
                _make_result("archive_zip", False, ["file"], 45.0),
            ]
        )
        assert "challenge" in table
        assert "encoding_base64" in table
        assert "PASS" in table
        assert "FAIL" in table
        assert "123" in table

    def test_render_summary_counts(self) -> None:
        from app.benchmarks.report import render_summary

        report = BenchmarkReport(results=[_make_result("a", True), _make_result("b", False)])
        text = render_summary(report)
        assert "Challenges:" in text
        assert "Passed:" in text
        assert "50%" in text

    def test_report_to_json_serializes(self) -> None:
        from app.benchmarks.report import report_to_json

        report = BenchmarkReport(results=[_make_result("a", True)])
        data = json.loads(report_to_json(report))
        assert data["total_challenges"] == 1
        assert data["passed"] == 1

    def test_failed_step_summary_lists_stuck_challenges(self) -> None:
        from app.benchmarks.report import failed_step_summary

        passed = _make_result("ok", True)
        failed = _make_result("bad", False)
        failed.milestones = [{"name": "Read", "passed": True}, {"name": "Decode", "passed": False}]
        failed.failed_step = "Decode"
        summary = failed_step_summary([passed, failed])
        assert len(summary) == 1
        assert summary[0]["challenge"] == "bad"
        assert summary[0]["failed_step"] == "Decode"
        assert summary[0]["milestones_reached"] == ["Read"]
