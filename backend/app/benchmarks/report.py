"""
Benchmark Reporting

Formats :class:`ChallengeResult` lists and a :class:`BenchmarkReport`
into human-readable tables and JSON for dashboards or CI.
"""

from __future__ import annotations

import json
from typing import Any

from app.benchmarks.models import BenchmarkReport, ChallengeResult


def _pad(text: str, width: int) -> str:
    """Left-justify *text* to *width*, truncating when needed."""
    return text[:width].ljust(width)


def _row(columns: list[str], widths: list[int]) -> str:
    return "  " + " | ".join(_pad(c, w) for c, w in zip(columns, widths)).rstrip()


def render_result_table(results: list[ChallengeResult]) -> str:
    """Render a per-challenge results table."""
    headers = [
        "challenge",
        "category",
        "difficulty",
        "pass",
        "status",
        "failed step",
        "tools",
        "time (ms)",
        "iters",
    ]
    widths = [22, 13, 10, 6, 11, 32, 26, 10, 6]

    lines = [
        _row(headers, widths),
        _row(["-" * w for w in widths], widths),
    ]
    for r in results:
        tools = ",".join(r.tools_used) or "-"
        failed = r.failed_step or "-"
        lines.append(
            _row(
                [
                    r.name,
                    r.category,
                    r.difficulty,
                    "PASS" if r.passed else "FAIL",
                    r.status,
                    failed,
                    tools,
                    f"{r.execution_time_ms:.0f}",
                    str(r.iterations),
                ],
                widths,
            )
        )
    return "\n".join(lines)


def render_summary(report: BenchmarkReport) -> str:
    """Render the aggregate benchmark summary."""
    lines = [
        "Benchmark summary",
        "================",
        f"  Challenges:       {len(report.results)}",
        f"  Passed:           {report.passed_count}",
        f"  Failed:           {len(report.results) - report.passed_count}",
        f"  Success rate:     {report.success_rate:.0%}",
        f"  Total time:       {report.total_time_ms:.0f} ms",
        f"  Total iterations: {report.total_iterations}",
        "",
        "  Tools used (by number of challenges):",
    ]
    for entry in report.tool_frequency():
        lines.append(
            f"    {entry['tool']:<10} {entry['challenges']} challenge(s)"
        )
    lines.append("")
    return "\n".join(lines)


def render_report(report: BenchmarkReport) -> str:
    """Render the complete report: summary + per-challenge table."""
    return (
        render_summary(report)
        + "\n"
        + render_result_table(report.results)
    )


def report_to_json(report: BenchmarkReport, indent: int = 2) -> str:
    """Serialize the report to JSON."""
    return json.dumps(report.to_dict(), indent=indent, default=str)


def failed_step_summary(results: list[ChallengeResult]) -> list[dict[str, Any]]:
    """Compact list of where each failed challenge got stuck."""
    out: list[dict[str, Any]] = []
    for r in results:
        if r.passed:
            continue
        reached = [
            m["name"] for m in r.milestones if m["passed"]
        ]
        out.append(
            {
                "challenge": r.name,
                "failed_step": r.failed_step,
                "milestones_reached": reached,
                "status": r.status,
            }
        )
    return out
