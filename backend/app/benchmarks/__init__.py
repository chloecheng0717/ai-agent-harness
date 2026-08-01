"""
CTF Benchmark Evaluation Suite

Local CTF-style challenges that exercise the full agent workflow through
the existing tool chain, plus a runner that measures current agent
capability (success rate, failed step, tools used, execution time, and
iteration count).

Modules
-------
models
    Challenge / Milestone / RunTrace / results + report dataclasses.
challenges
    Deterministic challenge builders (encoding, archive, file analysis,
    multi-step) and the challenge registry.
orchestrator
    BenchmarkOrchestrator: challenge-file injection, tool-parameter plan
    authoring, and execution-trace capture on top of the real lifecycle.
runner
    BenchmarkRunner: end-to-end execution and per-challenge evaluation.
report
    Human/JSON report rendering for runs and aggregates.
"""

from app.benchmarks.challenges import (
    build_all_challenges,
    challenge_categories,
    get_challenge,
)
from app.benchmarks.models import (
    BenchmarkReport,
    Challenge,
    ChallengeResult,
    Milestone,
    RunTrace,
    evaluate_challenge,
)
from app.benchmarks.report import (
    render_report,
    report_to_json,
)

__all__ = [
    "BenchmarkReport",
    "Challenge",
    "ChallengeResult",
    "Milestone",
    "RunTrace",
    "build_all_challenges",
    "challenge_categories",
    "evaluate_challenge",
    "get_challenge",
    "render_report",
    "report_to_json",
]
