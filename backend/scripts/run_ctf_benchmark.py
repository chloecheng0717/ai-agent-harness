"""
CTF Benchmark Runner CLI

Executes the local CTF challenge suite against the current agent
capability and prints a report (success rate, failed step, tools used,
execution time, number of iterations).

Usage
-----
    python scripts/run_ctf_benchmark.py [--list]
    python scripts/run_ctf_benchmark.py [--challenges name1,name2]
                                         [--max-iterations 10]
                                         [--verbose]
                                         [--output report.json]

The LLM provider, model, and sandbox image are read from the project
``.env`` (or environment variables).  A live Docker daemon and a valid
provider API key are required for a real run.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# -- path bootstrap -------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_BACKEND_DIR, ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Ensure the root .env is visible to pydantic-settings regardless of CWD.
_env_file = os.path.join(_PROJECT_ROOT, ".env")
if os.path.exists(_env_file):
    with open(_env_file, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _value = _line.split("=", 1)
                _key, _value = _key.strip(), _value.strip()
                if _key not in os.environ:
                    os.environ[_key] = _value

# -- imports --------------------------------------------------------

from app.benchmarks.challenges import build_all_challenges  # noqa: E402
from app.benchmarks.models import BenchmarkReport  # noqa: E402
from app.benchmarks.report import render_report, report_to_json  # noqa: E402
from app.benchmarks.runner import BenchmarkRunner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AI Agent Harness CTF benchmark suite."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available challenges and exit.",
    )
    parser.add_argument(
        "--challenges",
        default="",
        help="Comma-separated challenge names to run (default: all).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Cap orchestrator plan-execute-evaluate iterations.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print LLM requests, plans, and tool calls.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Write the JSON report to this file.",
    )
    return parser.parse_args()


def _select_challenges(names: str) -> list:
    from app.benchmarks.challenges import get_challenge

    all_challenges = build_all_challenges()
    if not names.strip():
        return all_challenges

    selected = []
    for name in names.split(","):
        name = name.strip()
        if not name:
            continue
        try:
            selected.append(get_challenge(name))
        except KeyError:
            available = ", ".join(c.name for c in all_challenges)
            print(f"ERROR: unknown challenge '{name}'. Available: {available}")
            sys.exit(2)
    return selected


def _print_challenge_list() -> None:
    challenges = build_all_challenges()
    print(f"{'name':<24}{'category':<14}{'difficulty':<10} capabilities")
    print("-" * 80)
    for c in challenges:
        print(
            f"{c.name:<24}{c.category:<14}{c.difficulty:<10} "
            + ", ".join(c.required_capabilities)
        )


async def main() -> int:
    args = parse_args()

    if args.list:
        _print_challenge_list()
        return 0

    challenges = _select_challenges(args.challenges)
    print(f"Running {len(challenges)} challenge(s)...\n")

    runner = BenchmarkRunner(
        verbose=args.verbose,
        max_iterations=args.max_iterations,
    )

    try:
        results = await runner.run_challenges(challenges)
    except Exception as exc:
        print(f"ERROR: benchmark run failed: {exc}")
        return 1

    report = BenchmarkReport(results=results)

    print("\n" + render_report(report) + "\n")

    if args.output:
        output_path = args.output
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(report_to_json(report))
        print(f"JSON report written to: {output_path}")

    return 0 if report.passed_count == len(report.results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
