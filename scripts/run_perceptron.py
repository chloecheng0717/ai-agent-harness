"""
Perceptron Play Naught (remote network CTF) benchmark runner.

Connects the agent harness to the live Perceptron service and prints a
milestone-by-milestone trace plus the final result.  Requires a running
Docker daemon, the ``ctf-sandbox:latest`` image, and a valid LLM API key
from the project ``.env``.
"""

from __future__ import annotations

import asyncio
import os
import sys

# -- path bootstrap -------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_BACKEND_DIR = os.path.join(_PROJECT_ROOT, "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Ensure the root .env is visible to pydantic-settings regardless of CWD
# (sandbox_network_disabled must stay False or the network plan is skipped).
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

from app.benchmarks.challenges.network import build_perceptron  # noqa: E402
from app.benchmarks.runner import BenchmarkRunner  # noqa: E402


async def main():
    challenge = build_perceptron()

    runner = BenchmarkRunner(
        verbose=True,
        max_iterations=10,
    )

    result = await runner.run_challenge(challenge)

    print("\n===== MILESTONES =====")
    for milestone in result.milestones:
        print(f"  - {milestone.get('name')}: "
              f"{'PASSED' if milestone.get('passed') else 'FAILED'}")

    print("\n===== RESULT =====")
    print("Passed:", result.passed)
    print("Flag found:", result.flag_found)
    print("Answer:")
    print(result.final_answer)


if __name__ == "__main__":
    asyncio.run(main())
