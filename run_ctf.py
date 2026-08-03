"""
run_ctf.py — General autonomous CTF solver CLI

Given a challenge folder (typically under ``challenges/<name>``), this
script:

* Reads the folder's description file (``challenge.txt``, or the sole
  ``*.txt``) and parses Title / Category / Points / Description /
  Hints / Challenge URL | Link / Files.
* Ships every sibling data file into the agent's sandbox workspace.
* Runs the full agent pipeline (orchestrator → planner → tool registry →
  Docker sandbox) against the goal until a flag-like token
  (``flag{...}`` / ``ctf{...}`` / ``academy{...}``) appears in real tool
  output or the iteration cap is reached.
* Prints the recovered flag(s) and a run summary.

Usage
-----
    python run_ctf.py challenges/My_Challenge
    python run_ctf.py My_Challenge
    python run_ctf.py challenges/My_Challenge/challenge.txt
    python run_ctf.py My_Challenge --verbose --max-iterations 10
    python run_ctf.py --list

The LLM provider, model, and sandbox image are read from the project
``.env`` (or environment variables).  A live Docker daemon and a valid
provider API key are required for a real run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# -- path bootstrap -------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.abspath(os.path.join(_THIS_DIR, "backend"))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Ensure the root .env is visible to pydantic-settings regardless of CWD.
_env_file = os.path.join(_THIS_DIR, ".env")
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

from app.ctf.runner import (  # noqa: E402
    load_challenge_folder,
    run_challenge_folder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a CTF challenge folder with the AI Agent Harness."
    )
    parser.add_argument(
        "challenge",
        nargs="?",
        default="",
        help=(
            "Challenge folder path, a challenge.txt path, or a folder name "
            "resolved under challenges/."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available challenge folders and exit.",
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
        help="Write the JSON result to this file.",
    )
    return parser.parse_args()


def resolve_folder(arg: str) -> Path:
    """Resolve a folder path, description-file path, or bare folder name."""
    path = Path(arg)
    if path.is_dir():
        return path
    if path.is_file():
        return path.parent
    candidate = Path(_THIS_DIR) / "challenges" / arg
    if candidate.is_dir():
        return candidate
    raise SystemExit(
        f"ERROR: challenge '{arg}' not found.\n"
        f"Pass a folder path, a challenge.txt path, or a name under "
        f"'{Path(_THIS_DIR) / 'challenges'}'."
    )


def list_challenges() -> None:
    challenges_dir = Path(_THIS_DIR) / "challenges"
    if not challenges_dir.is_dir():
        print(f"No '{challenges_dir}' folder found.")
        return
    rows: list[tuple[str, str, str]] = []
    for folder in sorted(challenges_dir.iterdir()):
        if not folder.is_dir():
            continue
        try:
            parsed = load_challenge_folder(folder)
        except (FileNotFoundError, NotADirectoryError):
            continue
        rows.append((folder.name, parsed.category, parsed.title))
    if not rows:
        print("No challenges found.")
        return
    print(f"{'name':<28}{'category':<18}title")
    print("-" * 80)
    for name, category, title in rows:
        print(f"{name:<28}{category:<18}{title}")


def print_result(result: dict[str, Any]) -> None:
    print(f"\nChallenge : {result.get('name')}")
    title = result.get("title") or result.get("name")
    if title:
        print(f"Title     : {title}")
    if result.get("category"):
        print(f"Category  : {result.get('category')}")
    files = result.get("files") or []
    print(f"Files     : {', '.join(files) if files else 'none'}")
    print(
        f"Status    : {result.get('status')} "
        f"({result.get('iterations')} iterations, "
        f"{result.get('execution_time_ms')} ms)"
    )
    tools = result.get("tools_used") or []
    print(f"Tools     : {', '.join(tools) if tools else 'none'}")

    if result.get("llm_unavailable"):
        print("\nLLM unavailable (e.g. daily quota exceeded) — nothing ran.")

    if result.get("flag_found"):
        for flag in result.get("flags", []):
            print(f"\nFLAG      : {flag}")
    else:
        print("\nNo flag recovered.")

    if result.get("errors"):
        print(f"\nErrors    : {len(result['errors'])}")
    final_answer = result.get("final_answer")
    if final_answer:
        print(f"\nFinal answer:\n{final_answer[:500]}")


async def main() -> int:
    args = parse_args()

    if args.list:
        list_challenges()
        return 0

    if not args.challenge:
        print("ERROR: a challenge folder is required (or use --list).")
        return 2

    folder = resolve_folder(args.challenge)
    print(f"Solving challenge: {folder}\n")

    try:
        result = await run_challenge_folder(
            folder,
            max_iterations=args.max_iterations,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"ERROR: challenge run failed: {exc}")
        return 1

    print_result(result)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nJSON result written to: {output_path}")

    return 0 if result["flag_found"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
