"""
Unit Tests for BenchmarkOrchestrator Challenge Planning

Covers the challenge-planning reliability fixes:

* The planner prompt must preserve challenge filenames and list the exact
  workspace paths of every available file (``_plan_context``).
* When a plan step is missing a file ``path``/``archive_path`` argument,
  the repair logic must pick the seed file the goal actually names rather
  than blindly using the first seed file.
"""

from app.benchmarks.orchestrator import BenchmarkOrchestrator
from app.models.state import AgentState, PlanStep


def _orchestrator(files: dict[str, bytes]) -> BenchmarkOrchestrator:
    return BenchmarkOrchestrator(challenge_files=files)


def _plan_step(tool_name: str, params: dict | None = None, description: str = "") -> PlanStep:
    return PlanStep(
        index=0,
        description=description or f"Run {tool_name}",
        tool_name=tool_name,
        tool_parameters=params or {},
    )


class TestPlanContextListsChallengeFiles:
    def test_empty_with_no_challenge_files(self) -> None:
        orch = _orchestrator({})
        state = AgentState()
        ctx = orch._plan_context(state)
        assert "Workspace files available" not in ctx

    def test_lists_every_file_with_workspace_path(self) -> None:
        orch = _orchestrator(
            {
                "logs.bin": b"payload",
                "message.txt": b"hello",
            }
        )
        state = AgentState()
        ctx = orch._plan_context(state)
        assert "/workspace/logs.bin" in ctx
        assert "/workspace/message.txt" in ctx
        # Filenames themselves are preserved, not mangled.
        assert "logs.bin" in ctx
        assert "message.txt" in ctx

    def test_normalises_windows_backslashes(self) -> None:
        orch = _orchestrator({r"sub\dir\data.bin": b"payload"})
        state = AgentState()
        ctx = orch._plan_context(state)
        assert "/workspace/sub/dir/data.bin" in ctx
        assert "sub\\dir" not in ctx

    def test_preserves_base_context_sections(self) -> None:
        orch = _orchestrator({"logs.bin": b"payload"})
        state = AgentState()
        state.metadata["replan_feedback"] = "still encoded"
        ctx = orch._plan_context(state)
        assert "still encoded" in ctx
        assert "/workspace/logs.bin" in ctx


class TestSelectSeedFile:
    def test_returns_first_when_goal_names_nothing(self) -> None:
        selected = BenchmarkOrchestrator._select_seed_file(
            ["a.txt", "b.bin"],
            goal="Find the hidden flag",
            step_description="Extract strings",
        )
        assert selected == "a.txt"

    def test_picks_file_named_in_goal(self) -> None:
        selected = BenchmarkOrchestrator._select_seed_file(
            ["readme.txt", "logs.bin"],
            goal="Analyse the binary payload in logs.bin to recover the flag",
            step_description="Extract strings from the file",
        )
        assert selected == "logs.bin"

    def test_picks_file_named_in_step_description(self) -> None:
        selected = BenchmarkOrchestrator._select_seed_file(
            ["a.txt", "b.bin", "c.zip"],
            goal="Recover the flag",
            step_description="Extract strings from c.zip contents",
        )
        assert selected == "c.zip"

    def test_explicit_workspace_path_outranks_basename(self) -> None:
        # Both basenames appear, but only one gets the full-path mention.
        selected = BenchmarkOrchestrator._select_seed_file(
            ["data.bin", "notes.txt"],
            goal="Read /workspace/data.bin then decode it",
            step_description="Use strings on /workspace/data.bin",
        )
        assert selected == "data.bin"


class TestRepairPlanParameters:
    def test_fills_missing_path_with_goal_file(self) -> None:
        orch = _orchestrator(
            {
                "readme.txt": b"not this one",
                "secret.bin": b"the payload",
            }
        )
        state = AgentState()
        state.goal = "Extract strings from secret.bin to find the flag"
        state.plan = [_plan_step("strings", {}, "Extract printable strings")]

        orch._repair_plan_parameters(state)

        assert state.plan[0].tool_parameters["path"] == "/workspace/secret.bin"

    def test_keeps_existing_path_parameter(self) -> None:
        orch = _orchestrator({"a.txt": b"x", "b.bin": b"y"})
        state = AgentState()
        state.goal = "Analyse b.bin"
        state.plan = [_plan_step("strings", {"path": "/workspace/b.bin"})]

        orch._repair_plan_parameters(state)

        assert state.plan[0].tool_parameters["path"] == "/workspace/b.bin"

    def test_keeps_content_alternative(self) -> None:
        orch = _orchestrator({"a.txt": b"x", "b.bin": b"y"})
        state = AgentState()
        state.goal = "Analyse b.bin"
        state.plan = [_plan_step("strings", {"content": "raw text"})]

        orch._repair_plan_parameters(state)

        assert "path" not in state.plan[0].tool_parameters
        assert state.plan[0].tool_parameters["content"] == "raw text"

    def test_repairs_archive_path(self) -> None:
        orch = _orchestrator({"a.txt": b"x", "b.zip": b"y"})
        state = AgentState()
        state.goal = "Unpack the archive b.zip"
        state.plan = [_plan_step("archive", {}, "Extract archive")]

        orch._repair_plan_parameters(state)

        assert state.plan[0].tool_parameters["archive_path"] == "/workspace/b.zip"

    def test_does_nothing_without_challenge_files(self) -> None:
        orch = _orchestrator({})
        state = AgentState()
        state.goal = "Analyse secret.bin"
        state.plan = [_plan_step("strings", {}, "Extract strings")]

        orch._repair_plan_parameters(state)

        assert state.plan[0].tool_parameters == {}

    def test_does_not_touch_non_path_tools(self) -> None:
        orch = _orchestrator({"a.txt": b"x", "b.bin": b"y"})
        state = AgentState()
        state.goal = "Decode the data"
        state.plan = [_plan_step("encoding", {"input": "data", "encoding": "base64"})]

        orch._repair_plan_parameters(state)

        assert state.plan[0].tool_parameters["input"] == "data"
        assert "path" not in state.plan[0].tool_parameters
