"""
Benchmark Orchestrator

A thin subclass of the production :class:`Orchestrator` that:

* Injects the challenge files into the persistent run workspace before
  any tool runs, so every challenge ships self-contained files.
* Generates plans whose steps carry real, LLM-authored ``tool_parameters``
  (the CTF tools need concrete arguments such as ``code`` for ``python``
  or ``input``/``encoding``/``mode`` for ``encoding``).  This mirrors the
  behaviour already proven in ``backend/scripts/test_ctf_run.py``.
* Captures an execution trace (every tool call with its output, the final
  plan, and the iteration count) that the benchmark reporter uses for
  pass/fail and "failed step" analysis.

No new tools are introduced and the underlying lifecycle is untouched.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import structlog

from app.agents.core.orchestrator import Orchestrator
from app.models.llm import ChatMessage
from app.models.state import AgentState, PlanStep


class BenchmarkOrchestrator(Orchestrator):
    """
    Orchestrator specialised for running a :class:`Challenge`.

    Parameters
    ----------
    challenge_files : dict[str, bytes] | None
        Workspace-relative path → content to write into the run workspace.
    verbose : bool
        When ``True``, print LLM requests, plans, and tool calls so a run
        can be followed step by step on stdout.
    **kwargs
        Forwarded to :class:`Orchestrator` (``llm_provider``,
        ``tool_registry``, ``sandbox``, ``max_iterations``, ...).
    """

    def __init__(
        self,
        challenge_files: dict[str, bytes] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.challenge_files: dict[str, bytes] = challenge_files or {}
        self.verbose = verbose
        self.logger = structlog.get_logger().bind(component="benchmark_orchestrator")

        # Captured benchmark evidence (see :meth:`run`).
        self.benchmark_tool_calls: list[dict[str, Any]] = []
        self.benchmark_plan: list[dict[str, Any]] = []
        self.benchmark_workspace_files: list[dict[str, Any]] = []
        self.benchmark_iterations: int = 0

        # Relative path -> sha256 of each challenge seed file, used to tell
        # execution-produced workspace files apart from injected inputs.
        self._baseline_files: dict[str, str] = {}

    # ── workspace: inject challenge files ───────────────────────

    def _create_workspace(self) -> str | None:
        """Create the run workspace and populate it with challenge files."""
        workspace_id = super()._create_workspace()
        if workspace_id is None or self.sandbox is None:
            return workspace_id

        workspace = self.sandbox.get_workspace(workspace_id)
        if workspace is None:
            return workspace_id

        for rel_path, content in self.challenge_files.items():
            rel = rel_path.lstrip("/\\")
            target = workspace.host_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = content.encode("utf-8") if isinstance(content, str) else content
            target.write_bytes(raw)
            self.logger.info(
                "Challenge file written to workspace",
                path=rel_path,
                bytes=len(raw),
            )

        self._baseline_files = {
            rel_path.lstrip("/\\").replace("\\", "/"): hashlib.sha256(
                content.encode("utf-8") if isinstance(content, str) else content
            ).hexdigest()
            for rel_path, content in self.challenge_files.items()
        }
        return workspace_id

    # ── workspace: capture execution-produced files ─────────────

    def _snapshot_workspace(self, workspace_id: str | None) -> None:
        """
        Record verified workspace files produced during execution.

        Every file in the run workspace whose content differs from the
        injected challenge seed files is captured (path, size, content).
        These are real execution artifacts: anything a tool wrote,
        extracted, decoded, or modified while the run was executing.
        """
        if workspace_id is None or self.sandbox is None:
            return
        workspace = self.sandbox.get_workspace(workspace_id)
        if workspace is None:
            return
        host = workspace.host_path
        if not host.is_dir():
            return

        for path in sorted(host.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(host).as_posix()
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if self._baseline_files.get(rel) == hashlib.sha256(data).hexdigest():
                continue
            self.benchmark_workspace_files.append(
                {
                    "path": rel,
                    "bytes": len(data),
                    "content": data.decode("utf-8", errors="replace"),
                }
            )

    def _cleanup_workspace(self, workspace_id: str | None) -> None:
        """Snapshot execution-produced workspace files before removal."""
        self._snapshot_workspace(workspace_id)
        super()._cleanup_workspace(workspace_id)

    # ── planning with LLM-authored tool parameters ──────────────

    def _format_tool_descriptions(self, tool_names: list[str]) -> str:
        """Describe tools including their parameter schemas."""
        parts: list[str] = []
        for name in tool_names:
            defn = (
                self.tool_registry.get_tool_definition(name)
                if self.tool_registry
                else None
            )
            if defn and defn.description:
                parts.append(f"  - {name}: {defn.description}")
            else:
                parts.append(f"  - {name}")
            if defn and defn.parameters:
                props = defn.parameters.get("properties", {})
                required = set(defn.parameters.get("required", []))
                for pname, pmeta in props.items():
                    ptype = pmeta.get("type", "any")
                    enums = pmeta.get("enum")
                    suffix = " | ".join(enums) if enums else ptype
                    flag = "(required)" if pname in required else ""
                    note = pmeta.get("description", "")
                    line = f"      {pname} ({suffix}) {flag}"
                    if note:
                        line += f" -- {note}"
                    parts.append(line.rstrip())
        return "\n".join(parts)

    async def create_plan(self, state: AgentState) -> AgentState:
        """Ask the LLM to author a plan whose steps carry real tool arguments."""
        if self.llm_provider is None or not state.tools_available:
            state = await super().create_plan(state)
            return state

        tools_desc = self._format_tool_descriptions(state.tools_available)
        system_msg = self._build_system_message(
            "You are a planning agent. Given a goal, produce a numbered "
            "execution plan.\n\n"
            f"Available tools:\n{tools_desc}\n\n"
            "Reply with a JSON array of objects, each having 'index' (int), "
            "'description' (str), 'tool_name' (str or null), "
            "'rationale' (str), and 'tool_parameters' (object). "
            "'tool_parameters' must contain the exact arguments that tool "
            "needs for THIS specific step -- e.g. for 'python' provide the "
            "real Python 'code' (or 'script_path') that performs that step "
            "(reading a file, decoding base64, extracting a zip, etc.); "
            "for 'encoding' provide 'input', 'encoding' "
            "('base64'/'hex'/'url'), and 'mode' ('encode'/'decode'/'auto'); "
            "for 'archive' provide 'archive_path' (and 'extract_dir' when "
            "useful); for 'file' or 'strings' provide 'path' or 'content'. "
            "Never guess or pad placeholders. Use {} when the step needs no "
            "tool. Reply with RAW JSON only -- no markdown, no code fences."
        )
        user_msg = ChatMessage(
            role="user",
            content=f"Goal: {state.goal}\n\n{self._plan_context(state)}".strip(),
        )

        try:
            raw = await self._llm_chat([system_msg, user_msg], state)
            raw = raw.strip()
            if raw.startswith("```"):
                first_nl = raw.find("\n")
                if first_nl != -1:
                    raw = raw[first_nl + 1:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                elif "```" in raw:
                    raw = raw.rsplit("```", 1)[0]
                raw = raw.strip()

            plan_data = json.loads(raw)
            if isinstance(plan_data, list):
                state.plan = [
                    PlanStep(
                        index=item.get("index", i),
                        description=item.get("description", f"Step {i}"),
                        tool_name=item.get("tool_name"),
                        rationale=item.get("rationale", ""),
                        tool_parameters=item.get("tool_parameters", {}),
                    )
                    for i, item in enumerate(plan_data)
                ]
                state.metadata["plan_source"] = "benchmark_llm"
        except Exception as exc:
            self.logger.warning(
                "LLM plan parsing failed, falling back to base planner",
                error=str(exc),
            )
            state = await super().create_plan(state)

        if self.verbose:
            self._print_plan(state)
        return state

    def _print_plan(self, state: AgentState) -> None:
        print("\n===== PLAN =====")
        for ps in state.plan:
            print(f"  Step {ps.index}: {ps.description}")
            print(f"    tool:   {ps.tool_name}")
            print(f"    params: {ps.tool_parameters}")

    # ── execution: capture trace ────────────────────────────────

    async def execute_plan(self, state: AgentState) -> AgentState:
        """Run every plan step (via the base lifecycle) and capture the trace."""
        state = await super().execute_plan(state)

        self.benchmark_plan = [p.model_dump() for p in state.plan]
        for record in state.results:
            self.benchmark_tool_calls.append(
                {
                    "step_index": record.get("step_index"),
                    "description": record.get("description"),
                    "tool_name": record.get("tool_name"),
                    "success": record.get("success"),
                    "output": record.get("output"),
                    "error": record.get("error"),
                }
            )

        if self.verbose:
            for record in state.results:
                status = "OK" if record.get("success") else "FAIL"
                tool = record.get("tool_name") or "(reasoning)"
                print(
                    f"  [{status}] step {record.get('step_index')}: {tool} "
                    f"-> {json.dumps(record.get('output'), default=str)[:200]}"
                )
        return state

    # ── entry point: merge benchmark evidence into the result ──

    async def run(
        self,
        task_id: Any,
        task_data: dict[str, Any],
        tools_available: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run the standard lifecycle and annotate the result with benchmark
        evidence (tool trace, plan, and iteration count) so the reporter can
        evaluate the run without re-deriving anything.
        """
        result = await super().run(task_id, task_data, tools_available)
        result["benchmark_trace"] = self.benchmark_tool_calls
        result["benchmark_plan"] = self.benchmark_plan
        result["benchmark_workspace_files"] = self.benchmark_workspace_files
        result["iterations"] = self.benchmark_iterations
        if self.verbose:
            print(
                f"\n===== RUN COMPLETE ====="
                f"\n  status:     {result.get('status')}"
                f"\n  iterations: {self.benchmark_iterations}"
                f"\n  tool calls: {len(self.benchmark_tool_calls)}"
            )
        return result

    # ── iteration tracking ──────────────────────────────────────

    async def _run_step(self, step: Any, state: AgentState) -> AgentState:
        """Track the current iteration number across plan-execute loops."""
        state = await super()._run_step(step, state)
        iteration = state.metadata.get("iteration")
        if isinstance(iteration, int) and iteration > self.benchmark_iterations:
            self.benchmark_iterations = iteration
        return state
