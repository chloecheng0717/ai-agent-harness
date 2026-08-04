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
from app.models.llm import ChatMessage, LLMUnavailableError
from app.models.state import (
    AgentState,
    PlanStep,
    extract_llm_tool_name,
    extract_llm_tool_params,
    parse_llm_plan_json,
)


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
                summary = defn.parameter_summary()
                if summary:
                    parts.append(summary)
        return "\n".join(parts)

    async def create_plan(self, state: AgentState) -> AgentState:
        """Ask the LLM to author a plan whose steps carry real tool arguments."""
        # A network challenge is solved by the deterministic socket plan so
        # the remote service is interacted with instead of the goal text
        # being misread as encoded data.  This also keeps offline benchmark
        # runs (sandbox network disabled) from attempting real connections.
        state = self._build_network_plan(state)
        if state.plan:
            if self.verbose:
                self._print_plan(state)
            return state

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
            "tool. Reply with RAW JSON only -- no markdown, no code fences.\n\n"
            "Tool guidance:\n"
            "- Use 'strings' to extract hidden/embedded/printable text from "
            "files, logs, and binaries. Prefer it whenever the goal is to "
            "find hidden text inside a file. ALWAYS supply 'path' (a file "
            "in the workspace) or 'content' for every 'strings' step.\n"
            "- Use 'file' ONLY to inspect a file's type or metadata (MIME "
            "type, magic bytes, signature). Never use it as the only step "
            "when the goal is to recover hidden content.\n"
            "- When data is encoded, decode ONE layer per step. If a decode "
            "still yields encoded-looking text (hex, base64, or URL-escaped), "
            "add a further step to decode the next layer.\n"
            "- A challenge that points at a remote service ('Service "
            "endpoint', 'host:port', 'nc host port', netcat, socket/TCP "
            "connection instructions) is a NETWORK challenge: use the "
            "'python' tool to open a socket, read the welcome banner, follow "
            "the service's commands (e.g. SET / SHOW / CHECK), and recover "
            "the flag from the service's reply. Do NOT treat the hostname, "
            "banner, or protocol text as encoded data to decode, and do not "
            "re-read the same file or re-decode the same input more than "
            "once.\n"
            "- The task is complete only when the literal flag text (matching "
            "'flag{...}') is recovered. Keep planning until the flag is "
            "recovered."
        )
        user_msg = ChatMessage(
            role="user",
            content=f"Goal: {state.goal}\n\n{self._plan_context(state)}".strip(),
        )

        try:
            raw = await self._llm_chat([system_msg, user_msg], state)
            plan_data = parse_llm_plan_json(raw)
            if plan_data is not None:
                state.plan = [
                    PlanStep(
                        index=item.get("index", i),
                        description=item.get("description", f"Step {i}"),
                        tool_name=extract_llm_tool_name(item),
                        rationale=item.get("rationale", ""),
                        tool_parameters=extract_llm_tool_params(item),
                    )
                    for i, item in enumerate(plan_data)
                ]
                state.metadata["plan_source"] = "benchmark_llm"
            else:
                self.logger.warning(
                    "LLM plan parsing failed, falling back to base planner",
                    error="no usable JSON array recovered",
                )
                state = await super().create_plan(state)
        except LLMUnavailableError:
            raise
        except Exception as exc:
            self.logger.warning(
                "LLM plan parsing failed, falling back to base planner",
                error=str(exc),
            )
            state = await super().create_plan(state)

        if self.verbose:
            self._print_plan(state)
        return state

    #: Tools that operate on a workspace file path, mapped to the parameter
    #: that carries the path.  The LLM frequently selects these without
    #: authoring the path, so the parameter is repaired deterministically.
    _PATH_TOOLS: dict[str, str] = {
        "strings": "path",
        "file": "path",
        "archive": "archive_path",
    }

    #: Tools that accept raw ``content`` as a full alternative to the path.
    _CONTENT_ALTERNATIVE_TOOLS = frozenset({"strings", "file"})

    async def select_tools(self, state: AgentState) -> AgentState:
        """Assign tools, then repair missing path parameters on the plan."""
        state = await super().select_tools(state)
        self._repair_plan_parameters(state)
        return state

    def _repair_plan_parameters(self, state: AgentState) -> None:
        """
        Deterministically fill missing path arguments for path-based tools.

        The LLM sometimes selects tools like ``strings`` without authoring
        a ``path`` argument, which makes execution fail with "Either
        'path' or 'content' must be provided".  When the challenge ships
        seed files and a step's parameters are missing the path-type input
        (and no ``content`` alternative is given), the first seed file is
        used so the step can actually run.
        """
        if not self.challenge_files:
            return
        seed_names = sorted(
            rel.lstrip("/\\").replace("\\", "/")
            for rel in self.challenge_files
        )
        if not seed_names:
            return
        for step in state.plan:
            path_param = self._PATH_TOOLS.get(step.tool_name or "")
            if path_param is None:
                continue
            params = dict(step.tool_parameters or {})
            if params.get(path_param):
                continue
            if (
                step.tool_name in self._CONTENT_ALTERNATIVE_TOOLS
                and params.get("content") is not None
            ):
                continue
            params[path_param] = "/workspace/" + seed_names[0]
            step.tool_parameters = params
            self.logger.info(
                "Filled missing tool path parameter",
                tool_name=step.tool_name,
                step_index=step.index,
                param=path_param,
                value=params[path_param],
            )

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
                output = record.get("output")
                if output is None and record.get("error"):
                    # A failed step often carries only an error; print it
                    # instead of the meaningless literal "null".
                    detail = f"error: {record.get('error')}"
                elif isinstance(output, dict) and output.get("stdout"):
                    detail = f"stdout: {output.get('stdout')}"
                else:
                    detail = json.dumps(output, default=str)
                print(
                    f"  [{status}] step {record.get('step_index')}: {tool} "
                    f"-> {detail[:200]}"
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
