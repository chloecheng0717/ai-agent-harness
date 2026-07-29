"""
CTF Run Test Script

End-to-end test that wires together the full CTF toolchain:
  DockerSandbox -> FileTool + StringsTool + PythonTool + EncodingTool
  -> ToolRegistry -> Orchestrator (real LLM provider) -> FlagDetector

Goal: "Find the flag hidden in /workspace/challenge.txt"

Every step -- LLM thoughts, tool calls, tool outputs, and flag-detection
results -- is printed to stdout so you can follow along.
"""

import asyncio
import base64
import json
import os
import sys
import tempfile
import traceback
from typing import Any, Dict, Optional
from uuid import uuid4

# Allow imports from the backend package
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_BACKEND_DIR, ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Ensure the root .env is visible to pydantic-settings regardless of CWD
_env_file = os.path.join(_PROJECT_ROOT, ".env")
if os.path.exists(_env_file):
    with open(_env_file, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _value = _line.split("=", 1)
                _key, _value = _key.strip(), _value.strip()
                # Only set if not already present so real env vars win
                if _key not in os.environ:
                    os.environ[_key] = _value

from app.agents.core.orchestrator import Orchestrator
from app.agents.core.reasoning_loop import FlagDetector
from app.config import get_settings
from app.models.llm import ChatMessage
from app.models.providers.openai_provider import OpenAIProvider
from app.models.state import AgentState, LifecycleStep, PlanStep, StepStatus
from app.models.tool import ToolExecutionRequest, ToolPermission
from app.sandbox.docker_sandbox import (
    DockerSandbox,
    SandboxConfig,
    DockerNotAvailableError,
    SandboxError,
)
from app.tools.plugins.encoding_tool import EncodingTool
from app.tools.plugins.file_tool import FileTool
from app.tools.plugins.python_tool import PythonTool
from app.tools.plugins.strings_tool import StringsTool
from app.tools.registry import ToolRegistry

# -- helpers ----------------------------------------------------------

SEP = "=" * 72

_TEST_DATA = os.path.join(_THIS_DIR, "..", "test_data", "challenge.txt")


def heading(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def subheading(title: str) -> None:
    print(f"\n  --- {title} ---")


def scan_and_report(
    detector: FlagDetector,
    text: str,
    source: str,
) -> Optional[str]:
    flag = detector.scan(text)
    if flag:
        print(f"    [!] FLAG DETECTED in {source}: {flag}")
    return flag


def scan_dict_and_report(
    detector: FlagDetector,
    data: Dict[str, Any],
    source: str,
) -> Optional[str]:
    flag = detector.scan_dict(data)
    if flag:
        print(f"    [!] FLAG DETECTED in {source}: {flag}")
    return flag


def trunc(text: str, n: int = 300) -> str:
    return text[:n] + "..." if len(text) > n else text


# -- Printing Orchestrator --------------------------------------------


class PrintingOrchestrator(Orchestrator):
    """Orchestrator that prints every lifecycle step in detail.

    - Intercepts LLM calls to print prompts + responses
    - Injects sensible ``tool_parameters`` into plan steps (the base
      Orchestrator's ``create_plan`` doesn't populate them)
    - Prints every tool invocation together with its output
    - Scans all text with ``FlagDetector``
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.detector = FlagDetector()
        # -- param injection table ------------------------------
        self._param_template: Dict[str, Dict[str, Any]] = {
            "file": {"path": "/workspace/challenge.txt"},
            "strings": {"path": "/workspace/challenge.txt", "min_length": 4},
            "python": {"code": "print(open('/workspace/challenge.txt').read())"},
            "encoding": {},
        }

    # -- LLM interceptor ----------------------------------------

    async def _llm_chat(
        self,
        messages: list[ChatMessage],
        state: AgentState,
        **kwargs: Any,
    ) -> str:
        subheading("LLM Request")
        for m in messages:
            preview = trunc(m.content, 400)
            print(f"  [{m.role.upper()}] {preview}")

        response = await super()._llm_chat(messages, state, **kwargs)

        print(f"  [ASSISTANT] {trunc(response, 600)}")
        scan_and_report(self.detector, response, "LLM response")
        return response

    # -- create_plan: strip markdown, inject tool_parameters ----

    async def create_plan(self, state: AgentState) -> AgentState:
        subheading("Creating plan via LLM ...")

        if self.llm_provider is None or not state.tools_available:
            state = await super().create_plan(state)
            return self._print_plan(state)

        # Custom plan generation that handles markdown-wrapped JSON
        # (the base orchestrator's json.loads chokes on ```json ... ```)
        tools_desc = ", ".join(state.tools_available)
        system_msg = self._build_system_message(
            "You are a planning agent. Given a goal, produce a numbered "
            f"execution plan. Available tools: {tools_desc}. "
            "Reply with a JSON array of objects, each having 'index' (int), "
            "'description' (str), 'tool_name' (str or null), and "
            "'rationale' (str). "
            "Reply with RAW JSON only -- no markdown, no code fences."
        )
        user_msg = ChatMessage(role="user", content=f"Goal: {state.goal}")
        try:
            raw = await self._llm_chat([system_msg, user_msg], state)
            raw = raw.strip()
            # Strip markdown code fences if the LLM ignored the instruction
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
                    )
                    for i, item in enumerate(plan_data)
                ]
                state.metadata["plan_source"] = "orchestrator_llm"
        except Exception:
            self.logger.warning("Custom plan parsing failed, falling back", exc_info=True)
            state = await super().create_plan(state)

        # Inject tool_parameters from template
        for ps in state.plan:
            if ps.tool_name and not ps.tool_parameters:
                injection = self._param_template.get(ps.tool_name, {})
                if injection:
                    ps.tool_parameters = dict(injection)

        return self._print_plan(state)

    def _print_plan(self, state: AgentState) -> AgentState:
        heading("PLAN (post-LLM, with injected parameters)")
        for ps in state.plan:
            print(f"  Step {ps.index}: {ps.description}")
            print(f"    tool:   {ps.tool_name}")
            print(f"    params: {ps.tool_parameters}")
            if ps.rationale:
                print(f"    why:    {ps.rationale}")
        return state

    # -- execute_plan: print tool calls --------------------------

    async def execute_plan(self, state: AgentState) -> AgentState:
        heading("EXECUTING PLAN")

        for plan_step in state.plan:
            print(f"\n  >>> Step {plan_step.index}: {plan_step.description}")
            print(f"      tool:   {plan_step.tool_name}")
            print(f"      params: {plan_step.tool_parameters}")

            if not plan_step.tool_name or self.tool_registry is None:
                state.results.append({
                    "step_index": plan_step.index,
                    "description": plan_step.description,
                    "tool_name": plan_step.tool_name,
                    "success": True,
                    "output": "(no tool required)",
                })
                continue

            request = ToolExecutionRequest(
                tool_name=plan_step.tool_name,
                parameters=plan_step.tool_parameters,
                agent_permissions=[
                    ToolPermission.READ,
                    ToolPermission.WRITE,
                    ToolPermission.EXECUTE,
                    ToolPermission.NETWORK,
                    ToolPermission.FILE_SYSTEM,
                ],
            )
            state.tools_used.append(plan_step.tool_name)

            try:
                result = await self.tool_registry.execute(request)
            except Exception as exc:
                print(f"      X EXCEPTION: {exc}")
                state.results.append({
                    "step_index": plan_step.index,
                    "description": plan_step.description,
                    "tool_name": plan_step.tool_name,
                    "success": False,
                    "output": None,
                    "error": str(exc),
                })
                state.add_error(f"Tool '{plan_step.tool_name}' error: {exc}")
                continue

            print(f"      -> success={result.success}")
            if result.output:
                out_str = json.dumps(result.output, indent=4) if isinstance(result.output, dict) else str(result.output)
                print(f"      -> output: {trunc(out_str, 600)}")
            if result.error:
                print(f"      -> error: {result.error}")

            # -- FlagDetector on tool output --------------------
            if result.output:
                self._scan_tool_output(result.output)

            state.results.append({
                "step_index": plan_step.index,
                "description": plan_step.description,
                "tool_name": plan_step.tool_name,
                "success": result.success,
                "output": result.output,
                "error": result.error,
            })

        print("\n  >>> Plan execution complete <<<")
        return state

    def _scan_tool_output(self, output: Any) -> None:
        """Recursively scan tool output for flags."""
        if isinstance(output, str):
            scan_and_report(self.detector, output, "tool output")
        elif isinstance(output, dict):
            scan_dict_and_report(self.detector, output, "tool output")
            for v in output.values():
                self._scan_tool_output(v)
        elif isinstance(output, list):
            for item in output:
                self._scan_tool_output(item)


# -- main -------------------------------------------------------------


async def main() -> int:
    print(f"\n{SEP}")
    print("  CTF RUN TEST")
    print(f"  Goal: Find the flag hidden in /workspace/challenge.txt")
    print(f"  Source: {_TEST_DATA}")
    print(f"{SEP}")

    # -- 1. Prepare challenge file ------------------------------
    heading("1. Preparing challenge file")
    if not os.path.exists(_TEST_DATA):
        print(f"  ERROR: test data not found at {_TEST_DATA}")
        return 1

    tmpdir = tempfile.mkdtemp(prefix="ctf_run_")
    challenge_path = os.path.join(tmpdir, "challenge.txt")
    with open(_TEST_DATA, encoding="utf-8") as f:
        challenge_text = f.read()
    with open(challenge_path, "w", encoding="utf-8") as f:
        f.write(challenge_text)

    print(f"  Copied from: {_TEST_DATA}")
    print(f"  Temp copy:   {challenge_path}")
    print(f"  Content preview: {repr(challenge_text.strip()[:80])}")

    # -- 2. Load settings ---------------------------------------
    heading("2. Loading settings")
    settings = get_settings()
    print(f"  sandbox_image:       {settings.sandbox_image}")
    print(f"  sandbox_timeout:     {settings.sandbox_timeout}s")
    print(f"  sandbox_network:     {'disabled' if settings.sandbox_network_disabled else 'enabled'}")
    print(f"  default_model:       {settings.default_model}")
    print(f"  llm_provider:        {settings.llm_provider}")

    # -- 3. Create DockerSandbox ---------------------------------
    heading("3. Creating DockerSandbox from config")
    sandbox_config = SandboxConfig(
        image=settings.sandbox_image,
        memory_limit=settings.sandbox_memory_limit,
        cpu_limit=settings.sandbox_cpu_limit,
        timeout=settings.sandbox_timeout,
        network_disabled=settings.sandbox_network_disabled,
        additional_volumes={
            "/workspace/challenge.txt": {
                "source": challenge_path,
                "type": "bind",
            },
        },
    )
    sandbox = DockerSandbox(sandbox_config)
    print(f"  Sandbox created: {sandbox_config.image}")

    # Quick health check
    healthy = await sandbox.check_health()
    if healthy:
        print("  Docker health: OK")
    else:
        print("  Docker health: UNREACHABLE (will try anyway)")

    # -- 4. Create tools ----------------------------------------
    heading("4. Creating CTF tools")
    tool_file = FileTool(sandbox)
    tool_strings = StringsTool(sandbox)
    tool_python = PythonTool(sandbox)
    tool_encoding = EncodingTool(sandbox)
    print("  FileTool, StringsTool, PythonTool, EncodingTool created")

    # -- 5. Register tools --------------------------------------
    heading("5. Registering tools in ToolRegistry")
    registry = ToolRegistry()
    for tool in (tool_file, tool_strings, tool_python, tool_encoding):
        registry.register_instance(tool)
        print(f"  Registered: {tool.definition.name}")

    # -- 6. Create LLM provider ---------------------------------
    heading("6. Creating LLM provider")
    try:
        llm_provider = OpenAIProvider()
        print(f"  Provider: {settings.llm_provider}")
        print(f"  Model:    {settings.default_model}")
    except Exception as exc:
        print(f"  FAILED: {exc}")
        print("  Cannot continue without an LLM provider.")
        return 1

    # -- 7. Create orchestrator ---------------------------------
    heading("7. Building Orchestrator")
    orch = PrintingOrchestrator(
        llm_provider=llm_provider,
        tool_registry=registry,
    )
    print("  Orchestrator ready")

    # -- 8. Run -------------------------------------------------
    heading("8. RUNNING ORCHESTRATOR")
    task_id = uuid4()
    print(f"  task_id: {task_id}")
    print(f"  goal:    Find the flag hidden in /workspace/challenge.txt")

    try:
        result = await orch.run(
            task_id=task_id,
            task_data={
                "description": "Find the flag hidden in /workspace/challenge.txt",
            },
            tools_available=["file", "strings", "python", "encoding"],
        )
        print(f"\n  Orchestrator finished: status={result['status']}")
    except Exception as exc:
        print(f"\n  Orchestrator crashed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    # -- 9. Result summary --------------------------------------
    heading("9. RESULT SUMMARY")
    print(f"  Status:       {result['status']}")
    print(f"  Goal:         {result['goal']}")
    print(f"  Final answer: {trunc(result.get('final_answer') or '(none)', 500)}")
    print(f"  Errors:       {result['errors']}")
    print(f"  Tools used:   {result['tools_used']}")
    print(f"  Token usage:  {result['token_usage']}")

    subheading("Lifecycle step history")
    for sr in result["step_history"]:
        dur = f"{sr['duration_ms']:.0f} ms" if sr["duration_ms"] is not None else "?"
        err = f"  error={sr['error']}" if sr["error"] else ""
        print(f"  [{sr['status']:>10}] {sr['step']:25s}  {dur:>8s}  {err}")

    # -- 10. Final FlagDetector pass ----------------------------
    heading("10. FLAG DETECTOR -- final scan")
    detector = FlagDetector()

    all_json = json.dumps(result)
    flag = scan_and_report(detector, all_json, "full result JSON")
    if not flag:
        print("  (no flag found in full result)")

    if result.get("final_answer"):
        scan_and_report(detector, result["final_answer"], "final_answer")

    if result.get("errors"):
        for i, err in enumerate(result["errors"]):
            scan_and_report(detector, err, f"errors[{i}]")

    # -- 11. Done -----------------------------------------------
    heading("11. CLEANUP")
    try:
        await sandbox.close()
        print("  Sandbox closed")
    except Exception as exc:
        print(f"  Sandbox close warning: {exc}")

    try:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"  Temp dir removed: {tmpdir}")
    except Exception:
        pass

    print(f"\n{SEP}")
    print("  TEST COMPLETE")
    print(f"{SEP}\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
