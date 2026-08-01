"""
CTF Run Test Script

End-to-end test that wires together the full CTF toolchain:
  DockerSandbox -> FileTool + StringsTool + PythonTool + EncodingTool
  -> ToolRegistry -> Orchestrator (real LLM provider) -> FlagDetector

Goal: "Find the flag hidden in /workspace/challenge.txt"

The challenge file contains a base64-encoded zip archive. The zip
contains flag.txt with the flag inside.

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
from typing import Any, Dict, List, Optional
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

_TEST_DATA = os.path.join(_THIS_DIR, "..", "test_data", "challenge_zip.txt")


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
    - Has the LLM generate each step's ``tool_parameters`` directly
      (real code for ``python``, ``input``/``mode`` for ``encoding``,
      etc.) as part of the plan JSON, using the tool parameter schemas
    - Prints every tool invocation together with its output
    - Scans all text with ``FlagDetector``
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.detector = FlagDetector()

    # -- tool descriptions: include parameter schemas ----------

    def _format_tool_descriptions(self, tool_names: List[str]) -> str:
        """Format tool names, descriptions, and parameter schemas.

        The extra parameter detail lets the LLM author correct, tool-
        specific ``tool_parameters`` for each plan step (instead of a
        hardcoded template being injected later).
        """
        parts: List[str] = []
        for name in tool_names:
            defn = self.tool_registry.get_tool_definition(name) if self.tool_registry else None
            if defn and defn.description:
                parts.append(f"  - {name}: {defn.description}")
            else:
                parts.append(f"  - {name}")
            if defn and defn.parameters:
                schema = defn.parameters
                props = schema.get("properties", {})
                required = set(schema.get("required", []))
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
        return response

    # -- create_plan: strip markdown, parse LLM tool_parameters -

    async def create_plan(self, state: AgentState) -> AgentState:
        subheading("Creating plan via LLM ...")

        if self.llm_provider is None or not state.tools_available:
            state = await super().create_plan(state)
            return self._print_plan(state)

        # Custom plan generation that handles markdown-wrapped JSON
        # (the base orchestrator's json.loads chokes on ```json ... ```)
        # and asks the LLM to author each step's tool_parameters.
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
            "(reading the file, decoding base64, extracting a zip, etc.); "
            "for 'encoding' provide 'input', 'encoding' ('base64'/'hex'/"
            "'url'), and 'mode' ('encode'/'decode'/'auto'); for 'file' or "
            "'strings' provide 'path' or 'content'. Never guess or pad "
            "placeholders. Use {} when the step needs no tool. "
            "Reply with RAW JSON only -- no markdown, no code fences."
        )
        user_msg = ChatMessage(role="user", content=f"Goal: {state.goal}\n\n{self._plan_context(state)}".strip())
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
                        tool_parameters=item.get("tool_parameters", {}),
                    )
                    for i, item in enumerate(plan_data)
                ]
                state.metadata["plan_source"] = "orchestrator_llm"
        except Exception:
            self.logger.warning("Custom plan parsing failed, falling back", exc_info=True)
            state = await super().create_plan(state)

        return self._print_plan(state)

    def _print_plan(self, state: AgentState) -> AgentState:
        heading("PLAN (post-LLM, with LLM-authored parameters)")
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
                workspace_id=state.metadata.get("workspace_id"),
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
    print("  CTF RUN TEST -- zip-in-base64 challenge")
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
        sandbox=sandbox,
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
