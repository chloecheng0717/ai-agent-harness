"""
Planner Agent

Specialised agent that converts a user request into an ordered, validated
execution plan.  The plan is a list of :class:`PlanStep` objects that the
Orchestrator can drive through its lifecycle.

Responsibilities
----------------
* Decompose a high-level goal into concrete steps.
* Assign priorities, dependencies, estimated tools, and expected outputs.
* Validate every plan before it is returned (structure + dependency graph).
* Revise plans when execution fails, incorporating error feedback.
"""

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import structlog

from app.agents.base import AgentConfig, BaseAgent
from app.agents.core.tool_selector import ToolSelector
from app.models.agent import AgentType
from app.models.llm import (
    BaseLLMProvider,
    ChatCompletionRequest,
    ChatMessage,
    LLMProviderError,
    LLMUnavailableError,
)
from app.models.state import PlanStep

if TYPE_CHECKING:
    from app.memory.manager import MemoryManager
    from app.tools.registry import ToolRegistry


# ── Constants ────────────────────────────────────────────────────

VALID_PRIORITIES = {"low", "medium", "high", "critical"}

MAX_PLAN_STEPS = 20
MAX_REVISION_ATTEMPTS = 3

# Tokens embedded in a goal that hint at encoded data.  Lookaround
# assertions (rather than \b) delimit base64 tokens so the optional
# "=" padding stays part of the match.
_B64_PADDED_RE = re.compile(
    r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{4,}={1,2}(?![A-Za-z0-9+/=])"
)
_B64_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{8,}(?![A-Za-z0-9+/=])"
)
_HEX_TOKEN_RE = re.compile(r"\b[0-9a-fA-F]{8,}\b")
_URL_TOKEN_RE = re.compile(r"%[0-9a-fA-F]{2}")

_PLANNING_SYSTEM_PROMPT = """\
You are an expert planning agent.  Given a goal, decompose it into a
numbered, ordered execution plan.

For each step provide a JSON object with ALL of these keys:
  - index            (int)    – zero-based ordering
  - description      (str)    – concise action description
  - tool_name        (str|null) – tool to invoke, or null if none needed
  - tool_parameters  (object) – parameters for the tool (empty if none)
  - depends_on       (int[])  – indices of steps that must complete first
  - rationale        (str)    – why this step is needed
  - priority         (str)    – "low" | "medium" | "high" | "critical"
  - estimated_tools  (str[])  – tools *likely* needed (may include tool_name)
  - expected_output  (str|null) – what this step should produce

Reply with ONLY a JSON array of these objects.  No markdown fences, no prose.

Rules:
- Steps must be topologically sorted so that depends_on indices always
  refer to *earlier* steps.
- Keep the plan as concise as possible (aim for 2-8 steps).
- Every step MUST have a non-empty description.
- If no tool is required, set tool_name to null and tool_parameters to {}.
"""

_REVISE_SYSTEM_PROMPT = """\
You are an expert planning agent revising a previously failed plan.

You will receive:
  1. The original goal.
  2. The failed plan (JSON array).
  3. The execution errors encountered.
  4. Optional feedback.

Produce a **new** JSON array of plan steps that addresses every reported
error.  Follow the exact same schema as the original plan.

Reply with ONLY the JSON array.
"""


# ── PlannerAgent ─────────────────────────────────────────────────


class PlannerAgent(BaseAgent):
    """
    Agent that turns a user goal into a validated execution plan.

    Parameters
    ----------
    config : AgentConfig
        Standard agent configuration.  ``type`` should be PLANNER.
    llm_provider : BaseLLMProvider | None
        LLM backend for plan generation.  When ``None`` a template-based
        fallback is used.
    max_plan_steps : int
        Upper bound on the number of steps in a plan (default 20).
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        max_plan_steps: int = MAX_PLAN_STEPS,
        memory_manager: Optional["MemoryManager"] = None,
        tool_registry: Optional["ToolRegistry"] = None,
    ) -> None:
        if config is None:
            config = AgentConfig(
                name="PlannerAgent",
                type=AgentType.PLANNER,
                description="Decomposes goals into validated execution plans.",
            )
        super().__init__(config)
        self.llm_provider = llm_provider
        self.max_plan_steps = max_plan_steps
        self.memory_manager = memory_manager
        self.tool_registry = tool_registry
        self.tool_selector = ToolSelector()
        self.logger = self.logger.bind(component="planner_agent")

    # ── BaseAgent interface ───────────────────────────────────

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an execution plan from the supplied goal.

        Parameters
        ----------
        input_data : dict
            Must contain ``"goal"`` (str).  May also contain:
            - ``"tools_available"`` (list[str])
            - ``"tool_descriptions"`` (str) – formatted tool names with descriptions
            - ``"context"`` (str) – extra context for the LLM
            - ``"previous_plan"`` (list[dict]) – for revision
            - ``"errors"`` (list[str]) – execution errors to address
            - ``"feedback"`` (str) – human feedback for revision

        Returns
        -------
        dict
            ``{"success": bool, "plan": list[dict], "errors": list[str]}``
        """
        goal: str = input_data.get("goal", "")
        tools_available: List[str] = input_data.get("tools_available", [])
        tool_descriptions: str = input_data.get("tool_descriptions", "")
        context: str = input_data.get("context", "")
        previous_plan: Optional[List[Dict[str, Any]]] = input_data.get("previous_plan")
        errors: List[str] = input_data.get("errors", [])
        feedback: str = input_data.get("feedback", "")

        if not goal:
            return {"success": False, "plan": [], "errors": ["No goal provided."]}

        # Resolve the available tool set from an explicit list or a registry.
        tool_registry: ToolRegistry | None = (
            input_data.get("tool_registry") or self.tool_registry
        )
        if not tools_available and tool_registry is not None:
            tools_available = [t["name"] for t in tool_registry.list_tools()]

        # Decide whether to generate or revise
        if previous_plan and errors:
            self.logger.info("Revising existing plan", goal=goal, error_count=len(errors))
            steps = await self.revise_plan(previous_plan, errors, feedback, tools_available)
        else:
            # Enrich context with previous plans from memory
            enriched_context = context
            if self.memory_manager:
                enriched_context = await self._enrich_with_memory(goal, context)
            self.logger.info("Generating new plan", goal=goal)
            steps = await self._plan_from_llm(goal, tools_available, enriched_context, tool_descriptions)

        # Post-process: deterministic tool selection, deduplication, and
        # dependency fixing make the plan concrete and reproducible.
        steps = self._refine_plan(steps, tools_available)

        # Validate
        validation_errors = self.validate_plan(steps)
        if validation_errors:
            self.logger.warning("Plan validation failed", errors=validation_errors)
            return {"success": False, "plan": [], "errors": validation_errors}

        self.update_memory("last_goal", goal)
        self.update_memory("last_plan", [s.model_dump() for s in steps])

        return {
            "success": True,
            "plan": [s.model_dump() for s in steps],
            "errors": [],
        }

    async def reflect(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on the quality of a planning attempt.

        Parameters
        ----------
        result : dict
            The result dict returned by :meth:`process`.

        Returns
        -------
        dict
            Reflection insights.
        """
        plan = result.get("plan", [])
        errors = result.get("errors", [])
        success = result.get("success", False)

        insights: Dict[str, Any] = {
            "plan_generated": success,
            "step_count": len(plan),
            "had_errors": len(errors) > 0,
        }

        if success and plan:
            tool_steps = sum(1 for s in plan if s.get("tool_name"))
            insights["tool_steps"] = tool_steps
            insights["reasoning_steps"] = len(plan) - tool_steps

            # Check dependency density
            dep_count = sum(len(s.get("depends_on", [])) for s in plan)
            insights["avg_dependencies"] = dep_count / len(plan) if plan else 0

            priorities = [s.get("priority", "medium") for s in plan]
            insights["priority_distribution"] = {
                p: priorities.count(p) for p in set(priorities)
            }

        if not success:
            insights["failure_reasons"] = errors

        self.logger.info("Planner reflection", **insights)
        self.update_memory("last_reflection", insights)
        return insights

    # ── LLM-based planning ────────────────────────────────────

    async def _enrich_with_memory(self, goal: str, context: str) -> str:
        """
        Retrieve similar past plans from episodic memory and append
        them to the context so the LLM can learn from prior experience.
        """
        if not self.memory_manager:
            return context

        try:
            from app.models.memory import MemoryType
            results = await self.memory_manager.search_memory(
                query=goal,
                memory_type=MemoryType.EPISODIC,
                limit=2,
                min_score=0.3,
            )
            if not results:
                return context

            parts = [context] if context else []
            parts.append("Previous similar tasks and their outcomes:")
            for r in results:
                parts.append(f"- {r.memory.content} (score: {r.score:.2f})")
            return "\n".join(parts)
        except Exception as exc:
            self.logger.warning("Failed to enrich with memory", error=str(exc))
            return context

    async def _plan_from_llm(
        self,
        goal: str,
        tools_available: List[str],
        context: str = "",
        tool_descriptions: str = "",
    ) -> List[PlanStep]:
        """
        Ask the LLM to generate a plan for the given goal.

        Falls back to a template plan if the LLM is unavailable or
        returns unparseable output.
        """
        if self.llm_provider is None:
            self.logger.info("No LLM provider, using template plan")
            return self._template_fallback(goal, tools_available)

        if tool_descriptions:
            tools_section = tool_descriptions
        else:
            tools_section = ", ".join(tools_available) if tools_available else "none available"
        user_content = f"Goal: {goal}\n\nAvailable tools:\n{tools_section}"
        if context:
            user_content += f"\n\nAdditional context: {context}"

        messages = [
            ChatMessage(role="system", content=_PLANNING_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_content),
        ]

        try:
            request = ChatCompletionRequest(messages=messages)
            response = await self.llm_provider.chat_completion(request)
            raw = response.content.strip()
            steps = self._parse_plan_json(raw)
            if steps:
                return steps
            self.logger.warning("LLM returned empty/unparseable plan, falling back")
        except LLMUnavailableError:
            raise
        except LLMProviderError as exc:
            self.logger.error("LLM planning failed", error=str(exc))
        except Exception as exc:
            self.logger.error("Unexpected planning error", error=str(exc))

        return self._template_fallback(goal, tools_available)

    async def revise_plan(
        self,
        previous_plan: List[Dict[str, Any]],
        errors: List[str],
        feedback: str = "",
        tools_available: list[str] | None = None,
    ) -> List[PlanStep]:
        """
        Ask the LLM to revise a previously failed plan.

        Parameters
        ----------
        previous_plan : list[dict]
            The plan that was attempted.
        errors : list[str]
            Errors encountered during execution.
        feedback : str, optional
            Additional human feedback.
        tools_available : list[str], optional
            Tools that may be used in the revised plan.  When no LLM is
            configured this drives deterministic failure recovery.

        Returns
        -------
        list[PlanStep]
            Revised plan steps.
        """
        available = tools_available or []
        if self.llm_provider is None:
            self.logger.info("No LLM provider for revision, using recovery fallback")
            return self._recover_from_failures(previous_plan, errors, available)

        goal = self.get_memory("last_goal") or "unknown task"
        user_content = (
            f"Goal: {goal}\n\n"
            f"Previous plan:\n{json.dumps(previous_plan, indent=2)}\n\n"
            f"Errors:\n" + "\n".join(f"- {e}" for e in errors)
        )
        if feedback:
            user_content += f"\n\nFeedback: {feedback}"

        messages = [
            ChatMessage(role="system", content=_REVISE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_content),
        ]

        for attempt in range(1, MAX_REVISION_ATTEMPTS + 1):
            try:
                request = ChatCompletionRequest(messages=messages)
                response = await self.llm_provider.chat_completion(request)
                raw = response.content.strip()
                steps = self._parse_plan_json(raw)
                if steps:
                    self.logger.info(
                        "Plan revised successfully",
                        attempt=attempt,
                        step_count=len(steps),
                    )
                    return self._refine_plan(steps, available)
            except LLMUnavailableError:
                raise
            except Exception as exc:
                self.logger.warning(
                    "Revision attempt failed",
                    attempt=attempt,
                    error=str(exc),
                )

        self.logger.error("All revision attempts failed, using template fallback")
        return self._template_fallback(goal, available)

    # ── Template fallback ─────────────────────────────────────

    @staticmethod
    def _plan_from_template(goal: str) -> List[PlanStep]:
        """
        Generate a simple single-step plan when no LLM is available.

        Parameters
        ----------
        goal : str
            The goal to accomplish.

        Returns
        -------
        list[PlanStep]
            A single-step plan.
        """
        return [
            PlanStep(
                index=0,
                description=goal,
                rationale="Fallback template plan — no LLM provider available.",
                priority="medium",
                expected_output="Result of the requested task.",
            )
        ]

    def _template_fallback(
        self,
        goal: str,
        tools_available: list[str],
    ) -> list[PlanStep]:
        """
        Build a fallback plan, preferring a tool-aware template whenever
        available tools exist and the goal matches a known tool pattern.
        """
        if tools_available:
            steps = self._plan_from_template_for_tools(goal, tools_available)
            if steps:
                return steps
        return self._plan_from_template(goal)

    @classmethod
    def _plan_from_template_for_tools(
        cls,
        goal: str,
        tools_available: list[str],
    ) -> list[PlanStep]:
        """
        Build a template plan that makes use of the available tools.

        Returns an empty list when the goal does not clearly map onto any
        available tool so the caller can fall back to the generic template.
        """
        tools = set(tools_available)
        lower = goal.lower()

        # Strong per-tool signals.  "extract printable strings from binary"
        # must resolve to the strings tool, not the archive tool.
        strings_hint = any(
            kw in lower for kw in ("strings", "binary", "printable", "ascii")
        )
        archive_hint = any(
            kw in lower for kw in (
                "archive", "zip", "tar", "gzip", "7z", "unzip", "decompress"
            )
        )
        extract_hint = "extract" in lower

        if strings_hint and "strings" in tools:
            return [
                PlanStep(
                    index=0,
                    description=f"Extract printable strings: {goal}",
                    tool_name="strings",
                    tool_parameters={},
                    depends_on=[],
                    rationale="String extraction reveals human-readable content in binary data.",
                    priority="high",
                    estimated_tools=["strings"],
                    expected_output="List of printable strings",
                )
            ]

        if "archive" in tools and (archive_hint or extract_hint):
            steps: list[PlanStep] = [
                PlanStep(
                    index=0,
                    description=f"Extract archive: {goal}",
                    tool_name="archive",
                    tool_parameters={},
                    depends_on=[],
                    rationale="Archive contents must be unpacked before they can be analysed.",
                    priority="high",
                    estimated_tools=["archive"],
                    expected_output="Extracted file names and destination path",
                )
            ]
            next_index = 1
            if "strings" in tools:
                steps.append(
                    PlanStep(
                        index=next_index,
                        description="Extract printable strings from the extracted files",
                        tool_name="strings",
                        tool_parameters={},
                        depends_on=[next_index - 1],
                        rationale="String extraction surfaces hidden text inside binary files.",
                        priority="high",
                        estimated_tools=["strings"],
                        expected_output="List of printable strings found in extracted files",
                    )
                )
                next_index += 1
            if "file" in tools:
                steps.append(
                    PlanStep(
                        index=next_index,
                        description="Identify the file types of extracted files",
                        tool_name="file",
                        tool_parameters={},
                        depends_on=[next_index - 1],
                        rationale="File type identification guides further analysis of the payload.",
                        priority="medium",
                        estimated_tools=["file"],
                        expected_output="MIME type and description for each extracted file",
                    )
                )
            return steps

        encoding = cls._detect_encoding(goal)
        encoding_keywords = (
            "encode", "decode", "base64", "hex", "url", "cipher", "encoded", "decoded"
        )
        if "encoding" in tools and (
            encoding is not None or any(kw in lower for kw in encoding_keywords)
        ):
            params: dict[str, Any] = {"mode": "auto"}
            if encoding is not None:
                params["encoding"] = encoding[0]
                params["input"] = encoding[1]
            return [
                PlanStep(
                    index=0,
                    description=f"Encode or decode data for: {goal}",
                    tool_name="encoding",
                    tool_parameters=params,
                    depends_on=[],
                    rationale="Encoded data must be transformed before its meaning can be assessed.",
                    priority="high",
                    estimated_tools=["encoding"],
                    expected_output="Decoded or encoded output from the encoding tool",
                )
            ]

        return []

    @staticmethod
    def _detect_encoding(goal: str) -> tuple[str, str] | None:
        """
        Return ``(encoding, token)`` for an encoding hint embedded in a
        goal, or ``None`` when no encoded token is present.

        Padded base64 is checked first (its ``=`` suffix makes it the
        least ambiguous), then hex, then long unpadded base64, then URL
        percent-encoding.
        """
        padded_match = _B64_PADDED_RE.search(goal)
        if padded_match:
            return ("base64", padded_match.group(0))

        hex_match = _HEX_TOKEN_RE.search(goal)
        if hex_match and len(hex_match.group(0)) % 2 == 0:
            return ("hex", hex_match.group(0))

        b64_match = _B64_TOKEN_RE.search(goal)
        if b64_match:
            return ("base64", b64_match.group(0))

        url_match = _URL_TOKEN_RE.search(goal)
        if url_match:
            return ("url", url_match.group(0))

        return None

    # ── JSON parsing ──────────────────────────────────────────

    def _parse_plan_json(self, raw: str) -> List[PlanStep]:
        """
        Parse an LLM JSON response into a list of PlanStep objects.

        Handles markdown code fences and other common LLM quirks.
        """
        cleaned = self._strip_code_fences(raw)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            self.logger.warning("Failed to parse plan JSON", error=str(exc))
            return []

        if not isinstance(data, list):
            self.logger.warning("Plan JSON is not an array", type=type(data).__name__)
            return []

        steps: List[PlanStep] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            try:
                step = PlanStep(
                    index=item.get("index", i),
                    description=str(item.get("description", f"Step {i}")),
                    tool_name=item.get("tool_name"),
                    tool_parameters=item.get("tool_parameters") or {},
                    depends_on=item.get("depends_on") or [],
                    rationale=item.get("rationale", ""),
                    priority=item.get("priority", "medium"),
                    estimated_tools=item.get("estimated_tools") or [],
                    expected_output=item.get("expected_output"),
                )
                steps.append(step)
            except Exception as exc:
                self.logger.warning(
                    "Skipping malformed plan step",
                    index=i,
                    error=str(exc),
                )

        return steps

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences from LLM output."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines if they are fences
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)
        return cleaned.strip()

    # ── Plan refinement (tool selection) ──────────────────────

    def _refine_plan(
        self,
        steps: list[PlanStep],
        tools_available: list[str],
    ) -> list[PlanStep]:
        """
        Post-process a generated or revised plan.

        * Assigns the best available tool to every step that needs one.
        * Replaces references to unavailable tools.
        * Collapses duplicate tool invocations into reuses.
        """
        if not steps or not tools_available:
            return steps
        self._assign_tools(steps, tools_available)
        self._dedupe_steps(steps)
        return steps

    def _assign_tools(
        self,
        steps: list[PlanStep],
        tools_available: list[str],
    ) -> None:
        """
        Deterministically assign tools to plan steps.

        An explicit available tool name is kept; otherwise the
        :class:`ToolSelector` picks the best match from the step's
        description, rationale, and expected output.  Steps referencing a
        tool that is not available are downgraded to reasoning steps.
        """
        available = list(tools_available)
        for step in steps:
            current = step.tool_name
            if current and current in available:
                continue
            best = self.tool_selector.select_for_step(step, available)
            if best is not None:
                step.tool_name = best
                if not step.estimated_tools:
                    step.estimated_tools = [best]
            elif current and current not in available:
                self.logger.warning(
                    "Step references unavailable tool",
                    tool_name=current,
                    step_index=step.index,
                )
                step.tool_name = None
                step.tool_parameters = {}

    def _dedupe_steps(self, steps: list[PlanStep]) -> None:
        """
        Avoid invoking the same tool with the same parameters twice.

        When a later step would repeat an identical ``(tool, parameters)``
        pair, it is converted into a reasoning step that depends on the
        earlier step's output instead of issuing a second tool call.
        """
        seen: dict[tuple[str, str], int] = {}
        for step in steps:
            if not step.tool_name:
                continue
            key = (
                step.tool_name,
                json.dumps(step.tool_parameters or {}, sort_keys=True),
            )
            if key in seen:
                first_index = seen[key]
                self.logger.info(
                    "Deduplicating repeated tool invocation",
                    tool_name=step.tool_name,
                    step_index=step.index,
                    reuses_index=first_index,
                )
                step.tool_name = None
                step.tool_parameters = {}
                if first_index not in step.depends_on:
                    step.depends_on.append(first_index)
                step.rationale = (
                    f"Reuses the output of step {first_index} to avoid a "
                    "duplicate tool call."
                )
            else:
                seen[key] = step.index

    # ── Deterministic failure recovery ────────────────────────

    def _recover_from_failures(
        self,
        previous_plan: list[dict[str, Any]],
        errors: list[str],
        tools_available: list[str],
    ) -> list[PlanStep]:
        """
        Repair a failed plan without an LLM.

        Failed steps are dropped (along with their dependents) and replaced
        by an alternative approach.  The same tool/parameters that failed
        are never repeated; a different dedicated tool — or ``python`` as a
        general fallback — is used instead.
        """
        if not tools_available:
            goal = self.get_memory("last_goal") or "unknown task"
            return self._plan_from_template(goal)

        tools = set(tools_available)
        error_text = " ".join(errors).lower()

        # Identify failed step indices from executor-style error messages
        # like "Step 0: <message>" or "Tool 'archive' failed: <message>".
        failed_indices: set[int] = set()
        for err in errors:
            match = re.search(r"\bstep\s+(\d+)", err, re.IGNORECASE)
            if match:
                failed_indices.add(int(match.group(1)))
        failed_tools: list[str] = []
        for step in previous_plan:
            name = step.get("tool_name")
            if not name:
                continue
            index = step.get("index", 0)
            if index in failed_indices or name.lower() in error_text:
                failed_tools.append(name)

        # Keep only steps that did not fail and do not depend on a failure.
        surviving: list[PlanStep] = []
        for step in previous_plan:
            index = step.get("index", 0)
            if index in failed_indices:
                continue
            if any(dep in failed_indices for dep in step.get("depends_on", [])):
                continue
            surviving.append(self._to_plan_step(step))

        # Build one recovery step per failed tool, using an alternative.
        recovery: list[PlanStep] = []
        for failed_tool in failed_tools:
            alternative = self._alternative_tool(failed_tool, tools)
            if alternative is None:
                self.logger.warning(
                    "No alternative tool for failed tool",
                    tool_name=failed_tool,
                )
                continue
            recovery.append(
                PlanStep(
                    index=0,
                    description=(
                        f"Recover from failed '{failed_tool}' by using "
                        f"'{alternative}' instead"
                    ),
                    tool_name=alternative,
                    tool_parameters={},
                    depends_on=[],
                    rationale=(
                        f"'{failed_tool}' failed, so try '{alternative}' "
                        "instead of repeating the failed action."
                    ),
                    priority="high",
                    estimated_tools=[alternative],
                    expected_output="Successful result via the alternative tool",
                )
            )

        combined = surviving + recovery
        if not combined:
            goal = self.get_memory("last_goal") or "unknown task"
            return (
                self._plan_from_template_for_tools(goal, tools_available)
                or self._plan_from_template(goal)
            )
        return self._reindex(combined)

    @staticmethod
    def _alternative_tool(failed_tool: str, tools: set[str]) -> str | None:
        """Choose a different tool to try after *failed_tool* failed."""
        alternatives: dict[str, list[str]] = {
            "archive": ["python", "strings", "file"],
            "encoding": ["python", "file"],
            "strings": ["file", "python"],
            "file": ["python"],
            "python": ["file", "strings", "encoding", "archive"],
            "calculator": ["python"],
            "web_search": ["file"],
        }
        for candidate in alternatives.get(failed_tool, []):
            if candidate in tools:
                return candidate
        return None

    @staticmethod
    def _reindex(steps: list[PlanStep]) -> list[PlanStep]:
        """
        Renumber plan step indices contiguously and remap dependencies.
        """
        old_to_new = {step.index: i for i, step in enumerate(steps)}
        reindexed: list[PlanStep] = []
        for i, step in enumerate(steps):
            reindexed.append(
                PlanStep(
                    index=i,
                    description=step.description,
                    tool_name=step.tool_name,
                    tool_parameters=step.tool_parameters,
                    depends_on=[
                        old_to_new[d] for d in step.depends_on if d in old_to_new
                    ],
                    rationale=step.rationale,
                    priority=step.priority,
                    estimated_tools=step.estimated_tools,
                    expected_output=step.expected_output,
                )
            )
        return reindexed

    @staticmethod
    def _to_plan_step(step: dict[str, Any]) -> PlanStep:
        """Convert a serialised step dict into a :class:`PlanStep`."""
        return PlanStep(
            index=step.get("index", 0),
            description=str(step.get("description", "Step")),
            tool_name=step.get("tool_name"),
            tool_parameters=step.get("tool_parameters") or {},
            depends_on=step.get("depends_on") or [],
            rationale=step.get("rationale", ""),
            priority=step.get("priority", "medium"),
            estimated_tools=step.get("estimated_tools") or [],
            expected_output=step.get("expected_output"),
        )

    # ── Plan validation ───────────────────────────────────────

    def validate_plan(self, steps: List[PlanStep]) -> List[str]:
        """
        Validate a plan for structural and logical correctness.

        Parameters
        ----------
        steps : list[PlanStep]
            The plan to validate.

        Returns
        -------
        list[str]
            Empty list if valid, otherwise a list of error descriptions.
        """
        errors: List[str] = []

        if not steps:
            errors.append("Plan is empty.")
            return errors

        if len(steps) > self.max_plan_steps:
            errors.append(
                f"Plan has {len(steps)} steps, exceeding limit of {self.max_plan_steps}."
            )

        indices = set()
        for step in steps:
            # Index uniqueness
            if step.index in indices:
                errors.append(f"Duplicate step index {step.index}.")
            indices.add(step.index)

            # Description required
            if not step.description or not step.description.strip():
                errors.append(f"Step {step.index} has empty description.")

            # Priority valid
            if step.priority not in VALID_PRIORITIES:
                errors.append(
                    f"Step {step.index} has invalid priority '{step.index}'."
                )

        # Validate dependencies reference existing indices
        max_index = max(s.index for s in steps) if steps else -1
        for step in steps:
            for dep in step.depends_on:
                if dep not in indices:
                    errors.append(
                        f"Step {step.index} depends on non-existent step {dep}."
                    )
                if dep >= step.index:
                    errors.append(
                        f"Step {step.index} depends on step {dep} which is not earlier."
                    )

        # Cycle detection via DFS
        if not errors:
            cycle_error = self._detect_cycle(steps)
            if cycle_error:
                errors.append(cycle_error)

        return errors

    @staticmethod
    def _detect_cycle(steps: List[PlanStep]) -> Optional[str]:
        """
        Detect dependency cycles using DFS.

        Returns None if no cycle, otherwise an error description.
        """
        index_map = {s.index: s for s in steps}
        visited: set = set()
        in_stack: set = set()

        def dfs(node: int) -> Optional[str]:
            if node in in_stack:
                return f"Dependency cycle detected involving step {node}."
            if node in visited:
                return None
            visited.add(node)
            in_stack.add(node)
            step = index_map.get(node)
            if step:
                for dep in step.depends_on:
                    result = dfs(dep)
                    if result:
                        return result
            in_stack.discard(node)
            return None

        for step in steps:
            cycle = dfs(step.index)
            if cycle:
                return cycle
        return None
