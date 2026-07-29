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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import structlog

from app.agents.base import AgentConfig, BaseAgent
from app.models.agent import AgentType
from app.models.llm import (
    BaseLLMProvider,
    ChatCompletionRequest,
    ChatMessage,
    LLMProviderError,
)
from app.models.state import AgentState, PlanStep

if TYPE_CHECKING:
    from app.memory.manager import MemoryManager


# ── Constants ────────────────────────────────────────────────────

VALID_PRIORITIES = {"low", "medium", "high", "critical"}

MAX_PLAN_STEPS = 20
MAX_REVISION_ATTEMPTS = 3

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

        # Decide whether to generate or revise
        if previous_plan and errors:
            self.logger.info("Revising existing plan", goal=goal, error_count=len(errors))
            steps = await self.revise_plan(previous_plan, errors, feedback)
        else:
            # Enrich context with previous plans from memory
            enriched_context = context
            if self.memory_manager:
                enriched_context = await self._enrich_with_memory(goal, context)
            self.logger.info("Generating new plan", goal=goal)
            steps = await self._plan_from_llm(goal, tools_available, enriched_context, tool_descriptions)

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
            return self._plan_from_template(goal)

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
        except LLMProviderError as exc:
            self.logger.error("LLM planning failed", error=str(exc))
        except Exception as exc:
            self.logger.error("Unexpected planning error", error=str(exc))

        return self._plan_from_template(goal)

    async def revise_plan(
        self,
        previous_plan: List[Dict[str, Any]],
        errors: List[str],
        feedback: str = "",
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

        Returns
        -------
        list[PlanStep]
            Revised plan steps.
        """
        if self.llm_provider is None:
            self.logger.info("No LLM provider for revision, using template fallback")
            goal = self.get_memory("last_goal") or "unknown task"
            return self._plan_from_template(goal)

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
                    return steps
            except Exception as exc:
                self.logger.warning(
                    "Revision attempt failed",
                    attempt=attempt,
                    error=str(exc),
                )

        self.logger.error("All revision attempts failed, using template fallback")
        return self._plan_from_template(goal)

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
