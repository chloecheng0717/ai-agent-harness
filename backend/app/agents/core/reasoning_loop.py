"""
Reasoning Loop Module

This module implements the ReAct (Reason + Act) pattern for agent reasoning.
It provides the core reasoning loop that agents use to process tasks.

Includes built-in flag detection that terminates the loop when a CTF flag
pattern is found in tool output or agent reasoning.
"""

import re
from typing import Any, Dict, List, Optional

import structlog

from app.config import get_settings

# Default patterns for CTF flag detection
DEFAULT_FLAG_PATTERNS: List[str] = [
    r"flag\{[^\}]+\}",
    r"CTF\{[^\}]+\}",
]


class ReasoningStep:
    """
    A single step in the reasoning loop.
    
    Attributes:
        thought: Agent's reasoning about the current state
        action: Action to take
        action_input: Input for the action
        observation: Result of the action
    """
    
    def __init__(
        self,
        thought: str,
        action: str,
        action_input: Dict[str, Any],
        observation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a reasoning step.
        
        Args:
            thought: Agent's reasoning
            action: Action to take
            action_input: Action input
            observation: Action result
        """
        self.thought = thought
        self.action = action
        self.action_input = action_input
        self.observation = observation


class FlagDetector:
    """
    Scans text for CTF flag patterns.

    Attributes:
        patterns: Compiled regex patterns to match.
    """

    def __init__(
        self,
        flag_patterns: Optional[List[str]] = None,
    ) -> None:
        """
        Initialise the detector.

        Parameters
        ----------
        flag_patterns : list[str] | None
            Regex strings.  Defaults to ``DEFAULT_FLAG_PATTERNS``.
        """
        raw = flag_patterns if flag_patterns else DEFAULT_FLAG_PATTERNS
        self.patterns = [re.compile(p, re.IGNORECASE) for p in raw]
        self.logger = structlog.get_logger().bind(component="flag_detector")

    def scan(self, text: str) -> Optional[str]:
        """
        Return the first flag match found in *text*, or ``None``.
        """
        if not text:
            return None
        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                self.logger.info(
                    "Flag detected",
                    flag=match.group(0),
                )
                return match.group(0)
        return None

    def scan_dict(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Recursively scan a dictionary (or list of dicts) for flag patterns.

        Handles nested dicts, lists, and string values.
        """
        if isinstance(data, str):
            return self.scan(data)
        if isinstance(data, dict):
            for value in data.values():
                result = self.scan_dict(value)
                if result:
                    return result
        if isinstance(data, list):
            for item in data:
                result = self.scan_dict(item)
                if result:
                    return result
        return None


class ReasoningLoop:
    """
    ReAct reasoning loop implementation.
    
    This class implements the Reason-Act-Observe pattern:
    1. Reason: Agent thinks about the current state
    2. Act: Agent decides on an action
    3. Observe: Agent observes the result
    
    The loop continues until:
    - The task is completed
    - Maximum iterations are reached
    - A CTF flag is detected in output (flag detection)
    - An error occurs
    
    Attributes:
        settings: Application settings
        logger: Structured logger
        max_iterations: Maximum number of iterations
        flag_detector: FlagDetector instance for flag detection
    """
    
    def __init__(
        self,
        max_iterations: Optional[int] = None,
        flag_patterns: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the reasoning loop.
        
        Args:
            max_iterations: Maximum iterations (uses config if not provided)
            flag_patterns: Custom regex patterns for flag detection.
                Defaults to ``DEFAULT_FLAG_PATTERNS``.
        """
        self.settings = get_settings()
        self.max_iterations = max_iterations or self.settings.max_iterations
        self.flag_detector = FlagDetector(flag_patterns)
        
        self.logger = structlog.get_logger().bind(component="reasoning_loop")
        self.logger.info(
            "Reasoning loop initialized",
            max_iterations=self.max_iterations,
        )
    
    async def run(
        self,
        initial_state: Dict[str, Any],
        reasoning_fn: Any,
        action_fn: Any,
        tool_registry: Any = None,
    ) -> Dict[str, Any]:
        """
        Run the reasoning loop.

        This method executes the ReAct loop until completion or max iterations.

        Args:
            initial_state: Initial state for reasoning.  When the state
                contains a ``"plan"`` list the loop becomes plan-driven: it
                walks the plan in dependency order, records each
                observation, and does not repeat failed steps.
            reasoning_fn: Async function for reasoning (thought generation).
                Skipped when the loop is plan-driven or an action is queued,
                which reduces unnecessary LLM calls.
            action_fn: Async function for action execution.
            tool_registry: Optional ToolRegistry.  When set and the chosen
                action matches a registered tool, the tool is executed
                through the registry so the loop only trusts real tool
                outputs.

        Returns:
            Dict containing final state and reasoning history
        """
        self.logger.debug("Starting reasoning loop")

        state = initial_state.copy()
        history: List[ReasoningStep] = []
        found_flag: Optional[str] = None
        plan_driven = isinstance(state.get("plan"), list) and bool(state["plan"])

        for iteration in range(self.max_iterations):
            self.logger.debug(
                "Reasoning iteration",
                iteration=iteration + 1,
                max_iterations=self.max_iterations,
            )

            # Reason: Generate thought.  Skipped in plan-driven mode and
            # when an action is already queued — this avoids needless LLM
            # calls while preserving the Plan → Execute → Observe loop.
            if plan_driven:
                thought = "Executing next plan step."
            elif state.get("next_action") is not None:
                thought = "Proceeding with the queued action."
            else:
                thought = await reasoning_fn(state)

            # Act: Decide and execute action
            action, action_input = await self._decide_action(thought, state)

            # Observe: Get action result.  Tool actions go through the real
            # registry when one is provided so observations reflect actual
            # tool output rather than model guesses.
            if tool_registry is not None and action in self._registry_tools(tool_registry):
                observation = await self._execute_tool(tool_registry, action, action_input)
            else:
                observation = await action_fn(action, action_input)

            # Check observation for flags
            if isinstance(observation, dict):
                found_flag = self.flag_detector.scan_dict(observation)
            elif isinstance(observation, str):
                found_flag = self.flag_detector.scan(observation)

            # Record step
            step = ReasoningStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            )
            history.append(step)

            # Update state
            state = await self._update_state(state, step)

            # Terminate if flag found
            if found_flag:
                self.logger.info(
                    "Flag found in observation — terminating loop",
                    flag=found_flag,
                )
                state["found_flag"] = found_flag
                break

            # Check if task is complete
            if await self._is_complete(state):
                self.logger.debug("Task completed in reasoning loop")
                break

        return {
            "state": state,
            "history": [
                {
                    "thought": step.thought,
                    "action": step.action,
                    "action_input": step.action_input,
                    "observation": step.observation,
                }
                for step in history
            ],
            "iterations": len(history),
            "found_flag": found_flag,
        }

    async def _decide_action(
        self,
        thought: str,
        state: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """
        Decide on an action based on the current thought and state.

        Priority:
        1. A queued ``next_action`` in state (consumed here).
        2. The next runnable plan step, in dependency order.  Steps that
           already executed (successfully or not) are never repeated.
        3. Fall back to a thinking step for free-form reasoning.

        Args:
            thought: Current reasoning thought
            state: Current state

        Returns:
            Tuple of (action_name, action_input)
        """
        # 1. Consume a pre-queued action.
        queued = state.get("next_action")
        if isinstance(queued, dict) and queued:
            state["next_action"] = None
            state["_current_step_index"] = queued.get("step_index")
            return queued.get("action", "think"), queued.get("action_input") or {}

        # 2. Plan-driven mode.
        plan = state.get("plan")
        if isinstance(plan, list) and plan:
            for step in plan:
                if step.get("executed"):
                    continue
                deps = step.get("depends_on") or []
                if not all(self._step_succeeded(plan, dep) for dep in deps):
                    continue
                state["_current_step_index"] = step.get("index")
                tool = step.get("tool_name")
                action = tool if tool else "think"
                return action, step.get("tool_parameters") or {}

        # 3. Free-form reasoning fallback.
        state["_current_step_index"] = None
        return "think", {"thought": thought}

    async def _update_state(
        self,
        state: Dict[str, Any],
        step: ReasoningStep,
    ) -> Dict[str, Any]:
        """
        Update state based on reasoning step.

        Records every action/observation, marks failed actions so they are
        never repeated, and advances plan-driven execution by flagging the
        current step as executed.
        """
        actions = state.setdefault("actions_taken", [])
        actions.append(
            {
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
            }
        )

        observations = state.setdefault("observations", [])
        observations.append(step.observation)

        success = self._observation_success(step.observation)
        if not success:
            state.setdefault("failed_actions", []).append(
                {
                    "action": step.action,
                    "action_input": step.action_input,
                }
            )

        plan = state.get("plan")
        current_index = state.get("_current_step_index")
        if isinstance(plan, list) and current_index is not None:
            for plan_step in plan:
                if plan_step.get("index") == current_index:
                    plan_step["executed"] = True
                    plan_step["success"] = success
                    plan_step["observation"] = step.observation

        return state

    async def _is_complete(self, state: Dict[str, Any]) -> bool:
        """
        Check if the task is complete.

        Returns True when:
        * A flag was found.
        * ``task_complete`` is set in state.
        * Every step of a plan-driven run has executed.
        """
        if state.get("found_flag"):
            return True
        if state.get("task_complete"):
            return True

        plan = state.get("plan")
        if isinstance(plan, list) and plan:
            return all(step.get("executed") for step in plan)

        return False

    # ── tool-aware execution helpers ──────────────────────────

    @staticmethod
    def _registry_tools(tool_registry: Any) -> list[str]:
        """Return the registered tool names for a ToolRegistry."""
        try:
            return [t["name"] for t in tool_registry.list_tools()]
        except Exception:
            return []

    @staticmethod
    async def _execute_tool(
        tool_registry: Any,
        action: str,
        action_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool through the registry and return a real result."""
        tool = tool_registry.get_tool(action)
        if tool is None:
            return {"success": False, "error": f"Tool '{action}' not found"}

        from app.models.tool import ToolExecutionRequest, ToolPermission

        request = ToolExecutionRequest(
            tool_name=action,
            parameters=action_input or {},
            agent_permissions=[
                ToolPermission.READ,
                ToolPermission.WRITE,
                ToolPermission.EXECUTE,
            ],
        )
        result = await tool_registry.execute(request)
        return {
            "tool_name": action,
            "success": result.success,
            "output": result.output,
            "error": result.error,
        }

    @staticmethod
    def _step_succeeded(plan: list[dict[str, Any]], index: int) -> bool:
        """Whether the plan step at *index* executed successfully."""
        for step in plan:
            if step.get("index") == index:
                return bool(step.get("success"))
        return False

    @staticmethod
    def _observation_success(observation: Any) -> bool:
        """Determine whether an observation indicates success."""
        if isinstance(observation, dict):
            if "success" in observation:
                return bool(observation.get("success"))
            if observation.get("error"):
                return False
        return True
