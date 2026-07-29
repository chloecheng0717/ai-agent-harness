"""
Orchestrator Module — Agent Runtime Engine

This module provides the main orchestrator that coordinates the entire
agent workflow.  It drives the lifecycle:

    INITIALIZE → UNDERSTAND_TASK → CREATE_PLAN → SELECT_TOOLS
    → EXECUTE_PLAN → EVALUATE_RESULT → REFLECT → RESPOND

The orchestrator is intentionally LLM-agnostic: concrete LLM calls are
delegated to pluggable provider interfaces that will be implemented later.
Each lifecycle step is an async method that reads and writes AgentState,
with structured logging, exception handling, and retry support.
"""

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, List, Optional

import structlog

from app.config import get_settings
from app.models.llm import (
    BaseLLMProvider,
    ChatCompletionRequest,
    ChatMessage,
    LLMProviderError,
)
from app.models.state import (
    AgentState,
    LifecycleStep,
    PlanStep,
    StepRecord,
    StepStatus,
)
from app.models.task import TaskStatus
from app.models.tool import ToolExecutionRequest, ToolPermission

if TYPE_CHECKING:
    from app.agents.specialized.planner import PlannerAgent
    from app.tools.registry import ToolRegistry
    from app.memory.manager import MemoryManager


# ── retry decorator ─────────────────────────────────────────────


class RetryExhaustedError(Exception):
    """Raised when all retry attempts for a step have been exhausted."""

    def __init__(self, step: str, attempts: int, last_error: str) -> None:
        self.step = step
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Step '{step}' failed after {attempts} attempts: {last_error}"
        )


# Type alias for async step functions
StepFn = Callable[[AgentState], Coroutine[Any, Any, AgentState]]


# ── Orchestrator ────────────────────────────────────────────────


class Orchestrator:
    """
    Main orchestrator — the Agent Runtime Engine.

    Responsibilities
    ----------------
    * Drive the eight-step lifecycle for every task.
    * Wrap each step in logging, timing, exception handling, and retry.
    * Expose hook points (``on_step_*``) for observability.
    * Delegate actual work (LLM calls, tool execution) to callbacks that
      will be injected by higher-level components.

    Attributes
    ----------
    settings : Application settings
    logger  : Structured logger bound to the orchestrator component
    max_retries : Maximum retry attempts per step (from config)
    max_iterations : Maximum plan-execute-evaluate loop iterations
    _step_registry : Mapping of lifecycle step → handler function
    """

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        planner: Optional["PlannerAgent"] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        memory_manager: Optional["MemoryManager"] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        self.settings = get_settings()
        self.logger = structlog.get_logger().bind(component="orchestrator")
        self.max_retries: int = 3
        self.max_iterations: int = max_iterations or self.settings.max_iterations
        self.llm_provider = llm_provider
        self.planner = planner
        self.tool_registry = tool_registry
        self.memory_manager = memory_manager

        # Step registry — maps lifecycle steps to handler methods.
        # Subclasses or callers can override individual entries.
        self._step_registry: Dict[LifecycleStep, StepFn] = {
            LifecycleStep.INITIALIZE: self.initialize,
            LifecycleStep.UNDERSTAND_TASK: self.understand_task,
            LifecycleStep.CREATE_PLAN: self.create_plan,
            LifecycleStep.SELECT_TOOLS: self.select_tools,
            LifecycleStep.EXECUTE_PLAN: self.execute_plan,
            LifecycleStep.EVALUATE_RESULT: self.evaluate_result,
            LifecycleStep.REFLECT: self.reflect,
            LifecycleStep.RESPOND: self.respond,
        }

        self.logger.info(
            "Orchestrator initialised",
            has_llm_provider=llm_provider is not None,
            has_planner=planner is not None,
            has_tool_registry=tool_registry is not None,
            has_memory_manager=memory_manager is not None,
        )

    # ── LLM helpers ──────────────────────────────────────────

    async def _llm_chat(
        self,
        messages: List[ChatMessage],
        state: AgentState,
        **kwargs: Any,
    ) -> str:
        """
        Send a chat request through the LLM provider and return the text.

        Falls back to an empty string when no provider is configured.
        Token usage is automatically recorded on *state*.

        Parameters
        ----------
        messages : list[ChatMessage]
            Messages to send.
        state : AgentState
            Current state (used for token tracking).
        **kwargs
            Extra parameters forwarded to ``ChatCompletionRequest``.

        Returns
        -------
        str
            The assistant's reply text.
        """
        if self.llm_provider is None:
            return ""

        request = ChatCompletionRequest(messages=messages, **kwargs)
        try:
            response = await self.llm_provider.chat_completion(request)
            state.token_usage.add(
                prompt=response.usage.prompt_tokens,
                completion=response.usage.completion_tokens,
                model=response.model,
            )
            return response.content
        except LLMProviderError as exc:
            self.logger.error("LLM call failed", error=str(exc))
            state.add_error(f"LLM error: {exc}")
            raise
        except Exception as exc:
            self.logger.error("Unexpected LLM error", error=str(exc))
            state.add_error(f"LLM error: {exc}")
            raise

    def _build_system_message(self, content: str) -> ChatMessage:
        return ChatMessage(role="system", content=content)

    def _state_to_messages(self, state: AgentState) -> List[ChatMessage]:
        """Convert the current message history to LLM-compatible messages."""
        return [ChatMessage(role=m.role, content=m.content) for m in state.messages]

    def _format_tool_descriptions(self, tool_names: List[str]) -> str:
        """Build a formatted string of tool names with descriptions from the registry."""
        if not tool_names:
            return "none"
        if self.tool_registry is None:
            return ", ".join(tool_names)
        parts: List[str] = []
        for name in tool_names:
            defn = self.tool_registry.get_tool_definition(name)
            if defn and defn.description:
                parts.append(f"  - {name}: {defn.description}")
            else:
                parts.append(f"  - {name}")
        return "\n".join(parts)

    def _plan_context(self, state: AgentState) -> str:
        """Build extra context string for the PlannerAgent."""
        parts: List[str] = []
        iteration = state.metadata.get("iteration", 1)
        parts.append(f"Iteration: {iteration}")
        if state.metadata.get("replan_errors"):
            parts.append(
                "Previous execution failed with errors: "
                + "; ".join(state.metadata["replan_errors"])
            )
        if state.metadata.get("replan_feedback"):
            parts.append(f"Feedback: {state.metadata['replan_feedback']}")
        if state.metadata.get("previous_plan"):
            prev_descs = [
                p.get("description", "")
                for p in state.metadata["previous_plan"]
            ]
            parts.append(
                "Previous plan steps: " + " -> ".join(prev_descs)
            )
        return "\n".join(parts)

    # ── public entry point ───────────────────────────────────

    async def run(
        self,
        task_id: Any,
        task_data: Dict[str, Any],
        tools_available: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full agent lifecycle for a task.

        This is the main entry point. It creates an :class:`AgentState`
        and drives a three-phase execution:

        1. **Initialisation** — ``INITIALIZE``, ``UNDERSTAND_TASK``
        2. **Iterative loop** — ``CREATE_PLAN → SELECT_TOOLS → EXECUTE_PLAN
           → EVALUATE_RESULT``, repeating up to *max_iterations* times until
           the task is complete.
        3. **Finalisation** — ``REFLECT → RESPOND``

        Parameters
        ----------
        task_id : UUID or str
            Identifier for the task.
        task_data : dict
            Must contain at least ``"description"``.  May also include
            ``"type"``, ``"metadata"``, ``"constraints"``, etc.
        tools_available : list[str], optional
            Tool names that may be used during execution.

        Returns
        -------
        dict
            Serialised result from :meth:`AgentState.to_result`.
        """
        state = AgentState(
            task_id=task_id,
            tools_available=tools_available or [],
        )
        state.add_message("user", task_data.get("description", ""))
        state.metadata.update(task_data.get("metadata", {}))

        self.logger.info(
            "Lifecycle started",
            task_id=str(state.task_id),
            max_iterations=self.max_iterations,
        )

        overall_start = time.monotonic()

        try:
            # ── Phase 1: Initialisation (always runs once) ───────────
            for step in (LifecycleStep.INITIALIZE, LifecycleStep.UNDERSTAND_TASK):
                state = await self._run_step(step, state)

            # ── Phase 2: Iterative plan-execute-evaluate loop ─────────
            _inner_steps = (
                LifecycleStep.CREATE_PLAN,
                LifecycleStep.SELECT_TOOLS,
                LifecycleStep.EXECUTE_PLAN,
                LifecycleStep.EVALUATE_RESULT,
            )

            for iteration in range(1, self.max_iterations + 1):
                state.metadata["iteration"] = iteration
                self.logger.info(
                    "Plan-execute-evaluate iteration",
                    iteration=iteration,
                    max_iterations=self.max_iterations,
                )

                if iteration > 1:
                    # Stash previous execution context before clearing
                    if state.errors:
                        state.metadata["replan_errors"] = state.errors.copy()
                    if state.plan:
                        state.metadata["previous_plan"] = [
                            p.model_dump() for p in state.plan
                        ]
                    state.plan.clear()
                    state.results.clear()
                    state.errors.clear()
                    state.tools_used.clear()

                for step in _inner_steps:
                    state = await self._run_step(step, state)

                if state.metadata.get("task_complete", False):
                    self.logger.info(
                        "Task completed successfully",
                        iteration=iteration,
                    )
                    break

                self.logger.info(
                    "Task not yet complete — re-planning",
                    iteration=iteration,
                    error_count=len(state.errors),
                )

            # ── Phase 3: Finalisation (always runs once) ─────────────
            state = await self._run_step(LifecycleStep.REFLECT, state)
            state = await self._run_step(LifecycleStep.RESPOND, state)

        except RetryExhaustedError as exc:
            state.status = StepStatus.FAILED
            state.add_error(str(exc))
            self.logger.error(
                "Lifecycle aborted",
                task_id=str(state.task_id),
                failed_step=exc.step,
                error=exc.last_error,
            )
        except Exception as exc:
            state.status = StepStatus.FAILED
            state.add_error(str(exc))
            self.logger.error(
                "Lifecycle failed unexpectedly",
                task_id=str(state.task_id),
                error=str(exc),
                error_type=type(exc).__name__,
            )

        overall_ms = (time.monotonic() - overall_start) * 1000
        self.logger.info(
            "Lifecycle finished",
            task_id=str(state.task_id),
            status=state.status.value,
            iterations=state.metadata.get("iteration", 1),
            overall_ms=round(overall_ms, 2),
            token_usage=state.token_usage.total_tokens,
        )

        await self.close()
        return state.to_result()

    # ── step runner with retry ───────────────────────────────

    async def _run_step(self, step: LifecycleStep, state: AgentState) -> AgentState:
        """
        Run a single lifecycle step with retry and error handling.

        Parameters
        ----------
        step : LifecycleStep
            The step to execute.
        state : AgentState
            Current shared state.

        Returns
        -------
        AgentState
            Updated state after step execution.
        """
        handler = self._step_registry.get(step)
        if handler is None:
            self.logger.warning("No handler registered for step", step=step.value)
            state.record_step(step, StepStatus.SKIPPED)
            return state

        state.current_step = step
        last_error: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            record = StepRecord(step=step, status=StepStatus.RUNNING)
            state.step_history.append(record)

            step_start = time.monotonic()
            self.logger.debug(
                "Step starting",
                step=step.value,
                attempt=attempt,
                max_retries=self.max_retries,
            )

            try:
                state = await handler(state)
                ms = (time.monotonic() - step_start) * 1000
                state.record_step(step, StepStatus.COMPLETED)
                self.logger.debug(
                    "Step completed",
                    step=step.value,
                    ms=round(ms, 2),
                )
                return state

            except Exception as exc:
                ms = (time.monotonic() - step_start) * 1000
                last_error = str(exc)
                retry_count = state.increment_retry(step.value)
                state.record_step(
                    step,
                    StepStatus.FAILED,
                    error=last_error,
                )
                self.logger.warning(
                    "Step failed",
                    step=step.value,
                    attempt=attempt,
                    error=last_error,
                    ms=round(ms, 2),
                )

                if attempt < self.max_retries:
                    backoff = min(2 ** attempt, 30)
                    self.logger.debug(
                        "Retrying step",
                        step=step.value,
                        backoff_s=backoff,
                    )
                    await asyncio.sleep(backoff)

        raise RetryExhaustedError(step.value, self.max_retries, last_error or "")

    # ── lifecycle steps ──────────────────────────────────────

    async def initialize(self, state: AgentState) -> AgentState:
        """
        Initialize the runtime for this task.

        Sets up initial context, validates inputs, and prepares
        the state for downstream steps.

        Parameters
        ----------
        state : AgentState
            Current shared state (with user message already set).

        Returns
        -------
        AgentState
            State ready for task understanding.
        """
        state.status = StepStatus.RUNNING
        state.add_message(
            "system",
            "Agent runtime initialized. Processing task.",
        )
        return state

    async def understand_task(self, state: AgentState) -> AgentState:
        """
        Analyse the user request and extract a clear goal.

        When an LLM provider is configured, this step asks the model to
        distil the user's intent into a concise goal statement.  Without
        a provider the last user message is used verbatim.

        Parameters
        ----------
        state : AgentState
            Current shared state.

        Returns
        -------
        AgentState
            State with ``goal`` populated.
        """
        user_messages = [m for m in state.messages if m.role == "user"]
        if not user_messages:
            return state

        if self.llm_provider is not None:
            messages = [
                self._build_system_message(
                    "You are a task analyst. Given the user's request, "
                    "extract a clear, concise goal statement. Reply with "
                    "only the goal text."
                ),
                ChatMessage(
                    role="user",
                    content=user_messages[-1].content,
                ),
            ]
            try:
                goal = await self._llm_chat(messages, state)
                if goal.strip():
                    state.goal = goal.strip()
            except Exception:
                self.logger.warning("LLM understand_task failed, falling back to raw message")
                state.goal = user_messages[-1].content
        else:
            state.goal = user_messages[-1].content

        return state

    async def create_plan(self, state: AgentState) -> AgentState:
        """
        Decompose the goal into an ordered list of :class:`PlanStep`.

        When a :class:`PlannerAgent` is attached, it is used for structured
        plan generation with validation, dependencies, priorities, and
        expected outputs.  When only an LLM provider is available the
        simpler orchestrator-level planning is used.  Otherwise a single-step
        fallback is produced.

        Parameters
        ----------
        state : AgentState
            Current shared state (with goal populated).

        Returns
        -------
        AgentState
            State with ``plan`` populated.
        """
        if state.plan:
            return state

        # ── PlannerAgent path ─────────────────────────────────
        tools_desc = self._format_tool_descriptions(state.tools_available)
        if self.planner is not None:
            try:
                result = await self.planner.process({
                    "goal": state.goal,
                    "tools_available": state.tools_available,
                    "tool_descriptions": tools_desc,
                    "context": self._plan_context(state),
                })
                if result["success"] and result["plan"]:
                    state.plan = [
                        PlanStep(
                            index=item["index"],
                            description=item["description"],
                            tool_name=item.get("tool_name"),
                            tool_parameters=item.get("tool_parameters", {}),
                            depends_on=item.get("depends_on", []),
                            rationale=item.get("rationale", ""),
                            priority=item.get("priority", "medium"),
                            estimated_tools=item.get("estimated_tools", []),
                            expected_output=item.get("expected_output"),
                        )
                        for item in result["plan"]
                    ]
                    state.metadata["plan_source"] = "planner_agent"
                    return state
                else:
                    self.logger.warning(
                        "PlannerAgent failed",
                        errors=result.get("errors", []),
                    )
            except Exception as exc:
                self.logger.warning(
                    "PlannerAgent raised exception, falling back",
                    error=str(exc),
                )

        # ── LLM orchestrator path ─────────────────────────────
        if self.llm_provider is not None:
            messages = [
                self._build_system_message(
                    "You are a planning agent. Given a goal, produce a numbered "
                    "execution plan.\n\n"
                    f"Available tools:\n{tools_desc}\n\n"
                    "Reply with a JSON array of objects, each having 'index' (int), "
                    "'description' (str), 'tool_name' (str or null), and "
                    "'rationale' (str). Reply ONLY with the JSON array."
                ),
                ChatMessage(role="user", content=f"Goal: {state.goal}\n\n{self._plan_context(state)}".strip()),
            ]
            try:
                raw = await self._llm_chat(messages, state)
                plan_data = __import__("json").loads(raw)
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
                    return state
            except Exception:
                self.logger.warning("LLM create_plan failed, using fallback plan")

        # ── Template fallback ─────────────────────────────────
        state.plan = [
            PlanStep(
                index=0,
                description=f"Execute task: {state.goal}",
                rationale="Single-step fallback plan",
            )
        ]
        state.metadata["plan_source"] = "fallback"
        return state

    async def select_tools(self, state: AgentState) -> AgentState:
        """
        Map each plan step to the tools needed to execute it.

        This step is the hook point for LLM-based tool selection.
        The default implementation does nothing — tool selection will
        be delegated to a dedicated selector component later.

        When a ToolRegistry is available, this step can validate that
        the tools in the plan are actually registered.

        Parameters
        ----------
        state : AgentState
            Current shared state (with plan populated).

        Returns
        -------
        AgentState
            State ready for execution.
        """
        # Validate tools in plan against registry if available
        if self.tool_registry is not None:
            available_tools = {
                t["name"] for t in self.tool_registry.list_tools()
            }
            
            for plan_step in state.plan:
                if plan_step.tool_name and plan_step.tool_name not in available_tools:
                    self.logger.warning(
                        "Tool not in registry",
                        tool_name=plan_step.tool_name,
                        step_index=plan_step.index,
                    )
        
        return state

    async def execute_plan(self, state: AgentState) -> AgentState:
        """
        Execute every step in the plan sequentially.

        Each plan step is executed in order.  Results are appended
        to ``state.results`` and errors to ``state.errors``.

        When a ToolRegistry is available, tool execution is delegated to it.
        Otherwise, a placeholder message is used.

        Parameters
        ----------
        state : AgentState
            Current shared state (with plan populated).

        Returns
        -------
        AgentState
            State with execution results populated.
        """
        for plan_step in state.plan:
            result: Dict[str, Any] = {
                "step_index": plan_step.index,
                "description": plan_step.description,
                "tool_name": plan_step.tool_name,
                "success": True,
                "output": None,
            }

            if plan_step.tool_name:
                state.tools_used.append(plan_step.tool_name)
                
                # Use ToolRegistry if available
                if self.tool_registry is not None:
                    # Create execution request with default permissions
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
                    
                    try:
                        tool_result = await self.tool_registry.execute(request)
                        result["success"] = tool_result.success
                        result["output"] = tool_result.output
                        if not tool_result.success:
                            result["error"] = tool_result.error
                            state.add_error(
                                f"Tool '{plan_step.tool_name}' failed: {tool_result.error}"
                            )
                    except Exception as exc:
                        result["success"] = False
                        result["error"] = str(exc)
                        state.add_error(
                            f"Tool '{plan_step.tool_name}' execution error: {exc}"
                        )
                else:
                    # Fallback when no tool registry
                    result["output"] = (
                        f"Tool '{plan_step.tool_name}' execution "
                        "delegated to tool runtime (no registry available)"
                    )
            else:
                result["output"] = (
                    f"Step {plan_step.index} completed "
                    "(no tool required — LLM reasoning placeholder)"
                )

            state.results.append(result)

        return state

    async def evaluate_result(self, state: AgentState) -> AgentState:
        """
        Validate execution results and check goal completion.

        With an LLM provider this step asks the model to evaluate whether
        the goal was achieved; without one a simple error-count heuristic
        is used.

        The outcome is stored in ``state.metadata["task_complete"]`` so
        that the outer loop in :meth:`run` knows whether to continue
        iterating.

        Parameters
        ----------
        state : AgentState
            Current shared state (with results populated).

        Returns
        -------
        AgentState
            State with evaluation recorded.
        """
        if self.llm_provider is not None:
            results_text = "\n".join(
                f"- Step {r.get('step_index', '?')}: {r.get('output', 'N/A')}"
                for r in state.results
            )
            errors_text = "\n".join(state.errors) if state.errors else "None"
            messages = [
                self._build_system_message(
                    "You are an execution evaluator. Given the goal, results, "
                    "and errors, evaluate whether the task was completed "
                    "successfully. Begin your response with COMPLETE: YES or "
                    "COMPLETE: NO, then provide a brief assessment."
                ),
                ChatMessage(
                    role="user",
                    content=(
                        f"Goal: {state.goal}\n\n"
                        f"Results:\n{results_text}\n\n"
                        f"Errors:\n{errors_text}"
                    ),
                ),
            ]
            try:
                evaluation = await self._llm_chat(messages, state)
                state.add_message("assistant", evaluation or "Evaluation complete.")
                self._set_task_complete_from_eval(state, evaluation)
                if not state.metadata.get("task_complete"):
                    state.metadata["replan_feedback"] = evaluation
            except Exception:
                self.logger.warning("LLM evaluate_result failed, using heuristic")
                self._evaluate_heuristic(state)
        else:
            self._evaluate_heuristic(state)

        return state

    def _set_task_complete_from_eval(self, state: AgentState, evaluation: str) -> None:
        """Parse the LLM evaluation response for the completion verdict."""
        stripped = evaluation.strip().upper()
        if stripped.startswith("COMPLETE: YES"):
            state.metadata["task_complete"] = True
        elif stripped.startswith("COMPLETE: NO"):
            state.metadata["task_complete"] = False
        else:
            state.metadata["task_complete"] = len(state.errors) == 0

    def _evaluate_heuristic(self, state: AgentState) -> None:
        """Fallback heuristic-based evaluation."""
        if state.errors:
            state.metadata["task_complete"] = False
            state.add_message(
                "assistant",
                f"Execution completed with {len(state.errors)} error(s).",
            )
        else:
            state.metadata["task_complete"] = True
            state.add_message(
                "assistant",
                "Execution completed successfully.",
            )

    async def reflect(self, state: AgentState) -> AgentState:
        """
        Reflect on the execution and extract lessons learned.

        With an LLM provider the model is asked to produce a short
        reflection; otherwise a summary is logged.

        Parameters
        ----------
        state : AgentState
            Current shared state.

        Returns
        -------
        AgentState
            State with reflection recorded.
        """
        summary = {
            "goal": state.goal,
            "steps_executed": len(state.results),
            "tools_used": list(set(state.tools_used)),
            "errors": len(state.errors),
            "tokens": state.token_usage.total_tokens,
        }

        if self.llm_provider is not None:
            messages = [
                self._build_system_message(
                    "You are a reflective agent. Given the task summary, "
                    "provide a brief reflection on what went well, what "
                    "could be improved, and any lessons learned. Be concise."
                ),
                ChatMessage(
                    role="user",
                    content=(
                        f"Goal: {state.goal}\n"
                        f"Steps executed: {len(state.results)}\n"
                        f"Tools used: {', '.join(set(state.tools_used)) or 'none'}\n"
                        f"Errors: {len(state.errors)}\n"
                        f"Tokens consumed: {state.token_usage.total_tokens}"
                    ),
                ),
            ]
            try:
                reflection = await self._llm_chat(messages, state)
                if reflection:
                    state.metadata["reflection"] = reflection
            except Exception:
                self.logger.warning("LLM reflect failed, logging summary only")

        self.logger.info("Reflection summary", **summary)

        # Store reflection in long-term memory
        if self.memory_manager:
            try:
                from app.models.memory import MemoryType
                reflection_text = (
                    f"Reflection for goal '{state.goal}': "
                    f"executed {len(state.results)} steps, "
                    f"used {', '.join(set(state.tools_used)) or 'no'} tools, "
                    f"{len(state.errors)} errors."
                )
                await self.memory_manager.store_memory(
                    content=reflection_text,
                    memory_type=MemoryType.LONG_TERM,
                    task_id=state.task_id,
                    metadata={"reflection": summary},
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to store reflection", error=str(exc)
                )

        return state

    async def respond(self, state: AgentState) -> AgentState:
        """
        Synthesise the final answer from all gathered results.

        With an LLM provider the model composes a coherent final answer;
        without one result outputs are concatenated.

        Parameters
        ----------
        state : AgentState
            Current shared state.

        Returns
        -------
        AgentState
            State with ``final_answer`` populated and status set to COMPLETED.
        """
        if self.llm_provider is not None:
            results_text = "\n".join(
                self._format_result_output(r) for r in state.results
            )
            messages = [
                self._build_system_message(
                    "You are a helpful assistant. Given the goal and execution "
                    "results, produce a clear, complete final answer for the "
                    "user. Be concise but thorough."
                ),
                ChatMessage(
                    role="user",
                    content=(
                        f"Goal: {state.goal}\n\n"
                        f"Execution results:\n{results_text or 'No results available.'}"
                    ),
                ),
            ]
            try:
                answer = await self._llm_chat(messages, state)
                state.final_answer = answer.strip() if answer and answer.strip() else self._fallback_answer(state)
            except Exception:
                self.logger.warning("LLM respond failed, concatenating results")
                state.final_answer = self._fallback_answer(state)
        else:
            state.final_answer = self._fallback_answer(state)

        state.status = StepStatus.COMPLETED

        # Store final answer in short-term memory
        if self.memory_manager and state.final_answer:
            try:
                from app.models.memory import MemoryType
                await self.memory_manager.store_memory(
                    content=state.final_answer,
                    memory_type=MemoryType.SHORT_TERM,
                    task_id=state.task_id,
                    metadata={"goal": state.goal},
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to store final answer", error=str(exc)
                )

        return state

    @staticmethod
    def _format_result_output(r: Dict[str, Any]) -> str:
        """Extract a readable summary from a single result dict.

        Tool outputs are often structured dicts (e.g. ``{"stdout": ...}``,
        ``{"strings": [...]}``).  This method pulls out the meaningful
        fields rather than ``str()``-dumping the whole dict.
        """
        output = r.get("output")
        step_label = f"Step {r.get('step_index', '?')}"
        if r.get("tool_name"):
            step_label += f" ({r['tool_name']})"
        if not output:
            return f"{step_label}: (no output)"
        if isinstance(output, str):
            return f"{step_label}: {output[:500]}"
        if isinstance(output, dict):
            # Known tool output shapes
            if "stdout" in output:
                text = output["stdout"]
                if len(text) > 500:
                    text = text[:500] + "..."
                return f"{step_label}: {text}"
            if "strings" in output:
                items = output["strings"]
                if items:
                    text = "\n".join(items)
                    if len(text) > 500:
                        text = text[:500] + "..."
                    return f"{step_label} ({len(items)} strings):\n{text}"
                return f"{step_label}: (no strings found)"
            if "mime_type" in output:
                return f"{step_label}: {output['mime_type']} ({output.get('description', '')})"
            if "decoded" in output or "encoded" in output:
                return f"{step_label}: decoded={output.get('decoded', '')!r}  encoded={output.get('encoded', '')!r}"
            # Generic dict fallback
            return f"{step_label}: {json.dumps(output, default=str)[:500]}"
        return f"{step_label}: {str(output)[:500]}"

    @staticmethod
    def _fallback_answer(state: AgentState) -> str:
        """Build a fallback answer by concatenating formatted result outputs."""
        parts: List[str] = []
        for r in state.results:
            formatted = Orchestrator._format_result_output(r)
            if formatted:
                parts.append(formatted)
        return "\n".join(parts) if parts else "No result produced."

    # ── backwards-compatible entry point ─────────────────────

    async def process_task(
        self,
        task_id: Any,
        task_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Legacy entry point that delegates to :meth:`run`.

        Kept for backward compatibility with existing route handlers.

        Parameters
        ----------
        task_id : UUID or str
            Task identifier.
        task_data : dict
            Task payload.

        Returns
        -------
        dict
            Result dictionary with ``success``, ``task_id``, ``result``,
            and ``history`` keys.
        """
        result = await self.run(task_id, task_data)

        return {
            "success": result["status"] == "completed",
            "task_id": result["task_id"],
            "result": result.get("final_answer"),
            "history": result.get("step_history", []),
        }

    # ── cleanup ───────────────────────────────────────────────

    async def close(self) -> None:
        """Release resources held by the LLM provider."""
        if self.llm_provider is not None:
            await self.llm_provider.close()
