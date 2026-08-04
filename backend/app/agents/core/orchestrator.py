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
import re
import time
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, List, Optional

import structlog

from app.agents.core.challenge_classifier import (
    NETWORK,
    build_network_plan,
    classify_challenge,
    extract_endpoint,
)
from app.agents.core.tool_selector import ToolSelector
from app.config import get_settings
from app.models.llm import (
    BaseLLMProvider,
    ChatCompletionRequest,
    ChatMessage,
    LLMProviderError,
    LLMUnavailableError,
)
from app.models.state import (
    AgentState,
    LifecycleStep,
    PlanStep,
    StepRecord,
    StepStatus,
    extract_llm_tool_name,
    extract_llm_tool_params,
    parse_llm_plan_json,
)
from app.models.task import TaskStatus
from app.models.tool import ToolExecutionRequest, ToolPermission

if TYPE_CHECKING:
    from app.agents.specialized.planner import PlannerAgent
    from app.tools.registry import ToolRegistry
    from app.memory.manager import MemoryManager
    from app.sandbox.docker_sandbox import DockerSandbox


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


# ── deterministic completion heuristics ─────────────────────────

#: A recovered flag-like token means the objective is done.  Used to
#: hard-confirm completion from real tool output regardless of the LLM
#: evaluator's verdict.
_FLAG_PATTERN_RE = re.compile(r"(?:flag|ctf|academy)\{[^\}]+\}", re.IGNORECASE)

#: A token that is entirely base64 (with padding, or containing the
#: distinctive ``+``/``/`` characters) is still encoded content.
_ENCODED_BASE64_RE = re.compile(
    r"^[A-Za-z0-9+/]{8,}={1,2}$|^[A-Za-z0-9+/]{16,}[+/][A-Za-z0-9+/]*$"
)
#: A token that is entirely hex and at least 10 bytes long is still
#: encoded content (e.g. the first decode of a base64-of-hex flag).
_ENCODED_HEX_RE = re.compile(r"^(?:[0-9a-fA-F]{2}){10,}$")
#: URL percent-encoding is still encoded content.
_URL_ESCAPE_RE = re.compile(r"%[0-9a-fA-F]{2}")

_HEX_CHARS = frozenset("0123456789abcdefABCDEF")
_BASE64_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
)


def _looks_like_encoded_token(token: str) -> bool:
    """Whether a single whitespace-free token is predominantly encoded data."""
    if not token:
        return False
    if _ENCODED_BASE64_RE.match(token):
        return True
    if _ENCODED_HEX_RE.match(token):
        return True
    if _URL_ESCAPE_RE.search(token):
        return len(_URL_ESCAPE_RE.findall(token)) >= 3
    return False


def _text_looks_encoded(text: str) -> bool:
    """
    Whether execution output still looks encoded, compressed, or wrapped.

    Used by :meth:`Orchestrator.evaluate_result` to refuse to declare a
    task complete while the work product is still base64/hex/URL data or a
    compression stream.  The check is deliberately conservative: a long,
    predominantly-alphabetic word or a short hex fragment does not count.
    """
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 20:
        hexish = sum(1 for c in compact if c in _HEX_CHARS)
        if hexish / len(compact) > 0.85 and len(compact) % 2 == 0:
            return True
        b64ish = sum(1 for c in compact if c in _BASE64_CHARS)
        if b64ish / len(compact) > 0.85 and any(
            marker in compact for marker in ("=", "+", "/")
        ):
            return True
    for line in text.splitlines():
        token = line.strip()
        if token and _looks_like_encoded_token(token):
            return True
    return False


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
        sandbox: Optional["DockerSandbox"] = None,
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
        self.sandbox = sandbox

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

        # Caches successful tool outputs within one plan-execute cycle so
        # identical tool calls are never repeated.  Cleared each iteration.
        self._execution_cache: dict[str, dict[str, Any]] = {}

        self.logger.info(
            "Orchestrator initialised",
            has_llm_provider=llm_provider is not None,
            has_planner=planner is not None,
            has_tool_registry=tool_registry is not None,
            has_memory_manager=memory_manager is not None,
            has_sandbox=sandbox is not None,
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
            if defn and defn.parameters:
                summary = defn.parameter_summary()
                if summary:
                    parts.append(summary)
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
        if state.metadata.get("force_replan"):
            parts.append(
                "MANDATORY: the previous iterations made no progress (the "
                "same tool calls produced the same output every time). "
                "Produce a fundamentally different plan: do not repeat any "
                "prior tool call, do not re-read the same file, and do not "
                "re-decode the same input."
            )
        if state.metadata.get("previous_plan"):
            prev_descs = [
                p.get("description", "")
                for p in state.metadata["previous_plan"]
            ]
            parts.append(
                "Previous plan steps: " + " -> ".join(prev_descs)
            )
        return "\n".join(parts)

    # ── deterministic network handling ────────────────────────

    def _network_enabled(self) -> bool:
        """
        Whether the sandbox allows outbound TCP connections.

        When no sandbox is attached (unit tests, per-call workspaces) the
        network is treated as available so the deterministic plan can be
        built and tested.  A sandbox with networking disabled blocks the
        override so offline runs never attempt a real connection.
        """
        if self.sandbox is None:
            return True
        try:
            return not bool(self.sandbox.config.network_disabled)
        except Exception:
            return True

    def _build_network_plan(self, state: AgentState) -> AgentState:
        """
        Populate a deterministic socket plan when the goal targets a remote
        TCP service, overriding LLM/template planning.

        The classification text combines the (possibly LLM-summarised) goal
        with the raw user description so a ``host:port`` endpoint is found
        even when the summariser dropped it.
        """
        if state.plan:
            return state
        if "python" not in state.tools_available:
            return state
        if not self._network_enabled():
            return state

        user_text = ""
        for message in reversed(state.messages):
            if message.role == "user":
                user_text = message.content
                break
        text = f"{state.goal}\n{user_text}".strip()

        if classify_challenge(text) != NETWORK:
            return state
        endpoint = extract_endpoint(text)
        if endpoint is None:
            return state

        host, port = endpoint
        state.plan = build_network_plan(host, port)
        state.metadata["plan_source"] = "network_detected"
        state.metadata["challenge_category"] = NETWORK
        state.metadata["network_endpoint"] = f"{host}:{port}"
        self.logger.info(
            "Network challenge detected; using deterministic socket plan",
            host=host,
            port=port,
        )
        return state

    # ── progress guard ────────────────────────────────────────

    def _iteration_tool_signatures(
        self,
        state: AgentState,
    ) -> list[tuple[str, str]]:
        """
        Fingerprint the tool invocations of the current plan as sorted
        ``(tool_name, canonical parameters)`` pairs.
        """
        signatures: set[tuple[str, str]] = set()
        for plan_step in state.plan:
            if not plan_step.tool_name:
                continue
            signatures.add(
                (
                    plan_step.tool_name,
                    json.dumps(plan_step.tool_parameters or {}, sort_keys=True),
                )
            )
        return sorted(signatures)

    def _update_progress_tracking(self, state: AgentState, iteration: int) -> None:
        """
        Detect plan-execute loops that make no progress — the same tool
        calls producing the same output every iteration — and flag them so
        the next plan is forced to change strategy.
        """
        history = state.metadata.setdefault("execution_history", [])
        signatures = self._iteration_tool_signatures(state)
        evidence = self._results_evidence_text(state)
        entry: dict[str, Any] = {
            "iteration": iteration,
            "tool_signatures": signatures,
            "evidence": evidence,
            "flag_found": bool(_FLAG_PATTERN_RE.search(evidence)),
        }

        streak = 0
        if history:
            previous = history[-1]
            same_tools = previous.get("tool_signatures") == entry["tool_signatures"]
            same_evidence = previous.get("evidence") == evidence
            if same_tools and same_evidence:
                streak = state.metadata.get("no_progress_streak", 0) + 1
        state.metadata["no_progress_streak"] = streak
        history.append(entry)

        if streak >= 2:
            state.metadata["force_replan"] = True
            state.metadata["replan_feedback"] = (
                "No progress has been made: the same tool calls have "
                "produced the same output every iteration. Do NOT repeat any "
                "of them. Keep already-fetched data in the workspace and "
                "change strategy completely (interact with the remote "
                "service, try a different encoding, or inspect a different "
                "file)."
            )
            self.logger.warning(
                "No progress across iterations — forcing replan",
                streak=streak,
                iteration=iteration,
            )

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

        # Create one persistent workspace for the entire run so that
        # every tool call shares the same sandbox directory. The ID is
        # stored on the state and consumed by execute_plan; it is removed
        # when the lifecycle finishes.
        workspace_id: Optional[str] = self._create_workspace()
        if workspace_id is not None:
            state.metadata["workspace_id"] = workspace_id

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
                    self._execution_cache.clear()

                for step in _inner_steps:
                    state = await self._run_step(step, state)

                if state.metadata.get("task_complete", False):
                    self.logger.info(
                        "Task completed successfully",
                        iteration=iteration,
                    )
                    break

                self._update_progress_tracking(state, iteration)

                self.logger.info(
                    "Task not yet complete — re-planning",
                    iteration=iteration,
                    error_count=len(state.errors),
                )

            # ── Phase 3: Finalisation (always runs once) ─────────────
            state = await self._run_step(LifecycleStep.REFLECT, state)
            state = await self._run_step(LifecycleStep.RESPOND, state)

        except LLMUnavailableError as exc:
            state.status = StepStatus.FAILED
            state.add_error(str(exc))
            state.metadata["llm_unavailable"] = True
            self.logger.error(
                "Lifecycle aborted: LLM unavailable",
                task_id=str(state.task_id),
                error=str(exc),
            )
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
        finally:
            self._cleanup_workspace(workspace_id)

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
        result = state.to_result()
        if state.metadata.get("llm_unavailable"):
            result["llm_unavailable"] = True
        return result

    # ── step runner with retry ───────────────────────────────

    def _create_workspace(self) -> Optional[str]:
        """
        Create a persistent workspace for this run (when a sandbox is set).

        Returns the workspace ID, or ``None`` when no sandbox is
        configured (tools then fall back to per-call workspaces).
        """
        if self.sandbox is None:
            return None
        try:
            workspace = self.sandbox.create_workspace()
            self.logger.info(
                "Created persistent workspace for run",
                workspace_id=workspace.workspace_id,
            )
            return workspace.workspace_id
        except Exception as exc:
            self.logger.warning(
                "Failed to create persistent workspace",
                error=str(exc),
            )
            return None

    def _cleanup_workspace(self, workspace_id: Optional[str]) -> None:
        """Remove the persistent workspace for a finished run."""
        if workspace_id is None or self.sandbox is None:
            return
        try:
            self.sandbox.delete_workspace(workspace_id)
            self.logger.info(
                "Deleted persistent workspace for run",
                workspace_id=workspace_id,
            )
        except Exception as exc:
            self.logger.warning(
                "Failed to delete persistent workspace",
                workspace_id=workspace_id,
                error=str(exc),
            )

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

            except LLMUnavailableError as exc:
                ms = (time.monotonic() - step_start) * 1000
                state.record_step(step, StepStatus.FAILED, error=str(exc))
                self.logger.error(
                    "Step aborted: LLM unavailable",
                    step=step.value,
                    error=str(exc),
                    ms=round(ms, 2),
                )
                raise

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
            except LLMUnavailableError:
                raise
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

        # A goal that targets a remote TCP service gets a deterministic
        # socket plan instead of LLM planning, so the service is interacted
        # with rather than re-decoded as if it were encoded data.
        state = self._build_network_plan(state)
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
                    "tool_registry": self.tool_registry,
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
            except LLMUnavailableError:
                raise
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
                    "'description' (str), 'tool_name' (str or null), "
                    "'rationale' (str), and 'tool_parameters' (object) with the "
                    "exact arguments that tool needs for that step. "
                    "Reply ONLY with the JSON array.\n\n"
                    "Tool guidance:\n"
                    "- Use 'strings' to extract hidden/embedded/printable text "
                    "from files, logs, and binaries. Prefer it whenever the goal "
                    "is to find hidden text inside a file.\n"
                    "- Use 'file' ONLY to inspect a file's type or metadata. "
                    "Never use it as the only step when the goal is to recover "
                    "hidden content.\n"
                    "- When data is encoded, decode ONE layer per step. If a "
                    "decode still yields encoded-looking text (hex, base64, or "
                    "URL-escaped), add a further step to decode the next layer.\n"
                    "- A challenge that points at a remote service ('Service "
                    "endpoint', 'host:port', 'nc host port', netcat, socket/TCP "
                    "instructions) is a NETWORK challenge: use the 'python' "
                    "tool to open a socket, read the banner, and follow the "
                    "service's commands to recover the flag. Do NOT treat the "
                    "hostname or banner as encoded data to decode, and do not "
                    "re-read the same file or re-decode the same input.\n"
                    "- Keep planning until the flag text (matching 'flag{...}') "
                    "is recovered."
                ),
                ChatMessage(role="user", content=f"Goal: {state.goal}\n\n{self._plan_context(state)}".strip()),
            ]
            try:
                raw = await self._llm_chat(messages, state)
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
                    state.metadata["plan_source"] = "orchestrator_llm"
                    return state
            except LLMUnavailableError:
                raise
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

        Uses a deterministic :class:`ToolSelector` to assign a concrete tool
        to every step that needs one.  Dedicated tools are preferred over
        generic fallbacks, and steps referencing unavailable tools are
        downgraded to reasoning steps.  The chosen tools are recorded in
        ``state.metadata["selected_tools"]``.

        Parameters
        ----------
        state : AgentState
            Current shared state (with plan populated).

        Returns
        -------
        AgentState
            State ready for execution.
        """
        if self.tool_registry is None:
            return state

        available_tools = [t["name"] for t in self.tool_registry.list_tools()]
        selector = ToolSelector()
        selected: list[str] = []

        for plan_step in state.plan:
            best = selector.select_for_step(plan_step, available_tools)
            if best is not None and plan_step.tool_name != best:
                self.logger.info(
                    "Selected tool for step",
                    step_index=plan_step.index,
                    previous_tool=plan_step.tool_name,
                    selected_tool=best,
                )
                plan_step.tool_name = best
            elif plan_step.tool_name and plan_step.tool_name not in available_tools:
                self.logger.warning(
                    "Tool not in registry",
                    tool_name=plan_step.tool_name,
                    step_index=plan_step.index,
                )
                plan_step.tool_name = None

            if plan_step.tool_name:
                selected.append(plan_step.tool_name)

        state.metadata["selected_tools"] = selected
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
                    # Reuse the output of an identical earlier successful call
                    # so the same tool is not invoked twice.
                    cache_key = (
                        f"{plan_step.tool_name}:"
                        + json.dumps(plan_step.tool_parameters or {}, sort_keys=True)
                    )
                    if cache_key in self._execution_cache:
                        prior = self._execution_cache[cache_key]
                        result["success"] = True
                        result["output"] = prior.get("output")
                        result["reused_output_of"] = prior.get("step_index")
                        self.logger.info(
                            "Reusing previous tool output",
                            step_index=plan_step.index,
                            reused_from=prior.get("step_index"),
                        )
                    else:
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
                            workspace_id=state.metadata.get("workspace_id"),
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
                            else:
                                self._execution_cache[cache_key] = {
                                    "step_index": plan_step.index,
                                    "output": tool_result.output,
                                }
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
            except LLMUnavailableError:
                raise
            except Exception:
                self.logger.warning("LLM evaluate_result failed, using heuristic")
                self._evaluate_heuristic(state)
        else:
            self._evaluate_heuristic(state)

        # The LLM verdict is a hint, not a contract.  Inspect the REAL tool
        # output: a recovered flag hard-confirms completion, while output
        # that is still encoded/compressed means the run must keep going
        # instead of stopping after the first successful transformation.
        self._enforce_deterministic_completion(state)

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

    # ── deterministic completion guard ─────────────────────────

    @staticmethod
    def _result_output_text(result: dict[str, Any]) -> str:
        """
        Extract the meaningful text from a single execution result's output.

        Handles the structured output shapes the CTF tools return
        (``stdout`` from ``python``, ``output`` from ``encoding``, and the
        ``strings`` list) and plain strings.
        """
        output = result.get("output")
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            for key in ("stdout", "output", "decoded", "encoded"):
                value = output.get(key)
                if isinstance(value, str) and value:
                    return value
            strings = output.get("strings")
            if isinstance(strings, list):
                return "\n".join(str(s) for s in strings if s)
        return ""

    def _results_evidence_text(self, state: AgentState) -> str:
        """Concatenate every real tool output produced this iteration."""
        return "\n".join(
            self._result_output_text(result)
            for result in state.results
            if self._result_output_text(result)
        )

    def _enforce_deterministic_completion(self, state: AgentState) -> None:
        """
        Reconcile the LLM/heuristic completion verdict with the actual tool
        output so the run neither stops early nor loops forever.

        * When a flag-like token appears in real tool output the task is
          done, even if the evaluator wrongly said ``COMPLETE: NO``.
        * When the output is still encoded/compressed (and no flag is
          present) the task is NOT done, even if the evaluator lazily said
          ``COMPLETE: YES``; the loop must continue decoding.
        """
        evidence = self._results_evidence_text(state)
        if _FLAG_PATTERN_RE.search(evidence):
            state.metadata["task_complete"] = True
            return

        if _text_looks_encoded(evidence):
            state.metadata["task_complete"] = False
            state.metadata["replan_feedback"] = (
                "The latest tool output is still encoded (base64/hex/URL) or "
                "compressed. Decode or unwrap every remaining layer until the "
                "flag is revealed, then report it."
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
            except LLMUnavailableError:
                raise
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
            except LLMUnavailableError:
                raise
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
