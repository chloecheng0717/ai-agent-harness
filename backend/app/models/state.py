"""
Agent State Model

This module defines the shared state model for the agent runtime engine.
AgentState is the single source of truth that flows through the entire
lifecycle: analysis → planning → tool selection → execution → validation → reflection.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    """Status of an individual lifecycle step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class LifecycleStep(str, Enum):
    """Lifecycle steps in order."""
    INITIALIZE = "initialize"
    UNDERSTAND_TASK = "understand_task"
    CREATE_PLAN = "create_plan"
    SELECT_TOOLS = "select_tools"
    EXECUTE_PLAN = "execute_plan"
    EVALUATE_RESULT = "evaluate_result"
    REFLECT = "reflect"
    RESPOND = "respond"


class StepRecord(BaseModel):
    """Record of a single lifecycle step execution."""
    step: LifecycleStep
    status: StepStatus
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0


class TokenUsage(BaseModel):
    """Tracks token consumption across the lifecycle."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    requests: int = 0

    def add(self, prompt: int, completion: int, model: str = "") -> None:
        """
        Record a single LLM request's token usage.

        Args:
            prompt: Number of prompt tokens consumed
            completion: Number of completion tokens generated
            model: Model identifier used for this request
        """
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.requests += 1
        if model:
            self.model = model


class Message(BaseModel):
    """A single message in the agent conversation."""
    role: str = Field(..., description="Message role: 'system', 'user', 'assistant', 'tool'")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(default=None, description="Optional sender name")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Message timestamp",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional message metadata",
    )


class PlanStep(BaseModel):
    """A single step in an execution plan."""
    index: int = Field(..., description="Step ordering index")
    description: str = Field(..., description="What this step does")
    tool_name: Optional[str] = Field(
        default=None,
        description="Tool to use for this step, if any",
    )
    tool_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to the tool",
    )
    depends_on: List[int] = Field(
        default_factory=list,
        description="Indices of steps this step depends on",
    )
    rationale: str = Field(
        default="",
        description="Why this step is needed",
    )
    priority: str = Field(
        default="medium",
        description="Step priority: 'low', 'medium', 'high', 'critical'",
    )
    estimated_tools: List[str] = Field(
        default_factory=list,
        description="Tools likely needed to complete this step",
    )
    expected_output: Optional[str] = Field(
        default=None,
        description="Description of expected output from this step",
    )


class AgentState(BaseModel):
    """
    Shared state model for the agent runtime engine.

    This is the single source of truth passed through every lifecycle step.
    Each step reads from and writes to this state, creating a clear
    audit trail of the agent's reasoning and actions.

    Lifecycle flow:
        INITIALIZE → UNDERSTAND_TASK → CREATE_PLAN → SELECT_TOOLS
        → EXECUTE_PLAN → EVALUATE_RESULT → REFLECT → RESPOND

    Attributes:
        task_id: Unique identifier for the task being processed
        goal: The high-level objective extracted from the user request
        messages: Conversation messages for context
        plan: Ordered list of steps to accomplish the goal
        tools_available: Names of tools available for this task
        tools_used: Names of tools actually invoked
        results: Raw results from each tool invocation
        errors: Errors encountered during execution
        final_answer: The synthesized final response
        current_step: Which lifecycle step is currently running
        status: Overall task status
        metadata: Arbitrary metadata carried through the pipeline
        token_usage: Cumulative token consumption
        step_history: Ordered record of every lifecycle step execution
        retry_counts: Per-step retry counters
    """

    task_id: UUID = Field(
        default_factory=uuid4,
        description="Unique task identifier",
    )
    goal: str = Field(
        default="",
        description="High-level objective extracted from user request",
    )
    messages: List[Message] = Field(
        default_factory=list,
        description="Conversation messages for context",
    )
    plan: List[PlanStep] = Field(
        default_factory=list,
        description="Ordered execution plan",
    )
    tools_available: List[str] = Field(
        default_factory=list,
        description="Names of tools available for this task",
    )
    tools_used: List[str] = Field(
        default_factory=list,
        description="Names of tools actually invoked",
    )
    results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Raw results from each tool invocation",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Errors encountered during execution",
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="Synthesized final response",
    )
    current_step: LifecycleStep = Field(
        default=LifecycleStep.INITIALIZE,
        description="Currently executing lifecycle step",
    )
    status: StepStatus = Field(
        default=StepStatus.PENDING,
        description="Overall task status",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata",
    )
    token_usage: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Cumulative token consumption",
    )
    step_history: List[StepRecord] = Field(
        default_factory=list,
        description="Ordered record of lifecycle step executions",
    )
    retry_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Per-step retry counters",
    )

    # ── helpers ──────────────────────────────────────────────

    def add_message(self, role: str, content: str, **kwargs: Any) -> Message:
        """
        Append a message to the conversation history.

        Args:
            role: Message role ('system', 'user', 'assistant', 'tool')
            content: Message content
            **kwargs: Additional message metadata

        Returns:
            The created Message instance
        """
        msg = Message(role=role, content=content, metadata=kwargs)
        self.messages.append(msg)
        return msg

    def add_error(self, error: str) -> None:
        """
        Record an error.

        Args:
            error: Error description
        """
        self.errors.append(error)

    def record_step(
        self,
        step: LifecycleStep,
        status: StepStatus,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> StepRecord:
        """
        Record the completion of a lifecycle step.

        Finds the matching in-progress record in step_history and marks it
        completed (or failed). If no in-progress record exists, creates one.

        Args:
            step: Which lifecycle step completed
            status: Final status of the step
            output: Optional output data from the step
            error: Optional error message

        Returns:
            The updated or created StepRecord
        """
        record: Optional[StepRecord] = None
        for r in reversed(self.step_history):
            if r.step == step and r.status == StepStatus.RUNNING:
                record = r
                break

        if record is None:
            record = StepRecord(step=step, status=StepStatus.RUNNING)
            self.step_history.append(record)

        record.status = status
        record.completed_at = datetime.now(timezone.utc)
        if record.started_at:
            delta = record.completed_at - record.started_at
            record.duration_ms = delta.total_seconds() * 1000
        record.output = output
        record.error = error
        return record

    def get_retry_count(self, step: str) -> int:
        """Get retry count for a named step."""
        return self.retry_counts.get(step, 0)

    def increment_retry(self, step: str) -> int:
        """
        Increment and return the retry count for a step.

        Args:
            step: Step name to increment

        Returns:
            New retry count
        """
        self.retry_counts[step] = self.retry_counts.get(step, 0) + 1
        return self.retry_counts[step]

    def to_result(self) -> Dict[str, Any]:
        """
        Serialize state into a result dictionary suitable for API responses.

        Returns:
            Dict with task_id, status, final_answer, errors, token_usage, history
        """
        return {
            "task_id": str(self.task_id),
            "status": self.status.value,
            "goal": self.goal,
            "final_answer": self.final_answer,
            "errors": self.errors,
            "tools_used": self.tools_used,
            "token_usage": self.token_usage.model_dump(),
            "step_history": [
                {
                    "step": r.step.value,
                    "status": r.status.value,
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                    "retry_count": r.retry_count,
                }
                for r in self.step_history
            ],
        }
