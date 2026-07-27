"""
Execution Models

This module defines Pydantic models for executor agent execution tracking.
These models capture the results, status, and history of plan step executions.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    """Status of an individual step execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"
    RETRYING = "retrying"


class ExecutionResult(BaseModel):
    """
    Result of executing a single plan step.

    Attributes:
        step_index: Index of the plan step that was executed.
        tool_name: Name of the tool that was invoked, if any.
        success: Whether execution completed successfully.
        output: Output data produced by the tool or reasoning step.
        error: Error message if execution failed.
        execution_time_ms: Wall-clock time of the execution in milliseconds.
        retry_count: Number of retries attempted before final result.
        timeout: Whether the execution timed out.
        metadata: Arbitrary metadata about the execution.
    """
    step_index: int = Field(..., description="Index of the executed plan step")
    tool_name: Optional[str] = Field(
        default=None,
        description="Tool invoked, if any",
    )
    success: bool = Field(..., description="Whether execution succeeded")
    output: Any = Field(default=None, description="Execution output data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time_ms: float = Field(
        default=0,
        description="Execution time in milliseconds",
    )
    retry_count: int = Field(
        default=0,
        description="Number of retries before final result",
    )
    timeout: bool = Field(
        default=False,
        description="Whether execution timed out",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata",
    )


class ExecutionRecord(BaseModel):
    """
    Full record of a plan step's execution lifecycle.

    Tracks the step from pending through completion or failure,
    including all retry attempts.

    Attributes:
        record_id: Unique identifier for this execution record.
        step_index: Index of the plan step.
        description: Description of the plan step.
        tool_name: Tool invoked, if any.
        status: Current execution status.
        started_at: When execution started.
        completed_at: When execution finished.
        duration_ms: Total wall-clock duration in milliseconds.
        result: Final ExecutionResult after all attempts.
        attempts: List of individual attempt results.
        error_message: Aggregated error message from failed attempts.
    """
    record_id: UUID = Field(
        default_factory=uuid4,
        description="Unique record identifier",
    )
    step_index: int = Field(..., description="Index of the plan step")
    description: str = Field(default="", description="Step description")
    tool_name: Optional[str] = Field(default=None, description="Tool invoked")
    status: ExecutionStatus = Field(
        default=ExecutionStatus.PENDING,
        description="Current execution status",
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="When execution started",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When execution finished",
    )
    duration_ms: Optional[float] = Field(
        default=None,
        description="Total duration in milliseconds",
    )
    result: Optional[ExecutionResult] = Field(
        default=None,
        description="Final execution result",
    )
    attempts: List[ExecutionResult] = Field(
        default_factory=list,
        description="All attempt results",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Aggregated error message",
    )

    def mark_started(self) -> None:
        """Mark the record as started."""
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def mark_completed(self, result: ExecutionResult) -> None:
        """Mark the record as completed with the given result."""
        self.status = ExecutionStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.result = result
        self.attempts.append(result)
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = delta.total_seconds() * 1000

    def mark_failed(self, result: ExecutionResult, error: str) -> None:
        """Mark the record as failed with the given result."""
        self.status = ExecutionStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.result = result
        self.attempts.append(result)
        self.error_message = error
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = delta.total_seconds() * 1000

    def mark_timed_out(self, error: str) -> None:
        """Mark the record as timed out."""
        self.status = ExecutionStatus.TIMED_OUT
        self.completed_at = datetime.now(timezone.utc)
        self.error_message = error
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = delta.total_seconds() * 1000

    def mark_retrying(self) -> None:
        """Mark the record as retrying."""
        self.status = ExecutionStatus.RETRYING

    def mark_skipped(self, reason: str = "") -> None:
        """Mark the record as skipped."""
        self.status = ExecutionStatus.SKIPPED
        self.completed_at = datetime.now(timezone.utc)
        self.error_message = reason or None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary for API responses."""
        return {
            "record_id": str(self.record_id),
            "step_index": self.step_index,
            "description": self.description,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "retry_count": len(self.attempts) - 1 if self.attempts else 0,
            "error": self.error_message,
            "success": self.result.success if self.result else False,
        }
