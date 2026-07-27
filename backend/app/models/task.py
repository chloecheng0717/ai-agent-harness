"""
Task Models

This module defines Pydantic models for task-related data structures.
Tasks represent user requests that will be processed by the agent system.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """
    Task status enumeration.
    
    Represents the lifecycle states of a task from creation to completion.
    """
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """
    Task priority levels.
    
    Used for task scheduling and resource allocation.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskCreate(BaseModel):
    """
    Task creation request model.
    
    This model is used when creating a new task. It contains
    all the information needed to initialize task processing.
    
    Attributes:
        type: Type of task (e.g., 'research', 'code_generation', 'analysis')
        description: Detailed description of the task
        priority: Task priority level
        metadata: Additional metadata for the task
        constraints: Any constraints or requirements
    """
    type: str = Field(..., description="Type of task to execute")
    description: str = Field(..., description="Detailed task description")
    priority: TaskPriority = Field(
        default=TaskPriority.MEDIUM,
        description="Task priority level"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional task metadata"
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Task constraints or requirements"
    )


class TaskResponse(BaseModel):
    """
    Task response model.
    
    This model represents a task in API responses. It includes
    the task data along with system-generated fields.
    
    Attributes:
        id: Unique task identifier
        type: Type of task
        description: Task description
        status: Current task status
        priority: Task priority level
        result: Task result (if completed)
        error: Error message (if failed)
        metadata: Additional metadata
        created_at: Task creation timestamp
        updated_at: Last update timestamp
    """
    id: UUID = Field(default_factory=uuid4, description="Unique task identifier")
    type: str = Field(..., description="Type of task")
    description: str = Field(..., description="Task description")
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current task status"
    )
    priority: TaskPriority = Field(
        default=TaskPriority.MEDIUM,
        description="Task priority level"
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Task result if completed"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional task metadata"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Task creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )


class TaskUpdate(BaseModel):
    """
    Task update request model.
    
    This model is used for updating task properties. All fields
    are optional to allow partial updates.
    
    Attributes:
        status: New task status
        result: Task result
        error: Error message
        metadata: Updated metadata
    """
    status: Optional[TaskStatus] = Field(
        default=None,
        description="New task status"
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Task result"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Updated metadata"
    )