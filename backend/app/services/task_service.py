"""
Task Service

This module provides business logic for task management.
It handles task creation, retrieval, and status updates.
"""

from typing import Dict, List, Optional
from uuid import UUID, uuid4

import structlog

from app.models.task import TaskCreate, TaskResponse, TaskStatus, TaskUpdate


class TaskService:
    """
    Service for task management.
    
    This class provides business logic for task operations.
    In a real implementation, this would interact with a database.
    
    Attributes:
        _tasks: In-memory task storage
        logger: Structured logger
    """
    
    def __init__(self) -> None:
        """Initialize the task service."""
        self._tasks: Dict[UUID, TaskResponse] = {}
        self.logger = structlog.get_logger().bind(component="task_service")
        self.logger.info("Task service initialized")
    
    async def create_task(self, task_data: TaskCreate) -> TaskResponse:
        """
        Create a new task.
        
        Args:
            task_data: Task creation data
            
        Returns:
            Created task response
        """
        task_id = uuid4()
        
        task = TaskResponse(
            id=task_id,
            type=task_data.type,
            description=task_data.description,
            priority=task_data.priority,
            metadata=task_data.metadata,
            status=TaskStatus.PENDING,
        )
        
        self._tasks[task_id] = task
        
        self.logger.info(
            "Task created",
            task_id=str(task_id),
            task_type=task.type,
            priority=task.priority,
        )
        
        return task
    
    async def get_task(self, task_id: UUID) -> Optional[TaskResponse]:
        """
        Get a task by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task response if found, None otherwise
        """
        return self._tasks.get(task_id)
    
    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[TaskResponse]:
        """
        List tasks with optional filtering.
        
        Args:
            status: Filter by status
            limit: Maximum number of tasks
            offset: Number of tasks to skip
            
        Returns:
            List of tasks
        """
        tasks = list(self._tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        # Apply pagination
        tasks = tasks[offset:offset + limit]
        
        return tasks
    
    async def update_task(
        self,
        task_id: UUID,
        update_data: TaskUpdate,
    ) -> Optional[TaskResponse]:
        """
        Update a task.
        
        Args:
            task_id: Task identifier
            update_data: Update data
            
        Returns:
            Updated task response if found, None otherwise
        """
        task = self._tasks.get(task_id)
        if not task:
            return None
        
        # Update fields
        if update_data.status is not None:
            task.status = update_data.status
        if update_data.result is not None:
            task.result = update_data.result
        if update_data.error is not None:
            task.error = update_data.error
        if update_data.metadata is not None:
            task.metadata.update(update_data.metadata)
        
        self.logger.info(
            "Task updated",
            task_id=str(task_id),
            status=task.status,
        )
        
        return task
    
    async def delete_task(self, task_id: UUID) -> bool:
        """
        Delete a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if deleted, False if not found
        """
        if task_id in self._tasks:
            del self._tasks[task_id]
            self.logger.info("Task deleted", task_id=str(task_id))
            return True
        return False