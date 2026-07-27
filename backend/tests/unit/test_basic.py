"""
Basic Unit Tests

This module contains basic unit tests to verify the project structure.
"""

import pytest
from uuid import uuid4

from app.config import get_settings
from app.models.task import TaskCreate, TaskResponse, TaskStatus, TaskPriority
from app.models.agent import AgentCreate, AgentResponse, AgentStatus, AgentType
from app.agents.base import BaseAgent, AgentConfig


def test_settings_load():
    """Test that settings can be loaded."""
    settings = get_settings()
    assert settings.app_name == "AI Agent Harness"
    assert settings.app_version == "0.1.0"


def test_task_creation():
    """Test task model creation."""
    task = TaskCreate(
        type="test",
        description="Test task",
        priority=TaskPriority.MEDIUM,
    )
    
    assert task.type == "test"
    assert task.description == "Test task"
    assert task.priority == TaskPriority.MEDIUM


def test_task_response():
    """Test task response model."""
    response = TaskResponse(
        type="test",
        description="Test task",
    )
    
    assert response.id is not None
    assert response.status == TaskStatus.PENDING
    assert response.created_at is not None


def test_agent_creation():
    """Test agent model creation."""
    agent = AgentCreate(
        name="Test Agent",
        type=AgentType.EXECUTOR,
        description="A test agent",
    )
    
    assert agent.name == "Test Agent"
    assert agent.type == AgentType.EXECUTOR


def test_agent_response():
    """Test agent response model."""
    response = AgentResponse(
        name="Test Agent",
        type=AgentType.EXECUTOR,
    )
    
    assert response.id is not None
    assert response.status == AgentStatus.IDLE
    assert response.created_at is not None


class StubAgent(BaseAgent):
    """Concrete BaseAgent stub for testing."""
    
    async def process(self, input_data):
        """Process input data."""
        return {"result": "processed"}
    
    async def reflect(self, result):
        """Reflect on result."""
        return {"insight": "test insight"}


def test_base_agent():
    """Test base agent functionality."""
    config = AgentConfig(
        name="Test Agent",
        type=AgentType.EXECUTOR,
    )
    
    agent = StubAgent(config)
    
    assert agent.id is not None
    assert agent.config.name == "Test Agent"
    assert agent.status == AgentStatus.IDLE
    
    # Test status
    status = agent.get_status()
    assert status["name"] == "Test Agent"
    assert status["type"] == "executor"
    
    # Test memory
    agent.update_memory("key", "value")
    assert agent.get_memory("key") == "value"
    assert agent.get_memory("missing") is None