"""
Agent Models

This module defines Pydantic models for agent-related data structures.
Agents are the core processing units that handle tasks using tools and memory.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """
    Agent status enumeration.
    
    Represents the current state of an agent instance.
    """
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    DISABLED = "disabled"


class AgentType(str, Enum):
    """
    Agent type enumeration.
    
    Defines the different types of specialized agents available.
    """
    PLANNER = "planner"
    EXECUTOR = "executor"
    RESEARCHER = "researcher"
    SECURITY = "security"
    REVIEWER = "reviewer"
    ORCHESTRATOR = "orchestrator"


class AgentCreate(BaseModel):
    """
    Agent creation request model.
    
    This model is used when creating a new agent instance. It contains
    all the configuration needed to initialize the agent.
    
    Attributes:
        name: Human-readable agent name
        type: Type of agent to create
        description: Agent description
        configuration: Agent-specific configuration
        tools: List of tools the agent can use
        model: LLM model to use for this agent
    """
    name: str = Field(..., description="Human-readable agent name")
    type: AgentType = Field(..., description="Type of agent to create")
    description: str = Field(default="", description="Agent description")
    configuration: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific configuration"
    )
    tools: List[str] = Field(
        default_factory=list,
        description="List of tool names the agent can use"
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM model to use (overrides default)"
    )


class AgentResponse(BaseModel):
    """
    Agent response model.
    
    This model represents an agent in API responses. It includes
    the agent data along with system-generated fields.
    
    Attributes:
        id: Unique agent identifier
        name: Human-readable agent name
        type: Type of agent
        status: Current agent status
        description: Agent description
        configuration: Agent configuration
        tools: List of available tools
        model: LLM model being used
        created_at: Agent creation timestamp
        updated_at: Last update timestamp
        last_active: Last time agent was active
    """
    id: UUID = Field(default_factory=uuid4, description="Unique agent identifier")
    name: str = Field(..., description="Human-readable agent name")
    type: AgentType = Field(..., description="Type of agent")
    status: AgentStatus = Field(
        default=AgentStatus.IDLE,
        description="Current agent status"
    )
    description: str = Field(default="", description="Agent description")
    configuration: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent configuration"
    )
    tools: List[str] = Field(
        default_factory=list,
        description="Available tools"
    )
    model: str = Field(default="gpt-4", description="LLM model being used")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Agent creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )
    last_active: Optional[datetime] = Field(
        default=None,
        description="Last time agent was active"
    )


class AgentUpdate(BaseModel):
    """
    Agent update request model.
    
    This model is used for updating agent properties. All fields
    are optional to allow partial updates.
    
    Attributes:
        name: New agent name
        status: New agent status
        description: Updated description
        configuration: Updated configuration
        tools: Updated tools list
        model: Updated LLM model
    """
    name: Optional[str] = Field(
        default=None,
        description="New agent name"
    )
    status: Optional[AgentStatus] = Field(
        default=None,
        description="New agent status"
    )
    description: Optional[str] = Field(
        default=None,
        description="Updated description"
    )
    configuration: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Updated configuration"
    )
    tools: Optional[List[str]] = Field(
        default=None,
        description="Updated tools list"
    )
    model: Optional[str] = Field(
        default=None,
        description="Updated LLM model"
    )