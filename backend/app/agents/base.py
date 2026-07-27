"""
Base Agent Class

This module defines the abstract base class for all agents in the system.
It provides the common interface and functionality that all agents must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from app.models.agent import AgentStatus, AgentType


class AgentConfig(BaseModel):
    """
    Agent configuration model.
    
    This model defines the configuration options for an agent.
    It can be extended for agent-specific configurations.
    
    Attributes:
        name: Human-readable agent name
        type: Type of agent
        description: Agent description
        model: LLM model to use
        temperature: LLM temperature
        max_tokens: Maximum tokens for LLM responses
        tools: List of tool names this agent can use
    """
    name: str = Field(..., description="Human-readable agent name")
    type: AgentType = Field(..., description="Type of agent")
    description: str = Field(default="", description="Agent description")
    model: str = Field(default="gpt-4", description="LLM model to use")
    temperature: float = Field(default=0.7, description="LLM temperature")
    max_tokens: int = Field(default=4096, description="Maximum tokens")
    tools: List[str] = Field(default_factory=list, description="Available tools")


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    This class defines the interface that all agents must implement.
    It provides common functionality for agent lifecycle management,
    task processing, and communication.
    
    The agent follows a simple lifecycle:
    1. Initialization: Agent is created with configuration
    2. Processing: Agent receives and processes tasks
    3. Reflection: Agent evaluates its performance
    4. Cleanup: Agent is shut down and resources are released
    
    Attributes:
        id: Unique agent identifier
        config: Agent configuration
        status: Current agent status
        memory: Agent's memory store
    """
    
    def __init__(self, config: AgentConfig) -> None:
        """
        Initialize the agent.
        
        Args:
            config: Agent configuration
        """
        self.id: UUID = uuid4()
        self.config = config
        self.status = AgentStatus.IDLE
        self.memory: Dict[str, Any] = {}
        
        # Setup structured logging
        self.logger = structlog.get_logger().bind(
            agent_id=str(self.id),
            agent_type=config.type.value,
            agent_name=config.name,
        )
        
        self.logger.info(
            "Agent initialized",
            model=config.model,
            tools=config.tools,
        )
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        This is the main method that agents must implement.
        It receives input data, processes it, and returns results.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Dict containing processing results
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    @abstractmethod
    async def reflect(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on the processing result.
        
        This method allows agents to evaluate their performance
        and identify areas for improvement.
        
        Args:
            result: The processing result to reflect on
            
        Returns:
            Dict containing reflection insights
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    async def start(self) -> None:
        """
        Start the agent.
        
        This method is called when the agent begins processing.
        It should initialize any necessary resources.
        """
        self.status = AgentStatus.IDLE
        self.logger.info("Agent started")
    
    async def stop(self) -> None:
        """
        Stop the agent.
        
        This method is called when the agent is being shut down.
        It should clean up any resources.
        """
        self.status = AgentStatus.DISABLED
        self.logger.info("Agent stopped")
    
    async def handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle errors that occur during processing.
        
        This method provides a consistent way to handle errors
        across all agents.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Dict containing error information
        """
        self.status = AgentStatus.ERROR
        self.logger.error(
            "Agent error",
            error=str(error),
            error_type=type(error).__name__,
        )
        
        return {
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "agent_id": str(self.id),
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status.
        
        Returns:
            Dict containing agent status information
        """
        return {
            "id": str(self.id),
            "name": self.config.name,
            "type": self.config.type.value,
            "status": self.status.value,
            "model": self.config.model,
            "tools": self.config.tools,
        }
    
    def update_memory(self, key: str, value: Any) -> None:
        """
        Update agent memory.
        
        Args:
            key: Memory key
            value: Memory value
        """
        self.memory[key] = value
        self.logger.debug("Memory updated", key=key)
    
    def get_memory(self, key: str) -> Optional[Any]:
        """
        Get value from agent memory.
        
        Args:
            key: Memory key
            
        Returns:
            Memory value or None if not found
        """
        return self.memory.get(key)