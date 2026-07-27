"""
Models Package

This package contains all Pydantic models for request/response validation.
Models define the data structures used throughout the application.
"""

from app.models.task import TaskCreate, TaskResponse, TaskUpdate, TaskStatus, TaskPriority
from app.models.agent import AgentCreate, AgentResponse, AgentUpdate, AgentStatus, AgentType
from app.models.memory import MemoryEntry, MemorySearchResult
from app.models.tool import (
    ToolDefinition,
    ToolResult,
    ToolStatus,
    ToolExecutionRequest,
    RiskLevel,
    ToolPermission,
    ConfirmationRequirement,
    SandboxCapability,
)
from app.models.state import (
    AgentState,
    LifecycleStep,
    StepStatus,
    StepRecord,
    TokenUsage,
    Message,
    PlanStep,
)
from app.models.llm import (
    BaseLLMProvider,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    LLMProviderConfig,
    LLMProviderType,
    ModelInfo,
    TokenUsageInfo,
    LLMProviderError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMTimeoutError,
)
from app.models.providers import OpenAIProvider

__all__ = [
    # Task models
    "TaskCreate",
    "TaskResponse",
    "TaskUpdate",
    "TaskStatus",
    "TaskPriority",
    
    # Agent models
    "AgentCreate",
    "AgentResponse",
    "AgentUpdate",
    "AgentStatus",
    "AgentType",
    
    # Memory models
    "MemoryEntry",
    "MemorySearchResult",
    
    # Tool models
    "ToolDefinition",
    "ToolResult",
    "ToolStatus",
    "ToolExecutionRequest",
    "RiskLevel",
    "ToolPermission",
    "ConfirmationRequirement",
    "SandboxCapability",

    # State models
    "AgentState",
    "LifecycleStep",
    "StepStatus",
    "StepRecord",
    "TokenUsage",
    "Message",
    "PlanStep",

    # LLM models
    "BaseLLMProvider",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "LLMProviderConfig",
    "LLMProviderType",
    "ModelInfo",
    "TokenUsageInfo",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMAuthenticationError",
    "LLMTimeoutError",

    # LLM providers
    "OpenAIProvider",
]