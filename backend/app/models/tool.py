"""
Tool Models

This module defines Pydantic models for tool-related data structures.
Tools are plugins that agents can use to interact with external systems.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    """
    Tool status enumeration.
    
    Represents the current state of a tool.
    """
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class RiskLevel(str, Enum):
    """
    Risk level enumeration for tool execution.
    
    Represents the potential risk associated with executing a tool.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolPermission(str, Enum):
    """
    Tool permission enumeration.
    
    Represents the permissions required to execute a tool.
    """
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    FILE_SYSTEM = "file_system"
    DATABASE = "database"
    ADMIN = "admin"


class ConfirmationRequirement(str, Enum):
    """
    Confirmation requirement enumeration.
    
    Defines when user confirmation is needed before tool execution.
    """
    NEVER = "never"
    ALWAYS = "always"
    HIGH_RISK = "high_risk"
    FIRST_USE = "first_use"


class SandboxCapability(str, Enum):
    """
    Sandbox capability enumeration.
    
    Defines the sandboxing requirements for tool execution.
    """
    NONE = "none"
    PROCESS = "process"
    CONTAINER = "container"
    VM = "vm"


class ToolDefinition(BaseModel):
    """
    Tool definition model.
    
    This model defines the interface and capabilities of a tool.
    It's used for tool registration and discovery.
    
    Attributes:
        name: Unique tool name
        description: Tool description
        version: Tool version
        parameters: JSON Schema for tool parameters
        returns: JSON Schema for tool return value
        tags: Tool tags for categorization
        requires_auth: Whether the tool requires authentication
        timeout: Tool execution timeout in seconds
        permissions: List of permissions required by the tool
        risk_level: Risk level of the tool
        confirmation_required: When confirmation is needed
        sandbox_capability: Sandboxing requirements
    """
    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="Tool description")
    version: str = Field(default="1.0.0", description="Tool version")
    parameters: Dict[str, Any] = Field(
        ...,
        description="JSON Schema for tool parameters"
    )
    returns: Dict[str, Any] = Field(
        ...,
        description="JSON Schema for tool return value"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tool tags for categorization"
    )
    requires_auth: bool = Field(
        default=False,
        description="Whether tool requires authentication"
    )
    timeout: int = Field(
        default=30,
        description="Execution timeout in seconds"
    )
    permissions: List[ToolPermission] = Field(
        default_factory=list,
        description="Permissions required by the tool"
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.LOW,
        description="Risk level of the tool"
    )
    confirmation_required: ConfirmationRequirement = Field(
        default=ConfirmationRequirement.NEVER,
        description="When confirmation is needed"
    )
    sandbox_capability: SandboxCapability = Field(
        default=SandboxCapability.NONE,
        description="Sandboxing requirements",
    )

    def parameter_summary(self) -> str:
        """
        Render the parameter schema as a compact, LLM-readable spec.

        Each property is listed with its type/enum, a ``(required)``
        marker when it is in the top-level ``required`` list, and its
        description.  ``oneOf``/``anyOf`` "exactly one of" constraints
        (used by tools that accept either ``path`` or ``content``) are
        rendered as a single line so the LLM knows an argument is
        mandatory.
        """
        lines: List[str] = []
        params = self.parameters or {}
        props = params.get("properties", {})
        required = set(params.get("required", []))

        for pname, pmeta in props.items():
            ptype = pmeta.get("type", "any")
            if isinstance(ptype, list):
                ptype = " | ".join(str(t) for t in ptype)
            enums = pmeta.get("enum")
            suffix = " | ".join(str(e) for e in enums) if enums else ptype
            flag = "(required)" if pname in required else ""
            note = pmeta.get("description", "")
            line = f"      {pname} ({suffix}) {flag}".rstrip()
            if note:
                line += f" -- {note}"
            lines.append(line)

        for alternative in params.get("oneOf") or params.get("anyOf") or []:
            if isinstance(alternative, dict) and alternative.get("required"):
                names = ", ".join(str(n) for n in alternative["required"])
                lines.append(f"      one of: {names} -- exactly one required")

        return "\n".join(lines)


class ToolResult(BaseModel):
    """
    Tool execution result model.
    
    This model represents the result of a tool execution,
    including success/failure status and output data.
    
    Attributes:
        success: Whether execution was successful
        output: Tool output data
        error: Error message if execution failed
        execution_time: Time taken to execute in milliseconds
        metadata: Additional result metadata
    """
    success: bool = Field(..., description="Whether execution was successful")
    output: Any = Field(default=None, description="Tool output data")
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    execution_time: float = Field(
        default=0,
        description="Execution time in milliseconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional result metadata"
    )


class ToolExecutionRequest(BaseModel):
    """
    Tool execution request model.
    
    This model represents a request to execute a tool.
    
    Attributes:
        tool_name: Name of the tool to execute
        parameters: Tool parameters
        timeout: Optional timeout override
        agent_id: ID of the agent requesting execution
        task_id: ID of the task associated with execution
        agent_permissions: Permissions granted to the requesting agent
        confirmation_token: Token from user confirmation if required
        sandbox_enabled: Whether to enable sandboxing for this execution
    """
    tool_name: str = Field(..., description="Name of the tool to execute")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool parameters"
    )
    timeout: Optional[int] = Field(
        default=None,
        description="Timeout override in seconds"
    )
    agent_id: Optional[UUID] = Field(
        default=None,
        description="ID of the agent requesting execution"
    )
    task_id: Optional[UUID] = Field(
        default=None,
        description="ID of the task associated with execution"
    )
    agent_permissions: List[ToolPermission] = Field(
        default_factory=list,
        description="Permissions granted to the requesting agent"
    )
    workspace_id: Optional[str] = Field(
        default=None,
        description="ID of the persistent shared workspace for this execution run"
    )
    confirmation_token: Optional[str] = Field(
        default=None,
        description="Token from user confirmation if required"
    )
    sandbox_enabled: bool = Field(
        default=False,
        description="Whether to enable sandboxing for this execution"
    )