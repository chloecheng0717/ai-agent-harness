"""
Base Tool Class

This module defines the abstract base class for all tools.
Tools are plugins that agents can use to interact with external systems.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import structlog

from app.models.tool import (
    ConfirmationRequirement,
    RiskLevel,
    SandboxCapability,
    ToolDefinition,
    ToolPermission,
    ToolResult,
)

if TYPE_CHECKING:
    from app.sandbox.workspace import Workspace


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    This class defines the interface that all tools must implement.
    Tools provide capabilities that agents can use to perform tasks.
    
    The tool interface supports:
    - Execution with parameters
    - Parameter validation
    - Health checking
    - Schema generation for LLM tool use
    - Permission checking
    - Risk assessment
    - Confirmation requirements
    - Sandbox capability
    
    Attributes:
        definition: Tool definition with metadata and schemas
        logger: Structured logger for tool operations
    """
    
    def __init__(self, definition: ToolDefinition) -> None:
        """
        Initialize the tool.
        
        Args:
            definition: Tool definition with metadata and schemas
        """
        self.definition = definition
        
        # Setup structured logging
        self.logger = structlog.get_logger().bind(
            tool_name=definition.name,
            tool_version=definition.version,
        )
        
        self.logger.info("Tool initialized")
    
    @property
    def name(self) -> str:
        """Get the tool name."""
        return self.definition.name
    
    @property
    def permissions(self) -> List[ToolPermission]:
        """Get the permissions required by this tool."""
        return self.definition.permissions
    
    @property
    def risk_level(self) -> RiskLevel:
        """Get the risk level of this tool."""
        return self.definition.risk_level
    
    @property
    def confirmation_required(self) -> ConfirmationRequirement:
        """Get when confirmation is required for this tool."""
        return self.definition.confirmation_required
    
    @property
    def sandbox_capability(self) -> SandboxCapability:
        """Get the sandbox capability of this tool."""
        return self.definition.sandbox_capability
    
    @abstractmethod
    async def execute(
        self,
        parameters: Dict[str, Any],
        workspace: Optional["Workspace"] = None,
    ) -> ToolResult:
        """
        Execute the tool with given parameters.

        This is the main method that tools must implement.
        It receives parameters, performs the operation, and returns results.

        Args:
            parameters: Tool parameters matching the definition schema
            workspace: Optional persistent shared workspace for the
                current agent execution run.  All tool calls within one
                run receive the same workspace so files persist between
                calls.

        Returns:
            ToolResult with success status, output, and metadata
        """
        pass
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate tool parameters against the definition schema.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        # Default implementation - subclasses can override for custom validation
        try:
            # In a real implementation, we would validate against the JSON Schema
            # For now, we just check that required fields are present
            required_fields = self.definition.parameters.get("required", [])
            for field in required_fields:
                if field not in parameters:
                    self.logger.warning("Missing required parameter", field=field)
                    return False
            return True
        except Exception as e:
            self.logger.error("Parameter validation failed", error=str(e))
            return False
    
    async def health_check(self) -> bool:
        """
        Check if the tool is healthy and available.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Default implementation - subclasses can override
            return True
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False
    
    def has_permission(self, permission: ToolPermission) -> bool:
        """
        Check if the tool requires a specific permission.
        
        Args:
            permission: Permission to check
            
        Returns:
            True if the tool requires this permission, False otherwise
        """
        return permission in self.definition.permissions
    
    def requires_any_permission(self, permissions: List[ToolPermission]) -> bool:
        """
        Check if the tool requires any of the given permissions.
        
        Args:
            permissions: List of permissions to check
            
        Returns:
            True if the tool requires at least one of these permissions, False otherwise
        """
        return any(p in self.definition.permissions for p in permissions)
    
    def requires_all_permissions(self, permissions: List[ToolPermission]) -> bool:
        """
        Check if the tool requires all of the given permissions.
        
        Args:
            permissions: List of permissions to check
            
        Returns:
            True if the tool requires all of these permissions, False otherwise
        """
        return all(p in self.definition.permissions for p in permissions)
    
    def can_execute_with_permissions(
        self, available_permissions: List[ToolPermission]
    ) -> bool:
        """
        Check if the tool can be executed with the available permissions.
        
        Args:
            available_permissions: Permissions available to the requester
            
        Returns:
            True if the tool can be executed, False otherwise
        """
        if not self.definition.permissions:
            return True
        return all(p in available_permissions for p in self.definition.permissions)
    
    def needs_confirmation(self) -> bool:
        """
        Check if the tool needs user confirmation before execution.
        
        Returns:
            True if confirmation is needed, False otherwise
        """
        return self.definition.confirmation_required != ConfirmationRequirement.NEVER
    
    def needs_confirmation_for_risk(self, risk_level: RiskLevel) -> bool:
        """
        Check if confirmation is needed for a specific risk level.
        
        Args:
            risk_level: Risk level to check
            
        Returns:
            True if confirmation is needed, False otherwise
        """
        if self.definition.confirmation_required == ConfirmationRequirement.ALWAYS:
            return True
        if self.definition.confirmation_required == ConfirmationRequirement.NEVER:
            return False
        if self.definition.confirmation_required == ConfirmationRequirement.HIGH_RISK:
            return risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        return False
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the tool schema for LLM tool use.
        
        This method returns a schema that can be used by LLMs
        to understand how to use this tool.
        
        Returns:
            Dict containing tool schema
        """
        return {
            "name": self.definition.name,
            "description": self.definition.description,
            "parameters": self.definition.parameters,
            "returns": self.definition.returns,
            "permissions": [p.value for p in self.definition.permissions],
            "risk_level": self.definition.risk_level.value,
            "confirmation_required": self.definition.confirmation_required.value,
            "sandbox_capability": self.definition.sandbox_capability.value,
        }
    
    def __repr__(self) -> str:
        """String representation of the tool."""
        return f"<{self.__class__.__name__}(name={self.definition.name})>"