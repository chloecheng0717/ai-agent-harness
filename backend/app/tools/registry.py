"""
Tool Registry

This module provides a centralized registry for tool management.
Tools can be registered, discovered, and executed through the registry.
"""

import asyncio
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Type

import structlog

from app.models.tool import (
    RiskLevel,
    ToolDefinition,
    ToolExecutionRequest,
    ToolPermission,
    ToolResult,
)
from app.tools.base import BaseTool


# Type alias for confirmation callbacks
ConfirmationCallback = Callable[[str, Dict[str, Any], RiskLevel], Coroutine[Any, Any, bool]]


class ToolRegistry:
    """
    Centralized tool registry.
    
    This class manages tool registration, discovery, and execution.
    It provides a single interface for agents to interact with tools.
    
    The registry supports:
    - Dynamic tool registration
    - Tool discovery by name or tags
    - Tool execution with validation
    - Health monitoring
    - Permission checking
    - Confirmation flow
    - Sandbox support
    - Execution history and metrics
    
    Attributes:
        _tools: Dictionary of registered tools
        _tool_classes: Dictionary of tool classes for lazy instantiation
        _execution_history: List of recent execution results
        _confirmation_callback: Callback for user confirmation
        _max_history_size: Maximum number of execution records to keep
    """
    
    def __init__(self, max_history_size: int = 100) -> None:
        """Initialize the tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        self._tool_definitions: Dict[str, ToolDefinition] = {}
        self._execution_history: List[Dict[str, Any]] = []
        self._max_history_size = max_history_size
        self._confirmation_callback: Optional[ConfirmationCallback] = None
        
        self.logger = structlog.get_logger().bind(component="tool_registry")
        self.logger.info("Tool registry initialized")
    
    def set_confirmation_callback(self, callback: ConfirmationCallback) -> None:
        """
        Set the confirmation callback for user confirmation.
        
        Args:
            callback: Async function that takes tool_name, parameters, and risk_level
                     and returns True if confirmed, False otherwise
        """
        self._confirmation_callback = callback
        self.logger.info("Confirmation callback set")
    
    def register(self, tool_class: Type[BaseTool], definition: ToolDefinition) -> None:
        """
        Register a tool class.
        
        This method registers a tool class for lazy instantiation.
        The tool will be instantiated when first accessed.
        
        Args:
            tool_class: Tool class to register
            definition: Tool definition with metadata
            
        Raises:
            ValueError: If tool name already registered
        """
        if definition.name in self._tool_classes:
            raise ValueError(f"Tool '{definition.name}' is already registered")
        
        self._tool_classes[definition.name] = tool_class
        self._tool_definitions[definition.name] = definition
        self.logger.info(
            "Tool class registered",
            tool_name=definition.name,
            tool_version=definition.version,
            permissions=[p.value for p in definition.permissions],
            risk_level=definition.risk_level.value,
        )
    
    def register_instance(self, tool: BaseTool) -> None:
        """
        Register a tool instance.
        
        This method registers a pre-instantiated tool instance.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If tool name already registered
        """
        if tool.definition.name in self._tools:
            raise ValueError(f"Tool '{tool.definition.name}' is already registered")
        
        self._tools[tool.definition.name] = tool
        self._tool_definitions[tool.definition.name] = tool.definition
        self.logger.info(
            "Tool instance registered",
            tool_name=tool.definition.name,
            permissions=[p.value for p in tool.definition.permissions],
            risk_level=tool.definition.risk_level.value,
        )
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance if registered, None otherwise
        """
        # Check if we have an instance
        if name in self._tools:
            return self._tools[name]
        
        # Check if we have a class for lazy instantiation
        if name in self._tool_classes and name in self._tool_definitions:
            tool_class = self._tool_classes[name]
            try:
                tool_instance = tool_class()
                self._tools[name] = tool_instance
                return tool_instance
            except Exception as e:
                self.logger.error(
                    "Failed to instantiate tool",
                    tool_name=name,
                    error=str(e),
                )
                return None
        
        return None
    
    def get_tool_definition(self, name: str) -> Optional[ToolDefinition]:
        """
        Get a tool definition by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool definition if registered, None otherwise
        """
        return self._tool_definitions.get(name)
    
    def check_permissions(
        self, tool_name: str, agent_permissions: List[ToolPermission]
    ) -> bool:
        """
        Check if the agent has the required permissions to execute a tool.
        
        Args:
            tool_name: Name of the tool
            agent_permissions: Permissions available to the agent
            
        Returns:
            True if agent has required permissions, False otherwise
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return False
        
        return tool.can_execute_with_permissions(agent_permissions)
    
    async def check_confirmation(
        self, tool_name: str, parameters: Dict[str, Any], risk_level: RiskLevel
    ) -> bool:
        """
        Check if user confirmation is required and obtain it.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            risk_level: Risk level of the tool
            
        Returns:
            True if confirmed or not required, False if denied
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return False
        
        if not tool.needs_confirmation():
            return True
        
        if not tool.needs_confirmation_for_risk(risk_level):
            return True
        
        if self._confirmation_callback is None:
            self.logger.warning(
                "Confirmation required but no callback set",
                tool_name=tool_name,
            )
            return False
        
        try:
            confirmed = await self._confirmation_callback(tool_name, parameters, risk_level)
            if not confirmed:
                self.logger.info(
                    "Tool execution denied by user",
                    tool_name=tool_name,
                )
            return confirmed
        except Exception as e:
            self.logger.error(
                "Confirmation callback failed",
                tool_name=tool_name,
                error=str(e),
            )
            return False
    
    async def execute(self, request: ToolExecutionRequest) -> ToolResult:
        """
        Execute a tool by name.
        
        This method finds the tool, validates parameters, checks permissions,
        handles confirmation, and executes it.
        
        Args:
            request: Tool execution request
            
        Returns:
            ToolResult with execution results
        """
        tool = self.get_tool(request.tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{request.tool_name}' not found",
            )
        
        # Check permissions
        if not self.check_permissions(request.tool_name, request.agent_permissions):
            self.logger.warning(
                "Permission denied",
                tool_name=request.tool_name,
                required_permissions=[p.value for p in tool.permissions],
                available_permissions=[p.value for p in request.agent_permissions],
            )
            return ToolResult(
                success=False,
                error=f"Insufficient permissions for tool '{request.tool_name}'",
            )
        
        # Check confirmation
        if not await self.check_confirmation(
            request.tool_name, request.parameters, tool.risk_level
        ):
            return ToolResult(
                success=False,
                error=f"Confirmation denied for tool '{request.tool_name}'",
            )
        
        # Validate parameters
        if not await tool.validate_parameters(request.parameters):
            return ToolResult(
                success=False,
                error="Invalid parameters",
            )
        
        # Execute tool with timeout
        timeout = request.timeout or tool.definition.timeout
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                tool.execute(request.parameters),
                timeout=timeout,
            )
            execution_time = (time.time() - start_time) * 1000
            result.execution_time = execution_time
            
            # Record execution history
            self._record_execution(
                request.tool_name,
                request.parameters,
                result,
                request.agent_id,
                request.task_id,
            )
            
            self.logger.info(
                "Tool executed successfully",
                tool_name=request.tool_name,
                execution_time=execution_time,
            )
            return result
            
        except asyncio.TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            error_result = ToolResult(
                success=False,
                error=f"Tool execution timed out after {timeout}s",
                execution_time=execution_time,
            )
            self._record_execution(
                request.tool_name,
                request.parameters,
                error_result,
                request.agent_id,
                request.task_id,
            )
            self.logger.error(
                "Tool execution timed out",
                tool_name=request.tool_name,
                timeout=timeout,
            )
            return error_result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_result = ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
            )
            self._record_execution(
                request.tool_name,
                request.parameters,
                error_result,
                request.agent_id,
                request.task_id,
            )
            self.logger.error(
                "Tool execution failed",
                tool_name=request.tool_name,
                error=str(e),
            )
            return error_result
    
    def _record_execution(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: ToolResult,
        agent_id: Optional[Any] = None,
        task_id: Optional[Any] = None,
    ) -> None:
        """
        Record an execution in the history.
        
        Args:
            tool_name: Name of the executed tool
            parameters: Tool parameters
            result: Execution result
            agent_id: Agent ID
            task_id: Task ID
        """
        record = {
            "tool_name": tool_name,
            "parameters": parameters,
            "success": result.success,
            "execution_time": result.execution_time,
            "agent_id": str(agent_id) if agent_id else None,
            "task_id": str(task_id) if task_id else None,
            "timestamp": time.time(),
        }
        
        self._execution_history.append(record)
        
        # Trim history if needed
        if len(self._execution_history) > self._max_history_size:
            self._execution_history = self._execution_history[-self._max_history_size:]
    
    def get_execution_history(
        self,
        tool_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get execution history.
        
        Args:
            tool_name: Optional filter by tool name
            limit: Maximum number of records to return
            
        Returns:
            List of execution records
        """
        history = self._execution_history
        if tool_name:
            history = [r for r in history if r["tool_name"] == tool_name]
        return history[-limit:]
    
    def get_execution_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Args:
            tool_name: Optional filter by tool name
            
        Returns:
            Dictionary with execution statistics
        """
        history = self._execution_history
        if tool_name:
            history = [r for r in history if r["tool_name"] == tool_name]
        
        if not history:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_execution_time": 0,
            }
        
        successful = sum(1 for r in history if r["success"])
        total_time = sum(r["execution_time"] for r in history)
        
        return {
            "total_executions": len(history),
            "successful_executions": successful,
            "failed_executions": len(history) - successful,
            "average_execution_time": total_time / len(history),
        }
    
    def list_tools(
        self,
        tags: Optional[List[str]] = None,
        permissions: Optional[List[ToolPermission]] = None,
        risk_level: Optional[RiskLevel] = None,
    ) -> List[Dict[str, Any]]:
        """
        List registered tools.
        
        Args:
            tags: Optional filter by tags
            permissions: Optional filter by required permissions
            risk_level: Optional filter by risk level
            
        Returns:
            List of tool schemas
        """
        tools = []
        
        # Add instantiated tools
        for tool in self._tools.values():
            if tags and not any(tag in tool.definition.tags for tag in tags):
                continue
            if permissions and not tool.requires_any_permission(permissions):
                continue
            if risk_level and tool.risk_level != risk_level:
                continue
            tools.append(tool.get_schema())
        
        # Add tool classes (not yet instantiated)
        for name, tool_class in self._tool_classes.items():
            if name in [t["name"] for t in tools]:
                continue  # Already added as instance
            
            definition = self._tool_definitions.get(name)
            if not definition:
                tools.append({
                    "name": name,
                    "description": "Tool class available for instantiation",
                })
                continue
            
            if tags and not any(tag in definition.tags for tag in tags):
                continue
            if permissions and not any(p in definition.permissions for p in permissions):
                continue
            if risk_level and definition.risk_level != risk_level:
                continue
            
            tools.append({
                "name": name,
                "description": definition.description,
                "permissions": [p.value for p in definition.permissions],
                "risk_level": definition.risk_level.value,
            })
        
        return tools
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all registered tools.
        
        Returns:
            Dict mapping tool names to health status
        """
        health = {}
        
        for name, tool in self._tools.items():
            health[name] = await tool.health_check()
        
        return health
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Tool name to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            if name in self._tool_definitions:
                del self._tool_definitions[name]
            self.logger.info("Tool unregistered", tool_name=name)
            return True
        
        if name in self._tool_classes:
            del self._tool_classes[name]
            if name in self._tool_definitions:
                del self._tool_definitions[name]
            self.logger.info("Tool class unregistered", tool_name=name)
            return True
        
        return False
    
    def clear(self) -> None:
        """Clear all registered tools and history."""
        self._tools.clear()
        self._tool_classes.clear()
        self._tool_definitions.clear()
        self._execution_history.clear()
        self.logger.info("Tool registry cleared")