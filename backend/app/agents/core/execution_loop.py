"""
Execution Loop Module

This module manages tool execution and result aggregation.
It coordinates between agents and the tool registry.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog

from app.models.tool import ToolExecutionRequest, ToolResult
from app.tools.registry import ToolRegistry


class ExecutionLoop:
    """
    Execution loop for tool-based task completion.
    
    This class manages the execution of tools by agents.
    It handles tool selection, parameter preparation, execution,
    and result aggregation.
    
    The execution loop:
    1. Receives an execution request
    2. Selects appropriate tools
    3. Prepares parameters
    4. Executes tools
    5. Aggregates results
    
    Attributes:
        tool_registry: Registry of available tools
        logger: Structured logger
    """
    
    def __init__(self, tool_registry: ToolRegistry) -> None:
        """
        Initialize the execution loop.
        
        Args:
            tool_registry: Registry of available tools
        """
        self.tool_registry = tool_registry
        self.logger = structlog.get_logger().bind(component="execution_loop")
        self.logger.info("Execution loop initialized")
    
    async def execute(
        self,
        tool_requests: List[ToolExecutionRequest],
        agent_id: Optional[UUID] = None,
    ) -> List[ToolResult]:
        """
        Execute multiple tool requests.
        
        This method executes a list of tool requests and returns results.
        Tools are executed sequentially by default.
        
        Args:
            tool_requests: List of tool execution requests
            agent_id: Optional agent ID for logging
            
        Returns:
            List of tool execution results
        """
        self.logger.info(
            "Executing tool requests",
            request_count=len(tool_requests),
            agent_id=str(agent_id) if agent_id else None,
        )
        
        results = []
        
        for request in tool_requests:
            # Set agent ID if provided
            if agent_id:
                request.agent_id = agent_id
            
            # Execute tool
            result = await self.tool_registry.execute(request)
            results.append(result)
            
            # Log result
            if result.success:
                self.logger.debug(
                    "Tool executed successfully",
                    tool_name=request.tool_name,
                    execution_time=result.execution_time,
                )
            else:
                self.logger.warning(
                    "Tool execution failed",
                    tool_name=request.tool_name,
                    error=result.error,
                )
        
        return results
    
    async def execute_with_retry(
        self,
        request: ToolExecutionRequest,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> ToolResult:
        """
        Execute a tool with retry logic.
        
        This method retries failed tool executions up to max_retries times.
        
        Args:
            request: Tool execution request
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            
        Returns:
            Tool execution result
        """
        import asyncio
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            result = await self.tool_registry.execute(request)
            
            if result.success:
                return result
            
            last_error = result.error
            self.logger.warning(
                "Tool execution failed, retrying",
                tool_name=request.tool_name,
                attempt=attempt + 1,
                max_retries=max_retries,
                error=last_error,
            )
            
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (attempt + 1))
        
        # All retries exhausted
        return ToolResult(
            success=False,
            error=f"Tool execution failed after {max_retries} retries: {last_error}",
        )
    
    async def execute_parallel(
        self,
        tool_requests: List[ToolExecutionRequest],
        agent_id: Optional[UUID] = None,
    ) -> List[ToolResult]:
        """
        Execute multiple tool requests in parallel.
        
        This method executes tools concurrently for better performance.
        
        Args:
            tool_requests: List of tool execution requests
            agent_id: Optional agent ID for logging
            
        Returns:
            List of tool execution results
        """
        import asyncio
        
        self.logger.info(
            "Executing tools in parallel",
            request_count=len(tool_requests),
            agent_id=str(agent_id) if agent_id else None,
        )
        
        # Create tasks for parallel execution
        tasks = []
        for request in tool_requests:
            if agent_id:
                request.agent_id = agent_id
            tasks.append(self.tool_registry.execute(request))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    ToolResult(
                        success=False,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)
        
        return final_results
    
    def get_available_tools(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get list of available tools.
        
        Args:
            tags: Optional filter by tags
            
        Returns:
            List of tool schemas
        """
        return self.tool_registry.list_tools(tags=tags)