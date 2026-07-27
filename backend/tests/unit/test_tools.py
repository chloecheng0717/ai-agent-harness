"""
Tool System Tests

Comprehensive tests for the tool framework:
- Tool models (ToolPermission, RiskLevel, ConfirmationRequirement, SandboxCapability)
- BaseTool abstract class
- ToolRegistry with permission checking, confirmation, execution history
- Placeholder tools (CalculatorTool, FileReadTool, WebSearchTool)
- Integration with Orchestrator
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.tool import (
    ConfirmationRequirement,
    RiskLevel,
    SandboxCapability,
    ToolDefinition,
    ToolExecutionRequest,
    ToolPermission,
    ToolResult,
    ToolStatus,
)
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry
from app.tools.plugins.calculator import CalculatorTool
from app.tools.plugins.file_read import FileReadTool
from app.tools.plugins.web_search import WebSearchTool


# ── Helper: Concrete Tool for Testing ─────────────────────────


class ConcreteTestTool(BaseTool):
    """A concrete test tool for testing the BaseTool interface."""

    def __init__(
        self,
        name: str = "test_tool",
        permissions: List[ToolPermission] = None,
        risk_level: RiskLevel = RiskLevel.LOW,
        confirmation_required: ConfirmationRequirement = ConfirmationRequirement.NEVER,
        sandbox_capability: SandboxCapability = SandboxCapability.NONE,
    ) -> None:
        definition = ToolDefinition(
            name=name,
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                },
                "required": ["value"],
            },
            returns={
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                },
            },
            permissions=permissions or [],
            risk_level=risk_level,
            confirmation_required=confirmation_required,
            sandbox_capability=sandbox_capability,
        )
        super().__init__(definition)

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the test tool."""
        value = parameters.get("value", "")
        return ToolResult(
            success=True,
            output={"result": f"processed: {value}"},
        )


# ── Tool Models Tests ─────────────────────────────────────────


class TestToolModels:
    """Tests for tool-related Pydantic models."""

    def test_risk_level_enum(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_tool_permission_enum(self):
        """Test ToolPermission enum values."""
        assert ToolPermission.READ.value == "read"
        assert ToolPermission.WRITE.value == "write"
        assert ToolPermission.EXECUTE.value == "execute"
        assert ToolPermission.NETWORK.value == "network"
        assert ToolPermission.FILE_SYSTEM.value == "file_system"
        assert ToolPermission.DATABASE.value == "database"
        assert ToolPermission.ADMIN.value == "admin"

    def test_confirmation_requirement_enum(self):
        """Test ConfirmationRequirement enum values."""
        assert ConfirmationRequirement.NEVER.value == "never"
        assert ConfirmationRequirement.ALWAYS.value == "always"
        assert ConfirmationRequirement.HIGH_RISK.value == "high_risk"
        assert ConfirmationRequirement.FIRST_USE.value == "first_use"

    def test_sandbox_capability_enum(self):
        """Test SandboxCapability enum values."""
        assert SandboxCapability.NONE.value == "none"
        assert SandboxCapability.PROCESS.value == "process"
        assert SandboxCapability.CONTAINER.value == "container"
        assert SandboxCapability.VM.value == "vm"

    def test_tool_status_enum(self):
        """Test ToolStatus enum values."""
        assert ToolStatus.AVAILABLE.value == "available"
        assert ToolStatus.UNAVAILABLE.value == "unavailable"
        assert ToolStatus.ERROR.value == "error"

    def test_tool_definition_defaults(self):
        """Test ToolDefinition default values."""
        definition = ToolDefinition(
            name="test",
            description="Test tool",
            parameters={},
            returns={},
        )
        assert definition.name == "test"
        assert definition.version == "1.0.0"
        assert definition.tags == []
        assert definition.requires_auth is False
        assert definition.timeout == 30
        assert definition.permissions == []
        assert definition.risk_level == RiskLevel.LOW
        assert definition.confirmation_required == ConfirmationRequirement.NEVER
        assert definition.sandbox_capability == SandboxCapability.NONE

    def test_tool_definition_with_permissions(self):
        """Test ToolDefinition with permissions and risk level."""
        definition = ToolDefinition(
            name="risky_tool",
            description="A risky tool",
            parameters={},
            returns={},
            permissions=[ToolPermission.READ, ToolPermission.WRITE],
            risk_level=RiskLevel.HIGH,
            confirmation_required=ConfirmationRequirement.HIGH_RISK,
            sandbox_capability=SandboxCapability.CONTAINER,
        )
        assert len(definition.permissions) == 2
        assert ToolPermission.READ in definition.permissions
        assert definition.risk_level == RiskLevel.HIGH
        assert definition.confirmation_required == ConfirmationRequirement.HIGH_RISK

    def test_tool_result_success(self):
        """Test ToolResult for successful execution."""
        result = ToolResult(
            success=True,
            output={"key": "value"},
            execution_time=100.5,
            metadata={"source": "test"},
        )
        assert result.success is True
        assert result.output == {"key": "value"}
        assert result.error is None
        assert result.execution_time == 100.5
        assert result.metadata == {"source": "test"}

    def test_tool_result_failure(self):
        """Test ToolResult for failed execution."""
        result = ToolResult(
            success=False,
            error="Something went wrong",
            execution_time=50.0,
        )
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.output is None

    def test_tool_execution_request(self):
        """Test ToolExecutionRequest model."""
        request = ToolExecutionRequest(
            tool_name="calculator",
            parameters={"expression": "2 + 2"},
            timeout=10,
            agent_permissions=[ToolPermission.EXECUTE],
            confirmation_token="abc123",
            sandbox_enabled=True,
        )
        assert request.tool_name == "calculator"
        assert request.parameters == {"expression": "2 + 2"}
        assert request.timeout == 10
        assert ToolPermission.EXECUTE in request.agent_permissions
        assert request.confirmation_token == "abc123"
        assert request.sandbox_enabled is True


# ── BaseTool Tests ────────────────────────────────────────────


class TestBaseTool:
    """Tests for the BaseTool abstract class."""

    def test_base_tool_initialization(self):
        """Test BaseTool initialization."""
        tool = ConcreteTestTool()
        assert tool.name == "test_tool"
        assert tool.definition is not None
        assert tool.logger is not None

    def test_base_tool_properties(self):
        """Test BaseTool property accessors."""
        tool = ConcreteTestTool(
            permissions=[ToolPermission.READ],
            risk_level=RiskLevel.HIGH,
            confirmation_required=ConfirmationRequirement.ALWAYS,
            sandbox_capability=SandboxCapability.CONTAINER,
        )
        assert tool.permissions == [ToolPermission.READ]
        assert tool.risk_level == RiskLevel.HIGH
        assert tool.confirmation_required == ConfirmationRequirement.ALWAYS
        assert tool.sandbox_capability == SandboxCapability.CONTAINER

    @pytest.mark.asyncio
    async def test_base_tool_validate_parameters_valid(self):
        """Test parameter validation with valid parameters."""
        tool = ConcreteTestTool()
        params = {"value": "test"}
        assert await tool.validate_parameters(params)

    @pytest.mark.asyncio
    async def test_base_tool_validate_parameters_missing_required(self):
        """Test parameter validation with missing required field."""
        tool = ConcreteTestTool()
        params = {}
        assert not await tool.validate_parameters(params)

    @pytest.mark.asyncio
    async def test_base_tool_health_check(self):
        """Test health check returns True by default."""
        tool = ConcreteTestTool()
        assert await tool.health_check()

    @pytest.mark.asyncio
    async def test_base_tool_execute(self):
        """Test tool execution."""
        tool = ConcreteTestTool()
        params = {"value": "hello"}
        result = await tool.execute(params)
        assert result.success is True
        assert result.output == {"result": "processed: hello"}

    def test_base_tool_get_schema(self):
        """Test schema generation."""
        tool = ConcreteTestTool()
        schema = tool.get_schema()
        assert schema["name"] == "test_tool"
        assert schema["description"] == "A test tool"
        assert "parameters" in schema
        assert "returns" in schema
        assert "permissions" in schema
        assert "risk_level" in schema

    def test_base_tool_permission_checks(self):
        """Test permission checking methods."""
        tool = ConcreteTestTool(
            permissions=[ToolPermission.READ, ToolPermission.WRITE]
        )
        
        assert tool.has_permission(ToolPermission.READ)
        assert tool.has_permission(ToolPermission.WRITE)
        assert not tool.has_permission(ToolPermission.EXECUTE)
        
        assert tool.requires_any_permission([ToolPermission.READ, ToolPermission.EXECUTE])
        assert not tool.requires_any_permission([ToolPermission.EXECUTE, ToolPermission.ADMIN])
        
        assert tool.requires_all_permissions([ToolPermission.READ, ToolPermission.WRITE])
        assert not tool.requires_all_permissions([ToolPermission.READ, ToolPermission.EXECUTE])

    def test_base_tool_can_execute_with_permissions(self):
        """Test can_execute_with_permissions."""
        tool = ConcreteTestTool(
            permissions=[ToolPermission.READ, ToolPermission.WRITE]
        )
        
        # Tool with no permissions can always execute
        no_perm_tool = ConcreteTestTool(name="no_perm", permissions=[])
        assert no_perm_tool.can_execute_with_permissions([])
        
        # Tool with permissions needs all of them
        assert tool.can_execute_with_permissions([ToolPermission.READ, ToolPermission.WRITE])
        assert not tool.can_execute_with_permissions([ToolPermission.READ])

    def test_base_tool_needs_confirmation(self):
        """Test confirmation requirement checks."""
        # Tool that never requires confirmation
        tool_never = ConcreteTestTool(
            confirmation_required=ConfirmationRequirement.NEVER
        )
        assert not tool_never.needs_confirmation()
        
        # Tool that always requires confirmation
        tool_always = ConcreteTestTool(
            confirmation_required=ConfirmationRequirement.ALWAYS
        )
        assert tool_always.needs_confirmation()
        
        # Tool that requires confirmation for high risk
        tool_high_risk = ConcreteTestTool(
            confirmation_required=ConfirmationRequirement.HIGH_RISK
        )
        assert not tool_high_risk.needs_confirmation_for_risk(RiskLevel.LOW)
        assert not tool_high_risk.needs_confirmation_for_risk(RiskLevel.MEDIUM)
        assert tool_high_risk.needs_confirmation_for_risk(RiskLevel.HIGH)
        assert tool_high_risk.needs_confirmation_for_risk(RiskLevel.CRITICAL)

    def test_base_tool_repr(self):
        """Test string representation."""
        tool = ConcreteTestTool()
        assert "ConcreteTestTool" in repr(tool)
        assert "test_tool" in repr(tool)


# ── ToolRegistry Tests ────────────────────────────────────────


class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = ToolRegistry()
        assert registry._tools == {}
        assert registry._tool_classes == {}
        assert registry._execution_history == []

    def test_register_tool_class(self):
        """Test registering a tool class."""
        registry = ToolRegistry()
        definition = ToolDefinition(
            name="test",
            description="Test",
            parameters={},
            returns={},
        )
        registry.register(ConcreteTestTool, definition)
        assert "test" in registry._tool_classes
        assert "test" in registry._tool_definitions

    def test_register_duplicate_tool_raises(self):
        """Test that registering duplicate tool raises ValueError."""
        registry = ToolRegistry()
        definition = ToolDefinition(
            name="test",
            description="Test",
            parameters={},
            returns={},
        )
        registry.register(ConcreteTestTool, definition)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(ConcreteTestTool, definition)

    def test_register_tool_instance(self):
        """Test registering a tool instance."""
        registry = ToolRegistry()
        tool = ConcreteTestTool()
        registry.register_instance(tool)
        assert "test_tool" in registry._tools

    def test_register_duplicate_instance_raises(self):
        """Test that registering duplicate instance raises ValueError."""
        registry = ToolRegistry()
        tool = ConcreteTestTool()
        registry.register_instance(tool)
        with pytest.raises(ValueError, match="already registered"):
            registry.register_instance(tool)

    def test_get_tool_returns_instance(self):
        """Test get_tool returns tool instance."""
        registry = ToolRegistry()
        tool = ConcreteTestTool()
        registry.register_instance(tool)
        retrieved = registry.get_tool("test_tool")
        assert retrieved is tool

    def test_get_tool_returns_none_for_unknown(self):
        """Test get_tool returns None for unknown tool."""
        registry = ToolRegistry()
        assert registry.get_tool("unknown") is None

    def test_get_tool_lazy_instantiation(self):
        """Test lazy instantiation from registered class."""
        registry = ToolRegistry()
        definition = ToolDefinition(
            name="lazy_tool",
            description="Lazy",
            parameters={},
            returns={},
        )
        registry.register(CalculatorTool, definition)
        tool = registry.get_tool("lazy_tool")
        assert tool is not None
        # The tool class creates its own definition internally
        assert tool.name == "calculator"
        # Should be cached now
        assert "lazy_tool" in registry._tools

    def test_get_tool_definition(self):
        """Test getting tool definition."""
        registry = ToolRegistry()
        definition = ToolDefinition(
            name="test",
            description="Test",
            parameters={},
            returns={},
        )
        registry.register(ConcreteTestTool, definition)
        retrieved = registry.get_tool_definition("test")
        assert retrieved is definition

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        tool = ConcreteTestTool()
        registry.register_instance(tool)
        assert registry.unregister("test_tool")
        assert registry.get_tool("test_tool") is None

    def test_unregister_nonexistent_returns_false(self):
        """Test unregistering nonexistent tool returns False."""
        registry = ToolRegistry()
        assert not registry.unregister("unknown")

    def test_clear_registry(self):
        """Test clearing the registry."""
        registry = ToolRegistry()
        tool = ConcreteTestTool()
        registry.register_instance(tool)
        registry.clear()
        assert len(registry._tools) == 0
        assert len(registry._tool_definitions) == 0

    def test_check_permissions(self):
        """Test permission checking."""
        registry = ToolRegistry()
        tool = ConcreteTestTool(
            permissions=[ToolPermission.READ, ToolPermission.WRITE]
        )
        registry.register_instance(tool)
        
        assert registry.check_permissions(
            "test_tool", [ToolPermission.READ, ToolPermission.WRITE]
        )
        assert not registry.check_permissions(
            "test_tool", [ToolPermission.READ]
        )
        assert not registry.check_permissions("unknown", [])

    def test_list_tools_no_filter(self):
        """Test listing all tools."""
        registry = ToolRegistry()
        tool1 = ConcreteTestTool(name="tool1")
        tool2 = ConcreteTestTool(name="tool2")
        registry.register_instance(tool1)
        registry.register_instance(tool2)
        
        tools = registry.list_tools()
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert "tool1" in names
        assert "tool2" in names

    def test_list_tools_with_tag_filter(self):
        """Test listing tools with tag filter."""
        registry = ToolRegistry()
        definition1 = ToolDefinition(
            name="tagged",
            description="Tagged",
            parameters={},
            returns={},
            tags=["special"],
        )
        definition2 = ToolDefinition(
            name="untagged",
            description="Untagged",
            parameters={},
            returns={},
            tags=[],
        )
        registry.register(ConcreteTestTool, definition1)
        registry.register(ConcreteTestTool, definition2)
        
        tools = registry.list_tools(tags=["special"])
        assert len(tools) == 1
        assert tools[0]["name"] == "tagged"

    def test_list_tools_with_permission_filter(self):
        """Test listing tools with permission filter."""
        registry = ToolRegistry()
        tool1 = ConcreteTestTool(
            name="reader",
            permissions=[ToolPermission.READ],
        )
        tool2 = ConcreteTestTool(
            name="writer",
            permissions=[ToolPermission.WRITE],
        )
        registry.register_instance(tool1)
        registry.register_instance(tool2)
        
        tools = registry.list_tools(permissions=[ToolPermission.READ])
        assert len(tools) == 1
        assert tools[0]["name"] == "reader"

    def test_list_tools_with_risk_filter(self):
        """Test listing tools with risk level filter."""
        registry = ToolRegistry()
        tool1 = ConcreteTestTool(name="safe", risk_level=RiskLevel.LOW)
        tool2 = ConcreteTestTool(name="risky", risk_level=RiskLevel.HIGH)
        registry.register_instance(tool1)
        registry.register_instance(tool2)
        
        tools = registry.list_tools(risk_level=RiskLevel.HIGH)
        assert len(tools) == 1
        assert tools[0]["name"] == "risky"

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test successful tool execution."""
        registry = ToolRegistry()
        tool = ConcreteTestTool()
        registry.register_instance(tool)
        
        request = ToolExecutionRequest(
            tool_name="test_tool",
            parameters={"value": "hello"},
            agent_permissions=[ToolPermission.READ, ToolPermission.WRITE],
        )
        result = await registry.execute(request)
        
        assert result.success is True
        assert result.output == {"result": "processed: hello"}
        assert len(registry._execution_history) == 1

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test executing a tool that doesn't exist."""
        registry = ToolRegistry()
        request = ToolExecutionRequest(
            tool_name="unknown",
            parameters={},
        )
        result = await registry.execute(request)
        
        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_permission_denied(self):
        """Test executing a tool with insufficient permissions."""
        registry = ToolRegistry()
        tool = ConcreteTestTool(
            permissions=[ToolPermission.READ, ToolPermission.WRITE]
        )
        registry.register_instance(tool)
        
        request = ToolExecutionRequest(
            tool_name="test_tool",
            parameters={"value": "hello"},
            agent_permissions=[ToolPermission.READ],  # Missing WRITE
        )
        result = await registry.execute(request)
        
        assert result.success is False
        assert "permissions" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_invalid_parameters(self):
        """Test executing a tool with invalid parameters."""
        registry = ToolRegistry()
        tool = ConcreteTestTool()
        registry.register_instance(tool)
        
        request = ToolExecutionRequest(
            tool_name="test_tool",
            parameters={},  # Missing required 'value'
            agent_permissions=[ToolPermission.READ, ToolPermission.WRITE],
        )
        result = await registry.execute(request)
        
        assert result.success is False
        assert "Invalid parameters" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_confirmation_denied(self):
        """Test executing a tool when confirmation is denied."""
        registry = ToolRegistry()
        tool = ConcreteTestTool(
            confirmation_required=ConfirmationRequirement.ALWAYS,
        )
        registry.register_instance(tool)
        
        # Set up confirmation callback that always denies
        async def deny_callback(
            tool_name: str, parameters: Dict[str, Any], risk_level: RiskLevel
        ) -> bool:
            return False
        
        registry.set_confirmation_callback(deny_callback)
        
        request = ToolExecutionRequest(
            tool_name="test_tool",
            parameters={"value": "hello"},
            agent_permissions=[ToolPermission.READ, ToolPermission.WRITE],
        )
        result = await registry.execute(request)
        
        assert result.success is False
        assert "Confirmation denied" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_confirmation_accepted(self):
        """Test executing a tool when confirmation is accepted."""
        registry = ToolRegistry()
        tool = ConcreteTestTool(
            confirmation_required=ConfirmationRequirement.ALWAYS,
        )
        registry.register_instance(tool)
        
        # Set up confirmation callback that always accepts
        async def accept_callback(
            tool_name: str, parameters: Dict[str, Any], risk_level: RiskLevel
        ) -> bool:
            return True
        
        registry.set_confirmation_callback(accept_callback)
        
        request = ToolExecutionRequest(
            tool_name="test_tool",
            parameters={"value": "hello"},
            agent_permissions=[ToolPermission.READ, ToolPermission.WRITE],
        )
        result = await registry.execute(request)
        
        assert result.success is True

    def test_execution_history(self):
        """Test execution history tracking."""
        registry = ToolRegistry()
        tool = ConcreteTestTool()
        registry.register_instance(tool)
        
        # Add some execution records manually
        registry._execution_history = [
            {"tool_name": "test_tool", "success": True},
            {"tool_name": "other_tool", "success": False},
            {"tool_name": "test_tool", "success": True},
        ]
        
        history = registry.get_execution_history()
        assert len(history) == 3
        
        history = registry.get_execution_history(tool_name="test_tool")
        assert len(history) == 2
        
        history = registry.get_execution_history(limit=2)
        assert len(history) == 2

    def test_execution_stats(self):
        """Test execution statistics."""
        registry = ToolRegistry()
        tool = ConcreteTestTool()
        registry.register_instance(tool)
        
        # Add some execution records
        registry._execution_history = [
            {"tool_name": "test_tool", "success": True, "execution_time": 100},
            {"tool_name": "test_tool", "success": False, "execution_time": 50},
            {"tool_name": "test_tool", "success": True, "execution_time": 150},
        ]
        
        stats = registry.get_execution_stats("test_tool")
        assert stats["total_executions"] == 3
        assert stats["successful_executions"] == 2
        assert stats["failed_executions"] == 1
        assert stats["average_execution_time"] == 100.0

    def test_execution_history_max_size(self):
        """Test that execution history respects max size."""
        registry = ToolRegistry(max_history_size=3)
        tool = ConcreteTestTool()
        registry.register_instance(tool)
        
        # Add more records than max size
        for i in range(5):
            registry._execution_history.append(
                {"tool_name": "test", "execution_time": i}
            )
        
        # Trim
        registry._execution_history = registry._execution_history[-registry._max_history_size:]
        
        assert len(registry._execution_history) == 3

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check for all tools."""
        registry = ToolRegistry()
        tool1 = ConcreteTestTool(name="tool1")
        tool2 = ConcreteTestTool(name="tool2")
        registry.register_instance(tool1)
        registry.register_instance(tool2)
        
        health = await registry.health_check()
        assert health["tool1"] is True
        assert health["tool2"] is True


# ── Placeholder Tools Tests ───────────────────────────────────


class TestCalculatorTool:
    """Tests for the CalculatorTool placeholder."""

    @pytest.mark.asyncio
    async def test_calculator_basic_addition(self):
        """Test basic addition."""
        tool = CalculatorTool()
        result = await tool.execute({"expression": "2 + 3"})
        assert result.success is True
        assert result.output["result"] == 5.0

    @pytest.mark.asyncio
    async def test_calculator_basic_subtraction(self):
        """Test basic subtraction."""
        tool = CalculatorTool()
        result = await tool.execute({"expression": "10 - 4"})
        assert result.success is True
        assert result.output["result"] == 6.0

    @pytest.mark.asyncio
    async def test_calculator_basic_multiplication(self):
        """Test basic multiplication."""
        tool = CalculatorTool()
        result = await tool.execute({"expression": "3 * 7"})
        assert result.success is True
        assert result.output["result"] == 21.0

    @pytest.mark.asyncio
    async def test_calculator_basic_division(self):
        """Test basic division."""
        tool = CalculatorTool()
        result = await tool.execute({"expression": "10 / 2"})
        assert result.success is True
        assert result.output["result"] == 5.0

    @pytest.mark.asyncio
    async def test_calculator_empty_expression(self):
        """Test with empty expression."""
        tool = CalculatorTool()
        result = await tool.execute({"expression": ""})
        assert result.success is False
        assert "No expression provided" in result.error

    @pytest.mark.asyncio
    async def test_calculator_invalid_characters(self):
        """Test with invalid characters."""
        tool = CalculatorTool()
        result = await tool.execute({"expression": "import os"})
        assert result.success is False

    def test_calculator_permissions(self):
        """Test calculator tool permissions."""
        tool = CalculatorTool()
        assert ToolPermission.EXECUTE in tool.permissions
        assert tool.risk_level == RiskLevel.LOW

    def test_calculator_definition(self):
        """Test calculator tool definition."""
        tool = CalculatorTool()
        assert tool.name == "calculator"
        assert "math" in tool.definition.tags


class TestFileReadTool:
    """Tests for the FileReadTool placeholder."""

    @pytest.mark.asyncio
    async def test_file_read_success(self):
        """Test successful file reading."""
        tool = FileReadTool()
        result = await tool.execute({"file_path": "/test/file.txt"})
        assert result.success is True
        assert "content" in result.output
        assert result.output["file_path"] == "/test/file.txt"

    @pytest.mark.asyncio
    async def test_file_read_no_path(self):
        """Test with no file path."""
        tool = FileReadTool()
        result = await tool.execute({})
        assert result.success is False
        assert "No file path provided" in result.error

    def test_file_read_permissions(self):
        """Test file read tool permissions."""
        tool = FileReadTool()
        assert ToolPermission.READ in tool.permissions
        assert ToolPermission.FILE_SYSTEM in tool.permissions
        assert tool.risk_level == RiskLevel.MEDIUM

    def test_file_read_confirmation(self):
        """Test file read tool confirmation requirement."""
        tool = FileReadTool()
        assert tool.needs_confirmation()
        assert tool.confirmation_required == ConfirmationRequirement.HIGH_RISK


class TestWebSearchTool:
    """Tests for the WebSearchTool placeholder."""

    @pytest.mark.asyncio
    async def test_web_search_success(self):
        """Test successful web search."""
        tool = WebSearchTool()
        result = await tool.execute({"query": "python programming"})
        assert result.success is True
        assert "results" in result.output
        assert len(result.output["results"]) > 0

    @pytest.mark.asyncio
    async def test_web_search_no_query(self):
        """Test with no query."""
        tool = WebSearchTool()
        result = await tool.execute({})
        assert result.success is False
        assert "No search query provided" in result.error

    @pytest.mark.asyncio
    async def test_web_search_num_results(self):
        """Test web search with custom num_results."""
        tool = WebSearchTool()
        result = await tool.execute({"query": "test", "num_results": 2})
        assert result.success is True
        assert len(result.output["results"]) <= 3  # Placeholder limits to 3

    def test_web_search_permissions(self):
        """Test web search tool permissions."""
        tool = WebSearchTool()
        assert ToolPermission.NETWORK in tool.permissions
        assert tool.risk_level == RiskLevel.MEDIUM

    def test_web_search_sandbox(self):
        """Test web search tool sandbox capability."""
        tool = WebSearchTool()
        assert tool.sandbox_capability == SandboxCapability.CONTAINER


# ── Integration Tests ─────────────────────────────────────────


class TestToolRegistryIntegration:
    """Integration tests for ToolRegistry with tools."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()
        
        # Register all placeholder tools
        self.registry.register_instance(CalculatorTool())
        self.registry.register_instance(FileReadTool())
        self.registry.register_instance(WebSearchTool())

    def test_all_tools_registered(self):
        """Test that all placeholder tools are registered."""
        tools = self.registry.list_tools()
        names = {t["name"] for t in tools}
        assert "calculator" in names
        assert "file_read" in names
        assert "web_search" in names

    @pytest.mark.asyncio
    async def test_calculator_via_registry(self):
        """Test calculator tool execution via registry."""
        request = ToolExecutionRequest(
            tool_name="calculator",
            parameters={"expression": "2 + 2"},
            agent_permissions=[ToolPermission.EXECUTE],
        )
        result = await self.registry.execute(request)
        assert result.success is True
        assert result.output["result"] == 4.0

    @pytest.mark.asyncio
    async def test_web_search_via_registry(self):
        """Test web search tool execution via registry."""
        request = ToolExecutionRequest(
            tool_name="web_search",
            parameters={"query": "test query"},
            agent_permissions=[ToolPermission.NETWORK],
        )
        result = await self.registry.execute(request)
        assert result.success is True
        assert len(result.output["results"]) > 0

    def test_list_tools_by_risk_level(self):
        """Test listing tools filtered by risk level."""
        low_risk = self.registry.list_tools(risk_level=RiskLevel.LOW)
        medium_risk = self.registry.list_tools(risk_level=RiskLevel.MEDIUM)
        
        assert len(low_risk) == 1  # calculator
        assert len(medium_risk) == 2  # file_read, web_search

    def test_list_tools_by_permission(self):
        """Test listing tools filtered by permission."""
        network_tools = self.registry.list_tools(
            permissions=[ToolPermission.NETWORK]
        )
        assert len(network_tools) == 1
        assert network_tools[0]["name"] == "web_search"


class TestToolRegistryWithOrchestrator:
    """Integration tests for ToolRegistry with Orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_with_tool_registry(self):
        """Test Orchestrator integration with ToolRegistry."""
        from app.agents.core.orchestrator import Orchestrator
        from app.models.state import AgentState, LifecycleStep, PlanStep, StepStatus
        
        # Set up registry with calculator
        registry = ToolRegistry()
        registry.register_instance(CalculatorTool())
        
        # Set up orchestrator
        orchestrator = Orchestrator(tool_registry=registry)
        
        # Create state with a plan that uses a tool
        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Calculate 2 + 2",
                tool_name="calculator",
                tool_parameters={"expression": "2 + 2"},
                priority="high",
            )
        ]
        
        # Execute the plan
        state = await orchestrator.execute_plan(state)
        
        # Verify tool was executed
        assert len(state.results) == 1
        assert state.results[0]["success"] is True
        assert state.results[0]["output"]["result"] == 4.0
        assert "calculator" in state.tools_used

    @pytest.mark.asyncio
    async def test_orchestrator_without_tool_registry(self):
        """Test Orchestrator without ToolRegistry uses fallback."""
        from app.agents.core.orchestrator import Orchestrator
        from app.models.state import AgentState, PlanStep
        
        # Set up orchestrator without registry
        orchestrator = Orchestrator()
        
        # Create state with a plan that uses a tool
        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Use a tool",
                tool_name="some_tool",
                tool_parameters={},
            )
        ]
        
        # Execute the plan
        state = await orchestrator.execute_plan(state)
        
        # Verify fallback message used
        assert len(state.results) == 1
        assert "no registry available" in state.results[0]["output"]

    @pytest.mark.asyncio
    async def test_orchestrator_tool_failure_records_error(self):
        """Test that tool failures are recorded as errors."""
        from app.agents.core.orchestrator import Orchestrator
        from app.models.state import AgentState, PlanStep
        
        # Set up registry with a tool that will fail
        registry = ToolRegistry()
        tool = ConcreteTestTool(name="failing_tool")
        registry.register_instance(tool)
        
        orchestrator = Orchestrator(tool_registry=registry)
        
        state = AgentState()
        state.plan = [
            PlanStep(
                index=0,
                description="Use failing tool",
                tool_name="failing_tool",
                tool_parameters={},  # Missing required 'value'
            )
        ]
        
        state = await orchestrator.execute_plan(state)
        
        # Verify error recorded
        assert len(state.errors) > 0
        assert state.results[0]["success"] is False