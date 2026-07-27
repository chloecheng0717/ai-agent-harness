"""
Tools Package

This package contains the tool system implementations:
- Base: Abstract base class for all tools
- Registry: Tool registration and discovery
- Plugins: Tool implementations (API, file, web search, code execution)

Tools are plugins that agents can use to interact with external systems
and perform specific operations.
"""

from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry
from app.tools.plugins import CalculatorTool, FileReadTool, WebSearchTool

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "CalculatorTool",
    "FileReadTool",
    "WebSearchTool",
]