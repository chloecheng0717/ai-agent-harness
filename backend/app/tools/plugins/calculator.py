"""
Calculator Tool

A placeholder tool that performs basic arithmetic calculations.
This is a simple example to demonstrate the tool framework.
"""

import math
from typing import Any, Dict

import structlog

from app.models.tool import (
    ConfirmationRequirement,
    RiskLevel,
    SandboxCapability,
    ToolDefinition,
    ToolPermission,
    ToolResult,
)
from app.tools.base import BaseTool


class CalculatorTool(BaseTool):
    """
    Calculator tool for basic arithmetic operations.
    
    This tool supports basic arithmetic operations:
    - Addition, subtraction, multiplication, division
    - Power and square root
    - Basic trigonometric functions
    
    This is a placeholder implementation for demonstration purposes.
    """
    
    def __init__(self) -> None:
        """Initialize the calculator tool."""
        definition = ToolDefinition(
            name="calculator",
            description="Perform basic arithmetic calculations",
            version="1.0.0",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
            returns={
                "type": "object",
                "properties": {
                    "result": {
                        "type": "number",
                        "description": "Calculation result",
                    }
                },
            },
            tags=["math", "calculation", "arithmetic"],
            permissions=[ToolPermission.EXECUTE],
            risk_level=RiskLevel.LOW,
            confirmation_required=ConfirmationRequirement.NEVER,
            sandbox_capability=SandboxCapability.PROCESS,
        )
        super().__init__(definition)
        self.logger = structlog.get_logger().bind(component="calculator_tool")
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute the calculator tool.
        
        Args:
            parameters: Must contain 'expression' key with the math expression
            
        Returns:
            ToolResult with the calculation result
        """
        expression = parameters.get("expression", "")
        
        if not expression:
            return ToolResult(
                success=False,
                error="No expression provided",
            )
        
        try:
            # Simple safe evaluation (placeholder implementation)
            # In a real implementation, this would use a proper math parser
            result = self._safe_eval(expression)
            return ToolResult(
                success=True,
                output={"result": result},
                metadata={"expression": expression},
            )
        except Exception as e:
            self.logger.error(
                "Calculation failed",
                expression=expression,
                error=str(e),
            )
            return ToolResult(
                success=False,
                error=f"Calculation failed: {str(e)}",
            )
    
    def _safe_eval(self, expression: str) -> float:
        """
        Safely evaluate a mathematical expression.
        
        This is a simplified implementation that supports basic operations.
        In production, use a proper math parsing library.
        """
        # Basic math operations only
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Expression contains invalid characters")
        
        # Replace common math functions
        expression = expression.replace("^", "**")
        expression = expression.replace("sqrt", "math.sqrt")
        expression = expression.replace("sin", "math.sin")
        expression = expression.replace("cos", "math.cos")
        expression = expression.replace("tan", "math.tan")
        expression = expression.replace("pi", "math.pi")
        expression = expression.replace("e", "math.e")
        
        # Evaluate in a restricted namespace
        namespace = {"math": math, "__builtins__": {}}
        result = eval(expression, namespace)  # noqa: S307
        return float(result)