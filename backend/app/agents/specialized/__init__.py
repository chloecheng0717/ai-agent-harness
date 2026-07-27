"""
Specialized Agents Package

This package contains specialized agent implementations:
- Planner Agent: Task decomposition and planning
- Executor Agent: Task execution with tools
- Research Agent: Information gathering and analysis
- Coding Agent: Code writing, debugging, and explanation
- Security Agent: Vulnerability analysis and risk assessment
- QA Agent: Testing and quality verification
- Supervisor Agent: Multi-agent coordination
- Agent Registry: Centralized agent management
"""

from app.agents.specialized.planner import PlannerAgent
from app.agents.specialized.executor import ExecutorAgent
from app.agents.specialized.research import ResearchAgent
from app.agents.specialized.coding import CodingAgent
from app.agents.specialized.security import SecurityAgent
from app.agents.specialized.qa import QAAgent
from app.agents.specialized.supervisor import SupervisorAgent
from app.agents.specialized.registry import AgentRegistry, AgentNotFoundError, DuplicateAgentError

__all__ = [
    "PlannerAgent",
    "ExecutorAgent",
    "ResearchAgent",
    "CodingAgent",
    "SecurityAgent",
    "QAAgent",
    "SupervisorAgent",
    "AgentRegistry",
    "AgentNotFoundError",
    "DuplicateAgentError",
]