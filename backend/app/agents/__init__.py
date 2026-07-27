"""
Agents Package

This package contains all agent implementations and the base agent class.
Agents are the core processing units that handle tasks using tools and memory.
"""

from app.agents.base import BaseAgent, AgentConfig
from app.agents.specialized.planner import PlannerAgent
from app.agents.specialized.executor import ExecutorAgent
from app.agents.specialized.research import ResearchAgent
from app.agents.specialized.coding import CodingAgent
from app.agents.specialized.security import SecurityAgent
from app.agents.specialized.qa import QAAgent
from app.agents.specialized.supervisor import SupervisorAgent
from app.agents.specialized.registry import AgentRegistry, AgentNotFoundError, DuplicateAgentError

__all__ = [
    "BaseAgent",
    "AgentConfig",
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