"""
Communication Package

This package contains communication system implementations:
- Event Bus: Redis Pub/Sub for decoupled agent communication
- Message Queue: Asynchronous task distribution
- Agent Message: Inter-agent communication protocol

The communication system enables agents to communicate with each other
and with external systems through an event-driven architecture.
"""

from app.communication.agent_message import AgentMessage, MessageStatus

__all__ = [
    "AgentMessage",
    "MessageStatus",
]