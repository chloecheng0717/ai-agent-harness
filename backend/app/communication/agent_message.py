"""
Agent Message Protocol

This module defines the message model and protocol for inter-agent
communication.  Messages flow between a Supervisor and specialized agents
following a request/response pattern built on top of the existing
communication infrastructure (EventBus / MessageQueue).
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MessageStatus(str, Enum):
    """Status of an agent message."""
    PENDING = "pending"
    SENT = "sent"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentMessage(BaseModel):
    """
    Message model for inter-agent communication.

    Represents a single communication unit exchanged between agents
    (typically Supervisor -> Specialized Agent and back).

    Attributes:
        message_id: Unique message identifier
        sender: Identifier of the sending agent
        receiver: Identifier of the receiving agent
        timestamp: When the message was created
        task: Description of the task or question
        response: Response content (populated after processing)
        status: Current status of the message
        metadata: Arbitrary metadata (priority, context, etc.)
    """
    message_id: UUID = Field(
        default_factory=uuid4,
        description="Unique message identifier",
    )
    sender: str = Field(
        ...,
        description="Identifier of the sending agent",
    )
    receiver: str = Field(
        ...,
        description="Identifier of the receiving agent",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the message was created",
    )
    task: str = Field(
        ...,
        description="Description of the task or question",
    )
    response: Optional[str] = Field(
        default=None,
        description="Response content (populated after processing)",
    )
    status: MessageStatus = Field(
        default=MessageStatus.PENDING,
        description="Current status of the message",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata",
    )

    def mark_sent(self) -> None:
        """Mark the message as sent."""
        self.status = MessageStatus.SENT

    def mark_processing(self) -> None:
        """Mark the message as being processed."""
        self.status = MessageStatus.PROCESSING

    def mark_completed(self, response: str) -> None:
        """Mark the message as completed with a response."""
        self.response = response
        self.status = MessageStatus.COMPLETED

    def mark_failed(self, error: str) -> None:
        """Mark the message as failed."""
        self.response = error
        self.status = MessageStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the message to a dictionary."""
        return {
            "message_id": str(self.message_id),
            "sender": self.sender,
            "receiver": self.receiver,
            "timestamp": self.timestamp.isoformat(),
            "task": self.task,
            "response": self.response,
            "status": self.status.value,
            "metadata": self.metadata,
        }
