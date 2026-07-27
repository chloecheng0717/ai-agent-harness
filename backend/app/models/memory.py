"""
Memory Models

This module defines Pydantic models for memory-related data structures.
Memory systems store conversation history, knowledge, and experiences.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """
    Memory type enumeration.
    
    Defines the different types of memory storage available.
    """
    SHORT_TERM = "short_term"  # Conversation history (Redis)
    LONG_TERM = "long_term"    # Knowledge base (Vector DB)
    EPISODIC = "episodic"      # Task experiences (PostgreSQL)


class MemoryEntry(BaseModel):
    """
    Memory entry model.
    
    This model represents a single memory entry that can be stored
    in any of the memory systems.
    
    Attributes:
        id: Unique memory identifier
        type: Type of memory storage
        content: Memory content (text or structured data)
        metadata: Additional metadata about the memory
        embedding: Vector embedding for semantic search
        agent_id: ID of the agent that created this memory
        task_id: ID of the task associated with this memory
        created_at: Memory creation timestamp
        expires_at: Optional expiration timestamp (for short-term memory)
    """
    id: UUID = Field(default_factory=uuid4, description="Unique memory identifier")
    type: MemoryType = Field(..., description="Type of memory storage")
    content: str = Field(..., description="Memory content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for semantic search"
    )
    agent_id: Optional[UUID] = Field(
        default=None,
        description="ID of the agent that created this memory"
    )
    task_id: Optional[UUID] = Field(
        default=None,
        description="ID of the task associated with this memory"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Memory creation timestamp"
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Expiration timestamp (for short-term memory)"
    )


class MemorySearchResult(BaseModel):
    """
    Memory search result model.
    
    This model represents a memory search result, including
    relevance scoring and the memory entry itself.
    
    Attributes:
        memory: The memory entry
        score: Relevance score (0-1)
        distance: Vector distance (if using vector search)
    """
    memory: MemoryEntry = Field(..., description="Memory entry")
    score: float = Field(..., ge=0, le=1, description="Relevance score")
    distance: Optional[float] = Field(
        default=None,
        description="Vector distance"
    )


class MemorySearchQuery(BaseModel):
    """
    Memory search query model.
    
    This model defines the parameters for searching memory.
    
    Attributes:
        query: Search query text
        type: Optional filter by memory type
        agent_id: Optional filter by agent ID
        task_id: Optional filter by task ID
        limit: Maximum number of results
        min_score: Minimum relevance score threshold
    """
    query: str = Field(..., description="Search query text")
    type: Optional[MemoryType] = Field(
        default=None,
        description="Filter by memory type"
    )
    agent_id: Optional[UUID] = Field(
        default=None,
        description="Filter by agent ID"
    )
    task_id: Optional[UUID] = Field(
        default=None,
        description="Filter by task ID"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results"
    )
    min_score: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Minimum relevance score"
    )