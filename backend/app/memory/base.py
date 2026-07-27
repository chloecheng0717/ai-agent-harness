"""
Base Memory Interface

This module defines the abstract base class for all memory implementations.
It provides the common interface that memory systems must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

import structlog

from app.models.memory import MemoryEntry, MemorySearchQuery, MemorySearchResult, MemoryType


class BaseMemory(ABC):
    """
    Abstract base class for memory implementations.
    
    This class defines the interface that all memory systems must implement.
    Memory systems store and retrieve information used by agents.
    
    The memory interface supports:
    - Adding new memories
    - Searching memories by query
    - Retrieving memories by ID
    - Updating existing memories
    - Deleting memories
    
    Attributes:
        memory_type: Type of memory storage
        name: Human-readable memory name
    """
    
    def __init__(self, memory_type: MemoryType, name: str) -> None:
        """
        Initialize the memory system.
        
        Args:
            memory_type: Type of memory storage
            name: Human-readable memory name
        """
        self.memory_type = memory_type
        self.name = name
        
        # Setup structured logging
        self.logger = structlog.get_logger().bind(
            memory_type=memory_type.value,
            memory_name=name,
        )
        
        self.logger.info("Memory system initialized")
    
    @abstractmethod
    async def add(self, entry: MemoryEntry) -> UUID:
        """
        Add a new memory entry.
        
        Args:
            entry: Memory entry to add
            
        Returns:
            UUID of the added memory entry
        """
        pass
    
    @abstractmethod
    async def search(self, query: MemorySearchQuery) -> List[MemorySearchResult]:
        """
        Search memories by query.
        
        Args:
            query: Search query with filters
            
        Returns:
            List of matching memory entries with relevance scores
        """
        pass
    
    @abstractmethod
    async def get(self, memory_id: UUID) -> Optional[MemoryEntry]:
        """
        Get a memory by ID.
        
        Args:
            memory_id: UUID of the memory to retrieve
            
        Returns:
            Memory entry if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def update(self, entry: MemoryEntry) -> bool:
        """
        Update an existing memory entry.
        
        Args:
            entry: Memory entry with updated data
            
        Returns:
            True if updated successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete(self, memory_id: UUID) -> bool:
        """
        Delete a memory entry.
        
        Args:
            memory_id: UUID of the memory to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def clear(self, agent_id: Optional[UUID] = None) -> int:
        """
        Clear memories, optionally filtered by agent.
        
        Args:
            agent_id: Optional agent ID to filter by
            
        Returns:
            Number of memories cleared
        """
        pass
    
    async def health_check(self) -> bool:
        """
        Check if the memory system is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Default implementation - subclasses can override
            return True
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False