"""
Memory Package

This package contains memory system implementations:
- Short-term: Conversation history (in-memory sliding window, optional Redis)
- Long-term: Knowledge base (ChromaDB vector DB with in-memory fallback)
- Episodic: Task experiences (PostgreSQL with in-memory fallback)

MemoryManager provides a unified facade used by all agents.
"""

from app.memory.base import BaseMemory
from app.memory.short_term import ShortTermMemory
from app.memory.long_term import LongTermMemory
from app.memory.episodic import EpisodicMemory
from app.memory.manager import MemoryManager

__all__ = [
    "BaseMemory",
    "ShortTermMemory",
    "LongTermMemory",
    "EpisodicMemory",
    "MemoryManager",
]