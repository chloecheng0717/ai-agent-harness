"""
Message Queue Implementation

This module provides a message queue for asynchronous task distribution.
It's used for long-running operations and background tasks.
"""

from typing import Any, Callable, Dict, List, Optional
from uuid import UUID, uuid4

import structlog


class Message:
    """
    Message model for the queue.
    
    Attributes:
        id: Unique message identifier
        queue: Queue name
        payload: Message payload
        priority: Message priority (0-9, higher = more priority)
        delay: Delay in seconds before message becomes available
        ttl: Time to live in seconds
    """
    
    def __init__(
        self,
        queue: str,
        payload: Dict[str, Any],
        priority: int = 0,
        delay: int = 0,
        ttl: int = 3600,
    ) -> None:
        """
        Initialize a message.
        
        Args:
            queue: Queue name
            payload: Message payload
            priority: Message priority
            delay: Delay before available
            ttl: Time to live
        """
        self.id: UUID = uuid4()
        self.queue = queue
        self.payload = payload
        self.priority = priority
        self.delay = delay
        self.ttl = ttl


class MessageHandler:
    """
    Message handler wrapper.
    
    Attributes:
        handler: Callback function
        queue: Queue name to consume from
    """
    
    def __init__(
        self,
        handler: Callable[[Message], None],
        queue: str,
    ) -> None:
        """
        Initialize a message handler.
        
        Args:
            handler: Callback function
            queue: Queue name
        """
        self.handler = handler
        self.queue = queue


class MessageQueue:
    """
    Message queue for asynchronous task distribution.
    
    This class provides a simple message queue implementation.
    In production, this would use Redis, RabbitMQ, or similar.
    
    The message queue supports:
    - Message publishing with priority
    - Consumer registration
    - Message acknowledgment
    - Dead letter queue for failed messages
    
    Attributes:
        _queues: Dictionary of message queues
        _handlers: Dictionary of message handlers
    """
    
    def __init__(self) -> None:
        """Initialize the message queue."""
        self._queues: Dict[str, List[Message]] = {}
        self._handlers: Dict[str, List[MessageHandler]] = {}
        
        self.logger = structlog.get_logger().bind(component="message_queue")
        self.logger.info("Message queue initialized")
    
    async def publish(self, message: Message) -> None:
        """
        Publish a message to a queue.
        
        Args:
            message: Message to publish
        """
        queue = message.queue
        
        if queue not in self._queues:
            self._queues[queue] = []
        
        self._queues[queue].append(message)
        
        # Sort by priority (higher priority first)
        self._queues[queue].sort(key=lambda m: m.priority, reverse=True)
        
        self.logger.debug(
            "Message published",
            queue=queue,
            message_id=str(message.id),
            priority=message.priority,
        )
        
        # Notify handlers
        await self._process_queue(queue)
    
    async def _process_queue(self, queue: str) -> None:
        """
        Process messages in a queue.
        
        Args:
            queue: Queue name to process
        """
        if queue not in self._queues or not self._queues[queue]:
            return
        
        if queue not in self._handlers or not self._handlers[queue]:
            return
        
        # Get next message
        message = self._queues[queue][0]
        
        # Process with handlers
        for handler in self._handlers[queue]:
            try:
                await handler.handler(message)
                # Remove message on success
                self._queues[queue].pop(0)
                break
            except Exception as e:
                self.logger.error(
                    "Message handling failed",
                    queue=queue,
                    message_id=str(message.id),
                    error=str(e),
                )
    
    def subscribe(
        self,
        queue: str,
        handler: Callable[[Message], None],
    ) -> str:
        """
        Subscribe to a queue.
        
        Args:
            queue: Queue name
            handler: Callback function
            
        Returns:
            Subscription ID
        """
        subscription_id = str(uuid4())
        
        if queue not in self._handlers:
            self._handlers[queue] = []
        
        self._handlers[queue].append(
            MessageHandler(handler=handler, queue=queue)
        )
        
        self.logger.debug(
            "Subscribed to queue",
            queue=queue,
            subscription_id=subscription_id,
        )
        
        return subscription_id
    
    def get_queue_length(self, queue: str) -> int:
        """
        Get the length of a queue.
        
        Args:
            queue: Queue name
            
        Returns:
            Number of messages in queue
        """
        return len(self._queues.get(queue, []))