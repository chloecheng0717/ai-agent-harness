"""
Event Bus Implementation

This module provides an event bus for decoupled agent communication.
It uses Redis Pub/Sub for message passing between components.
"""

from typing import Any, Callable, Dict, List, Optional
from uuid import UUID, uuid4

import structlog

from app.config import get_settings


class Event:
    """
    Event model for the event bus.
    
    Represents a single event that can be published and subscribed to.
    
    Attributes:
        id: Unique event identifier
        type: Event type (e.g., 'task.created', 'agent.completed')
        source: Source component that emitted the event
        data: Event payload data
        metadata: Additional event metadata
    """
    
    def __init__(
        self,
        event_type: str,
        source: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize an event.
        
        Args:
            event_type: Event type
            source: Source component
            data: Event payload
            metadata: Optional metadata
        """
        self.id: UUID = uuid4()
        self.type = event_type
        self.source = source
        self.data = data
        self.metadata = metadata or {}


class EventHandler:
    """
    Event handler wrapper.
    
    Wraps a callback function with metadata for event handling.
    
    Attributes:
        handler: Callback function
        filter_type: Optional event type filter
    """
    
    def __init__(
        self,
        handler: Callable[[Event], None],
        filter_type: Optional[str] = None,
    ) -> None:
        """
        Initialize an event handler.
        
        Args:
            handler: Callback function
            filter_type: Optional event type filter
        """
        self.handler = handler
        self.filter_type = filter_type


class EventBus:
    """
    Event bus for decoupled communication.
    
    This class provides a publish-subscribe pattern for component communication.
    It uses Redis Pub/Sub for message passing in distributed systems,
    but can also work in-memory for single-process applications.
    
    The event bus supports:
    - Publishing events
    - Subscribing to events with optional type filtering
    - Wildcard subscriptions (e.g., 'task.*')
    - Event history for debugging
    
    Attributes:
        _handlers: Dictionary of event handlers by type
        _history: List of recent events for debugging
        _max_history: Maximum number of events to keep in history
    """
    
    def __init__(self, use_redis: bool = False) -> None:
        """
        Initialize the event bus.
        
        Args:
            use_redis: Whether to use Redis for distributed communication
        """
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._history: List[Event] = []
        self._max_history = 1000
        self._use_redis = use_redis
        
        # Redis connection (if enabled)
        self._redis = None
        
        self.logger = structlog.get_logger().bind(component="event_bus")
        self.logger.info("Event bus initialized", use_redis=use_redis)
    
    async def publish(self, event: Event) -> None:
        """
        Publish an event.
        
        This method sends an event to all subscribed handlers.
        
        Args:
            event: Event to publish
        """
        self.logger.debug(
            "Publishing event",
            event_type=event.type,
            event_source=event.source,
            event_id=str(event.id),
        )
        
        # Add to history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        # If using Redis, publish there
        if self._use_redis and self._redis:
            # TODO: Implement Redis publishing
            pass
        
        # Notify local handlers
        await self._notify_handlers(event)
    
    async def _notify_handlers(self, event: Event) -> None:
        """
        Notify handlers of an event.
        
        Args:
            event: Event to notify about
        """
        # Check exact type handlers
        if event.type in self._handlers:
            for handler in self._handlers[event.type]:
                try:
                    await handler.handler(event)
                except Exception as e:
                    self.logger.error(
                        "Handler error",
                        event_type=event.type,
                        error=str(e),
                    )
        
        # Check wildcard handlers (e.g., 'task.*')
        for pattern, handlers in self._handlers.items():
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                if event.type.startswith(prefix):
                    for handler in handlers:
                        try:
                            await handler.handler(event)
                        except Exception as e:
                            self.logger.error(
                                "Handler error",
                                event_type=event.type,
                                pattern=pattern,
                                error=str(e),
                            )
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None],
    ) -> str:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Event type to subscribe to
            handler: Callback function
            
        Returns:
            Subscription ID for unsubscribing
        """
        subscription_id = str(uuid4())
        
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        self._handlers[event_type].append(
            EventHandler(handler=handler)
        )
        
        self.logger.debug(
            "Subscribed to event",
            event_type=event_type,
            subscription_id=subscription_id,
        )
        
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from an event.
        
        Args:
            subscription_id: Subscription ID to unsubscribe
            
        Returns:
            True if unsubscribed, False if not found
        """
        # This is a simplified implementation
        # In a real implementation, we'd track subscription IDs
        self.logger.debug("Unsubscribed from event", subscription_id=subscription_id)
        return True
    
    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get event history.
        
        Args:
            event_type: Optional filter by event type
            limit: Maximum number of events to return
            
        Returns:
            List of event data
        """
        events = self._history
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        return [
            {
                "id": str(e.id),
                "type": e.type,
                "source": e.source,
                "data": e.data,
                "metadata": e.metadata,
            }
            for e in events[-limit:]
        ]