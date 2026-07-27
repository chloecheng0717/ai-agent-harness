"""
Agent Registry

Centralised registry for managing specialised agent classes.  The registry
supports registering, retrieving, listing, and validating agent types.
It is used by the :class:`SupervisorAgent` to resolve agent keys to
concrete classes.
"""

from typing import Any, Dict, List, Optional, Type

import structlog

from app.agents.base import BaseAgent


class AgentNotFoundError(Exception):
    """Raised when a requested agent is not in the registry."""


class DuplicateAgentError(Exception):
    """Raised when attempting to register an agent key that already exists."""


class AgentRegistry:
    """
    Registry that maps string keys to :class:`BaseAgent` subclasses.

    Example::

        registry = AgentRegistry()
        registry.register("researcher", ResearchAgent)
        agent_cls = registry.get("researcher")
        agent = agent_cls()

    Attributes
    ----------
    _agents : dict mapping agent key -> agent class
    """

    def __init__(self) -> None:
        self._agents: Dict[str, Type[BaseAgent]] = {}
        self.logger = structlog.get_logger().bind(component="agent_registry")
        self.logger.info("Agent registry initialised")

    # ── Core API ──────────────────────────────────────────────

    def register(
        self,
        key: str,
        agent_class: Type[BaseAgent],
        overwrite: bool = False,
    ) -> None:
        """
        Register an agent class under a string key.

        Parameters
        ----------
        key : str
            Unique key for the agent (e.g. ``"researcher"``).
        agent_class : type
            A :class:`BaseAgent` subclass.
        overwrite : bool
            If ``True``, silently replace an existing registration.

        Raises
        ------
        DuplicateAgentError
            If *key* is already registered and *overwrite* is ``False``.
        TypeError
            If *agent_class* is not a :class:`BaseAgent` subclass.
        """
        if not (isinstance(agent_class, type) and issubclass(agent_class, BaseAgent)):
            raise TypeError(
                f"agent_class must be a BaseAgent subclass, got {agent_class!r}"
            )

        if key in self._agents and not overwrite:
            raise DuplicateAgentError(
                f"Agent '{key}' is already registered. "
                "Use overwrite=True to replace it."
            )

        self._agents[key] = agent_class
        self.logger.info(
            "Agent registered",
            key=key,
            class_name=agent_class.__name__,
            overwritten=key in self._agents and overwrite,
        )

    def get(self, key: str) -> Optional[Type[BaseAgent]]:
        """
        Retrieve an agent class by key.

        Parameters
        ----------
        key : str
            The agent key.

        Returns
        -------
        type or None
            The agent class, or ``None`` if not found.
        """
        return self._agents.get(key)

    def get_or_raise(self, key: str) -> Type[BaseAgent]:
        """
        Retrieve an agent class by key, raising if not found.

        Parameters
        ----------
        key : str
            The agent key.

        Returns
        -------
        type
            The agent class.

        Raises
        ------
        AgentNotFoundError
            If the key is not registered.
        """
        cls = self._agents.get(key)
        if cls is None:
            raise AgentNotFoundError(f"Agent '{key}' is not registered")
        return cls

    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all registered agents.

        Returns
        -------
        list[dict]
            Each dict contains ``"key"`` and ``"class_name"``.
        """
        return [
            {"key": key, "class_name": cls.__name__}
            for key, cls in self._agents.items()
        ]

    def list_keys(self) -> List[str]:
        """
        Return all registered agent keys.

        Returns
        -------
        list[str]
            Sorted list of keys.
        """
        return sorted(self._agents.keys())

    def exists(self, key: str) -> bool:
        """
        Check whether an agent key is registered.

        Parameters
        ----------
        key : str
            The agent key.

        Returns
        -------
        bool
        """
        return key in self._agents

    def unregister(self, key: str) -> bool:
        """
        Remove an agent from the registry.

        Parameters
        ----------
        key : str
            The agent key to remove.

        Returns
        -------
        bool
            ``True`` if the agent was removed, ``False`` if not found.
        """
        if key in self._agents:
            del self._agents[key]
            self.logger.info("Agent unregistered", key=key)
            return True
        return False

    def clear(self) -> None:
        """Remove all registered agents."""
        self._agents.clear()
        self.logger.info("Agent registry cleared")

    def count(self) -> int:
        """Return the number of registered agents."""
        return len(self._agents)

    def validate_agent(self, key: str) -> bool:
        """
        Validate that an agent key is registered and its class is usable.

        Parameters
        ----------
        key : str
            The agent key.

        Returns
        -------
        bool
            ``True`` if the agent is registered and is a concrete class.
        """
        cls = self._agents.get(key)
        if cls is None:
            return False
        return isinstance(cls, type) and issubclass(cls, BaseAgent)
