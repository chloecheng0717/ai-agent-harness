"""
Reasoning Loop Module

This module implements the ReAct (Reason + Act) pattern for agent reasoning.
It provides the core reasoning loop that agents use to process tasks.

Includes built-in flag detection that terminates the loop when a CTF flag
pattern is found in tool output or agent reasoning.
"""

import re
from typing import Any, Dict, List, Optional

import structlog

from app.config import get_settings

# Default patterns for CTF flag detection
DEFAULT_FLAG_PATTERNS: List[str] = [
    r"flag\{[^\}]+\}",
    r"CTF\{[^\}]+\}",
]


class ReasoningStep:
    """
    A single step in the reasoning loop.
    
    Attributes:
        thought: Agent's reasoning about the current state
        action: Action to take
        action_input: Input for the action
        observation: Result of the action
    """
    
    def __init__(
        self,
        thought: str,
        action: str,
        action_input: Dict[str, Any],
        observation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a reasoning step.
        
        Args:
            thought: Agent's reasoning
            action: Action to take
            action_input: Action input
            observation: Action result
        """
        self.thought = thought
        self.action = action
        self.action_input = action_input
        self.observation = observation


class FlagDetector:
    """
    Scans text for CTF flag patterns.

    Attributes:
        patterns: Compiled regex patterns to match.
    """

    def __init__(
        self,
        flag_patterns: Optional[List[str]] = None,
    ) -> None:
        """
        Initialise the detector.

        Parameters
        ----------
        flag_patterns : list[str] | None
            Regex strings.  Defaults to ``DEFAULT_FLAG_PATTERNS``.
        """
        raw = flag_patterns if flag_patterns else DEFAULT_FLAG_PATTERNS
        self.patterns = [re.compile(p, re.IGNORECASE) for p in raw]
        self.logger = structlog.get_logger().bind(component="flag_detector")

    def scan(self, text: str) -> Optional[str]:
        """
        Return the first flag match found in *text*, or ``None``.
        """
        if not text:
            return None
        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                self.logger.info(
                    "Flag detected",
                    flag=match.group(0),
                )
                return match.group(0)
        return None

    def scan_dict(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Recursively scan a dictionary (or list of dicts) for flag patterns.

        Handles nested dicts, lists, and string values.
        """
        if isinstance(data, str):
            return self.scan(data)
        if isinstance(data, dict):
            for value in data.values():
                result = self.scan_dict(value)
                if result:
                    return result
        if isinstance(data, list):
            for item in data:
                result = self.scan_dict(item)
                if result:
                    return result
        return None


class ReasoningLoop:
    """
    ReAct reasoning loop implementation.
    
    This class implements the Reason-Act-Observe pattern:
    1. Reason: Agent thinks about the current state
    2. Act: Agent decides on an action
    3. Observe: Agent observes the result
    
    The loop continues until:
    - The task is completed
    - Maximum iterations are reached
    - A CTF flag is detected in output (flag detection)
    - An error occurs
    
    Attributes:
        settings: Application settings
        logger: Structured logger
        max_iterations: Maximum number of iterations
        flag_detector: FlagDetector instance for flag detection
    """
    
    def __init__(
        self,
        max_iterations: Optional[int] = None,
        flag_patterns: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the reasoning loop.
        
        Args:
            max_iterations: Maximum iterations (uses config if not provided)
            flag_patterns: Custom regex patterns for flag detection.
                Defaults to ``DEFAULT_FLAG_PATTERNS``.
        """
        self.settings = get_settings()
        self.max_iterations = max_iterations or self.settings.max_iterations
        self.flag_detector = FlagDetector(flag_patterns)
        
        self.logger = structlog.get_logger().bind(component="reasoning_loop")
        self.logger.info(
            "Reasoning loop initialized",
            max_iterations=self.max_iterations,
        )
    
    async def run(
        self,
        initial_state: Dict[str, Any],
        reasoning_fn: Any,
        action_fn: Any,
    ) -> Dict[str, Any]:
        """
        Run the reasoning loop.
        
        This method executes the ReAct loop until completion or max iterations.
        
        Args:
            initial_state: Initial state for reasoning
            reasoning_fn: Async function for reasoning (thought generation)
            action_fn: Async function for action execution
            
        Returns:
            Dict containing final state and reasoning history
        """
        self.logger.debug("Starting reasoning loop")
        
        state = initial_state.copy()
        history: List[ReasoningStep] = []
        found_flag: Optional[str] = None
        
        for iteration in range(self.max_iterations):
            self.logger.debug(
                "Reasoning iteration",
                iteration=iteration + 1,
                max_iterations=self.max_iterations,
            )
            
            # Reason: Generate thought
            thought = await reasoning_fn(state)
            
            # Act: Decide and execute action
            action, action_input = await self._decide_action(thought, state)
            
            # Observe: Get action result
            observation = await action_fn(action, action_input)
            
            # Check observation for flags
            if isinstance(observation, dict):
                found_flag = self.flag_detector.scan_dict(observation)
            elif isinstance(observation, str):
                found_flag = self.flag_detector.scan(observation)
            
            # Record step
            step = ReasoningStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            )
            history.append(step)
            
            # Update state
            state = await self._update_state(state, step)
            
            # Terminate if flag found
            if found_flag:
                self.logger.info(
                    "Flag found in observation — terminating loop",
                    flag=found_flag,
                )
                state["found_flag"] = found_flag
                break
            
            # Check if task is complete
            if await self._is_complete(state):
                self.logger.debug("Task completed in reasoning loop")
                break
        
        return {
            "state": state,
            "history": [
                {
                    "thought": step.thought,
                    "action": step.action,
                    "action_input": step.action_input,
                    "observation": step.observation,
                }
                for step in history
            ],
            "iterations": len(history),
            "found_flag": found_flag,
        }
    
    async def _decide_action(
        self,
        thought: str,
        state: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """
        Decide on an action based on the current thought.
        
        Args:
            thought: Current reasoning thought
            state: Current state
            
        Returns:
            Tuple of (action_name, action_input)
        """
        # TODO: Implement action decision using LLM
        # This is a placeholder
        return "think", {"thought": thought}
    
    async def _update_state(
        self,
        state: Dict[str, Any],
        step: ReasoningStep,
    ) -> Dict[str, Any]:
        """
        Update state based on reasoning step.
        
        Args:
            state: Current state
            step: Reasoning step
            
        Returns:
            Updated state
        """
        # TODO: Implement state update logic
        return state
    
    async def _is_complete(self, state: Dict[str, Any]) -> bool:
        """
        Check if the task is complete.
        
        Args:
            state: Current state
            
        Returns:
            True if complete, False otherwise
        """
        # TODO: Implement completion check
        return False