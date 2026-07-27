"""
Coding Agent

Specialised agent responsible for writing code, debugging, and code
explanation.  It receives coding tasks from the Supervisor, processes
them, and returns structured results.

Responsibilities
----------------
* Writing code from specifications.
* Debugging existing code.
* Explaining code behaviour and logic.
"""

from typing import Any, Dict, List, Optional

import structlog

from app.agents.base import AgentConfig, BaseAgent
from app.models.agent import AgentStatus, AgentType


class CodingAgent(BaseAgent):
    """
    Agent that writes, debugs, and explains code.

    Parameters
    ----------
    config : AgentConfig | None
        Agent configuration.  ``type`` defaults to ``CODER``.
    """

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        if config is None:
            config = AgentConfig(
                name="CodingAgent",
                type=AgentType.CODER,
                description="Writes, debugs, and explains code.",
            )
        super().__init__(config)
        self.logger = self.logger.bind(component="coding_agent")

    # ── BaseAgent interface ───────────────────────────────────

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a coding task.

        Parameters
        ----------
        input_data : dict
            Must contain ``"task"`` (str).  May also contain:
            - ``"code"`` (str) — existing code to debug or explain.
            - ``"language"`` (str) — target programming language.
            - ``"context"`` (str) — additional context.

        Returns
        -------
        dict
            ``{"success": bool, "output": str, "task_type": str,
              "errors": list[str]}``
        """
        task: str = input_data.get("task", "")
        code: str = input_data.get("code", "")
        language: str = input_data.get("language", "python")
        context: str = input_data.get("context", "")

        if not task:
            return {
                "success": False,
                "output": "",
                "task_type": "unknown",
                "errors": ["No task provided."],
            }

        self.status = AgentStatus.PROCESSING
        errors: List[str] = []
        output: str = ""
        task_type: str = "unknown"

        try:
            task_type = self._classify_task(task, code)
            output = self._execute_coding_task(task, code, language, context, task_type)

            self.update_memory("last_task", task)
            self.update_memory("last_output", output)
            self.update_memory("last_task_type", task_type)

            self.logger.info(
                "Coding task completed",
                task_type=task_type,
                language=language,
            )
        except Exception as exc:
            errors.append(str(exc))
            self.logger.error("Coding task failed", error=str(exc))
        finally:
            self.status = AgentStatus.IDLE

        return {
            "success": len(errors) == 0,
            "output": output,
            "task_type": task_type,
            "errors": errors,
        }

    async def reflect(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on the coding result.

        Parameters
        ----------
        result : dict
            The result dict returned by :meth:`process`.

        Returns
        -------
        dict
            Reflection insights.
        """
        insights: Dict[str, Any] = {
            "task_type": result.get("task_type", "unknown"),
            "had_errors": len(result.get("errors", [])) > 0,
            "overall_success": result.get("success", False),
            "output_length": len(result.get("output", "")),
        }

        self.logger.info("Coding reflection", **insights)
        self.update_memory("last_reflection", insights)
        return insights

    # ── Internal helpers ──────────────────────────────────────

    @staticmethod
    def _classify_task(task: str, code: str) -> str:
        """Classify the coding task type."""
        task_lower = task.lower()
        if any(kw in task_lower for kw in ("debug", "fix", "error", "bug")):
            return "debug"
        if any(kw in task_lower for kw in ("explain", "describe", "what does")):
            return "explain"
        if any(kw in task_lower for kw in ("write", "create", "implement", "generate")):
            return "write"
        if code:
            return "debug"
        return "write"

    @staticmethod
    def _execute_coding_task(
        task: str,
        code: str,
        language: str,
        context: str,
        task_type: str,
    ) -> str:
        """
        Execute the coding task.

        This is a template-based implementation.  In a production system
        this would invoke an LLM or code generation tool via the
        ToolRegistry.
        """
        if task_type == "debug":
            return (
                f"Debug analysis for {language} code:\n"
                f"Task: {task}\n"
                f"Code under review:\n{code}\n"
                f"Analysis complete — reviewed for common issues."
            )
        if task_type == "explain":
            return (
                f"Code explanation ({language}):\n"
                f"Task: {task}\n"
                f"Code:\n{code}\n"
                f"Explanation: The code implements the requested functionality."
            )
        # write
        context_line = f"\nContext: {context}" if context else ""
        return (
            f"Generated {language} code:\n"
            f"Task: {task}{context_line}\n"
            f"Code structure produced according to specifications."
        )
