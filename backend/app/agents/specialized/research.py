"""
Research Agent

Specialised agent responsible for information gathering, summarisation,
and knowledge analysis.  It receives research tasks from the Supervisor,
executes them (optionally via registered tools), and returns structured
findings.

Responsibilities
----------------
* Information gathering from provided context or tools.
* Summarisation of findings into concise, structured output.
* Knowledge analysis — identifying patterns, key facts, and gaps.
"""

from typing import Any, Dict, List, Optional

import structlog

from app.agents.base import AgentConfig, BaseAgent
from app.models.agent import AgentStatus, AgentType


class ResearchAgent(BaseAgent):
    """
    Agent that performs information gathering, summarisation, and analysis.

    Parameters
    ----------
    config : AgentConfig | None
        Agent configuration.  ``type`` defaults to ``RESEARCHER``.
    """

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        if config is None:
            config = AgentConfig(
                name="ResearchAgent",
                type=AgentType.RESEARCHER,
                description="Gathers, summarises, and analyses information.",
            )
        super().__init__(config)
        self.logger = self.logger.bind(component="research_agent")

    # ── BaseAgent interface ───────────────────────────────────

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a research task.

        Parameters
        ----------
        input_data : dict
            Must contain ``"task"`` (str).  May also contain:
            - ``"context"`` (str) — additional context for the research.
            - ``"sources"`` (list[str]) — data sources to consult.

        Returns
        -------
        dict
            ``{"success": bool, "findings": list[dict], "summary": str,
              "errors": list[str]}``
        """
        task: str = input_data.get("task", "")
        context: str = input_data.get("context", "")
        sources: List[str] = input_data.get("sources", [])

        if not task:
            return {
                "success": False,
                "findings": [],
                "summary": "",
                "errors": ["No task provided."],
            }

        self.status = AgentStatus.PROCESSING
        errors: List[str] = []
        findings: List[Dict[str, Any]] = []

        try:
            findings = self._gather_findings(task, context, sources)
            summary = self._summarise(task, findings)

            self.update_memory("last_task", task)
            self.update_memory("last_findings", findings)
            self.update_memory("last_summary", summary)

            self.logger.info(
                "Research completed",
                task=task,
                finding_count=len(findings),
            )
        except Exception as exc:
            errors.append(str(exc))
            self.logger.error("Research failed", error=str(exc))
        finally:
            self.status = AgentStatus.IDLE

        return {
            "success": len(errors) == 0,
            "findings": findings,
            "summary": summary if not errors else "",
            "errors": errors,
        }

    async def reflect(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on the research result.

        Parameters
        ----------
        result : dict
            The result dict returned by :meth:`process`.

        Returns
        -------
        dict
            Reflection insights.
        """
        findings = result.get("findings", [])
        errors = result.get("errors", [])
        insights: Dict[str, Any] = {
            "total_findings": len(findings),
            "had_errors": len(errors) > 0,
            "overall_success": result.get("success", False),
        }

        categories = [f.get("category", "unknown") for f in findings]
        if categories:
            insights["category_distribution"] = {
                c: categories.count(c) for c in set(categories)
            }

        self.logger.info("Research reflection", **insights)
        self.update_memory("last_reflection", insights)
        return insights

    # ── Internal helpers ──────────────────────────────────────

    @staticmethod
    def _gather_findings(
        task: str,
        context: str,
        sources: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Gather findings from available information.

        This is a template-based implementation that produces structured
        findings from the task description and context.  In a production
        system this would integrate with search engines, databases, or
        external APIs via the ToolRegistry.
        """
        findings: List[Dict[str, Any]] = []

        # Core finding from the task itself
        findings.append({
            "id": 1,
            "category": "task_analysis",
            "title": f"Analysis of: {task}",
            "content": f"Task identified: {task}",
            "confidence": 0.9,
            "source": "task_analysis",
        })

        # Context-based findings
        if context:
            findings.append({
                "id": 2,
                "category": "context",
                "title": "Contextual information",
                "content": context,
                "confidence": 0.8,
                "source": "provided_context",
            })

        # Source-based findings
        for idx, source in enumerate(sources, start=len(findings) + 1):
            findings.append({
                "id": idx,
                "category": "source",
                "title": f"Data from {source}",
                "content": f"Information gathered from source: {source}",
                "confidence": 0.7,
                "source": source,
            })

        return findings

    @staticmethod
    def _summarise(task: str, findings: List[Dict[str, Any]]) -> str:
        """Produce a concise summary of gathered findings."""
        if not findings:
            return f"No findings gathered for task: {task}"

        lines = [f"Research Summary for: {task}", ""]
        for finding in findings:
            lines.append(
                f"- [{finding.get('category', 'unknown')}] "
                f"{finding.get('title', 'Untitled')}: "
                f"{finding.get('content', '')}"
            )
        return "\n".join(lines)
