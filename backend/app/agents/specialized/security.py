"""
Security Agent

Specialised agent responsible for vulnerability analysis, security
review, and risk assessment.  It receives security tasks from the
Supervisor, analyses the target, and returns structured findings.

Responsibilities
----------------
* Vulnerability analysis of code, configurations, or systems.
* Security review with actionable recommendations.
* Risk assessment with severity classification.
"""

from typing import Any, Dict, List, Optional

import structlog

from app.agents.base import AgentConfig, BaseAgent
from app.models.agent import AgentStatus, AgentType


class SecurityAgent(BaseAgent):
    """
    Agent that performs security analysis, vulnerability scanning,
    and risk assessment.

    Parameters
    ----------
    config : AgentConfig | None
        Agent configuration.  ``type`` defaults to ``SECURITY``.
    """

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        if config is None:
            config = AgentConfig(
                name="SecurityAgent",
                type=AgentType.SECURITY,
                description="Performs vulnerability analysis and security review.",
            )
        super().__init__(config)
        self.logger = self.logger.bind(component="security_agent")

    # ── BaseAgent interface ───────────────────────────────────

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a security analysis task.

        Parameters
        ----------
        input_data : dict
            Must contain ``"task"`` (str).  May also contain:
            - ``"target"`` (str) — the target to analyse (URL, code, config).
            - ``"context"`` (str) — additional context.
            - ``"scan_type"`` (str) — type of scan (vulnerability, compliance, etc.).

        Returns
        -------
        dict
            ``{"success": bool, "vulnerabilities": list[dict],
              "recommendations": list[str], "risk_level": str,
              "errors": list[str]}``
        """
        task: str = input_data.get("task", "")
        target: str = input_data.get("target", "")
        context: str = input_data.get("context", "")
        scan_type: str = input_data.get("scan_type", "vulnerability")

        if not task:
            return {
                "success": False,
                "vulnerabilities": [],
                "recommendations": [],
                "risk_level": "unknown",
                "errors": ["No task provided."],
            }

        self.status = AgentStatus.PROCESSING
        errors: List[str] = []
        vulnerabilities: List[Dict[str, Any]] = []
        recommendations: List[str] = []
        risk_level: str = "low"

        try:
            vulnerabilities = self._analyse_vulnerabilities(task, target, context, scan_type)
            recommendations = self._generate_recommendations(vulnerabilities, scan_type)
            risk_level = self._assess_risk_level(vulnerabilities)

            self.update_memory("last_task", task)
            self.update_memory("last_vulnerabilities", vulnerabilities)
            self.update_memory("last_risk_level", risk_level)

            self.logger.info(
                "Security analysis completed",
                task=task,
                vulnerability_count=len(vulnerabilities),
                risk_level=risk_level,
            )
        except Exception as exc:
            errors.append(str(exc))
            self.logger.error("Security analysis failed", error=str(exc))
        finally:
            self.status = AgentStatus.IDLE

        return {
            "success": len(errors) == 0,
            "vulnerabilities": vulnerabilities,
            "recommendations": recommendations,
            "risk_level": risk_level,
            "errors": errors,
        }

    async def reflect(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on the security analysis result.

        Parameters
        ----------
        result : dict
            The result dict returned by :meth:`process`.

        Returns
        -------
        dict
            Reflection insights.
        """
        vulns = result.get("vulnerabilities", [])
        errors = result.get("errors", [])
        insights: Dict[str, Any] = {
            "total_vulnerabilities": len(vulns),
            "risk_level": result.get("risk_level", "unknown"),
            "had_errors": len(errors) > 0,
            "overall_success": result.get("success", False),
            "recommendation_count": len(result.get("recommendations", [])),
        }

        severities = [v.get("severity", "unknown") for v in vulns]
        if severities:
            insights["severity_distribution"] = {
                s: severities.count(s) for s in set(severities)
            }

        self.logger.info("Security reflection", **insights)
        self.update_memory("last_reflection", insights)
        return insights

    # ── Internal helpers ──────────────────────────────────────

    @staticmethod
    def _analyse_vulnerabilities(
        task: str,
        target: str,
        context: str,
        scan_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Analyse the target for vulnerabilities.

        This is a template-based implementation.  In a production system
        this would invoke security scanning tools via the ToolRegistry.
        """
        vulnerabilities: List[Dict[str, Any]] = []

        vulnerabilities.append({
            "id": "VULN-001",
            "title": f"Security analysis of: {task}",
            "description": f"Performed {scan_type} scan on target: {target or 'N/A'}",
            "severity": "info",
            "category": "analysis",
            "target": target,
        })

        if target:
            vulnerabilities.append({
                "id": "VULN-002",
                "title": "Target-specific finding",
                "description": f"Reviewed target configuration: {target}",
                "severity": "low",
                "category": "configuration",
                "target": target,
            })

        return vulnerabilities

    @staticmethod
    def _generate_recommendations(
        vulnerabilities: List[Dict[str, Any]],
        scan_type: str,
    ) -> List[str]:
        """Generate recommendations based on vulnerabilities found."""
        recommendations: List[str] = []

        recommendations.append(
            f"Completed {scan_type} scan. Review all findings for context-specific risk."
        )

        high_severity = [
            v for v in vulnerabilities
            if v.get("severity") in ("high", "critical")
        ]
        if high_severity:
            recommendations.append(
                f"URGENT: {len(high_severity)} high/critical severity finding(s) "
                "require immediate attention."
            )

        recommendations.append("Apply principle of least privilege where applicable.")
        recommendations.append("Keep all dependencies up to date.")

        return recommendations

    @staticmethod
    def _assess_risk_level(vulnerabilities: List[Dict[str, Any]]) -> str:
        """Assess overall risk level based on vulnerabilities."""
        severities = [v.get("severity", "low") for v in vulnerabilities]
        if "critical" in severities:
            return "critical"
        if "high" in severities:
            return "high"
        if "medium" in severities:
            return "medium"
        if "low" in severities:
            return "low"
        return "info"
