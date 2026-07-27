"""
Unit Tests for SecurityAgent

Tests cover:
- Agent creation and configuration
- Process with valid task
- Process with missing task
- Vulnerability analysis
- Risk level assessment
- Recommendation generation
- Reflect method
"""

import pytest

from app.agents.base import AgentConfig, BaseAgent
from app.agents.specialized.security import SecurityAgent
from app.models.agent import AgentStatus, AgentType


# ── Agent Init Tests ──────────────────────────────────────────


class TestSecurityAgentInit:
    def test_creates_with_defaults(self) -> None:
        agent = SecurityAgent()
        assert agent.config.name == "SecurityAgent"
        assert agent.config.type == AgentType.SECURITY
        assert agent.config.description != ""

    def test_creates_with_custom_config(self) -> None:
        config = AgentConfig(name="CustomSecurity", type=AgentType.SECURITY)
        agent = SecurityAgent(config=config)
        assert agent.config.name == "CustomSecurity"

    def test_is_base_agent(self) -> None:
        agent = SecurityAgent()
        assert isinstance(agent, BaseAgent)


# ── Risk Assessment Tests ─────────────────────────────────────


class TestRiskAssessment:
    def test_critical_risk(self) -> None:
        vulns = [{"severity": "critical"}]
        assert SecurityAgent._assess_risk_level(vulns) == "critical"

    def test_high_risk(self) -> None:
        vulns = [{"severity": "high"}]
        assert SecurityAgent._assess_risk_level(vulns) == "high"

    def test_medium_risk(self) -> None:
        vulns = [{"severity": "medium"}]
        assert SecurityAgent._assess_risk_level(vulns) == "medium"

    def test_low_risk(self) -> None:
        vulns = [{"severity": "low"}]
        assert SecurityAgent._assess_risk_level(vulns) == "low"

    def test_info_risk(self) -> None:
        vulns = [{"severity": "info"}]
        assert SecurityAgent._assess_risk_level(vulns) == "info"

    def test_mixed_severity(self) -> None:
        vulns = [{"severity": "low"}, {"severity": "high"}]
        assert SecurityAgent._assess_risk_level(vulns) == "high"

    def test_empty_vulnerabilities(self) -> None:
        assert SecurityAgent._assess_risk_level([]) == "info"


# ── Recommendation Tests ──────────────────────────────────────


class TestRecommendations:
    def test_basic_recommendations(self) -> None:
        recs = SecurityAgent._generate_recommendations([], "vulnerability")
        assert len(recs) >= 1

    def test_high_severity_urgent(self) -> None:
        vulns = [{"severity": "high"}]
        recs = SecurityAgent._generate_recommendations(vulns, "vulnerability")
        urgent = [r for r in recs if "URGENT" in r]
        assert len(urgent) == 1


# ── Process Tests ─────────────────────────────────────────────


class TestSecurityAgentProcess:
    @pytest.mark.asyncio
    async def test_no_task_returns_error(self) -> None:
        agent = SecurityAgent()
        result = await agent.process({})
        assert result["success"] is False
        assert "No task provided" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_basic_task(self) -> None:
        agent = SecurityAgent()
        result = await agent.process({"task": "scan web application for vulnerabilities"})
        assert result["success"] is True
        assert len(result["vulnerabilities"]) >= 1
        assert len(result["recommendations"]) >= 1
        assert result["risk_level"] in ("info", "low", "medium", "high", "critical")

    @pytest.mark.asyncio
    async def test_with_target(self) -> None:
        agent = SecurityAgent()
        result = await agent.process({
            "task": "vulnerability scan",
            "target": "https://example.com",
            "scan_type": "vulnerability",
        })
        assert result["success"] is True
        assert len(result["vulnerabilities"]) >= 2

    @pytest.mark.asyncio
    async def test_returns_to_idle(self) -> None:
        agent = SecurityAgent()
        await agent.process({"task": "security check"})
        assert agent.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_stores_memory(self) -> None:
        agent = SecurityAgent()
        await agent.process({"task": "audit system"})
        assert agent.get_memory("last_task") == "audit system"
        assert agent.get_memory("last_risk_level") is not None


# ── Reflect Tests ─────────────────────────────────────────────


class TestSecurityAgentReflect:
    @pytest.mark.asyncio
    async def test_reflect_on_success(self) -> None:
        agent = SecurityAgent()
        result = {
            "success": True,
            "vulnerabilities": [{"severity": "high"}, {"severity": "low"}],
            "recommendations": ["fix it"],
            "risk_level": "high",
            "errors": [],
        }
        insights = await agent.reflect(result)
        assert insights["total_vulnerabilities"] == 2
        assert insights["risk_level"] == "high"
        assert insights["recommendation_count"] == 1

    @pytest.mark.asyncio
    async def test_reflect_on_failure(self) -> None:
        agent = SecurityAgent()
        result = {
            "success": False,
            "vulnerabilities": [],
            "recommendations": [],
            "risk_level": "unknown",
            "errors": ["failed"],
        }
        insights = await agent.reflect(result)
        assert insights["had_errors"] is True
