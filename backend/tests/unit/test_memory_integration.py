"""
Integration Tests for Agent-Memory Integration

Tests cover:
- SupervisorAgent with MemoryManager
- PlannerAgent with MemoryManager
- ExecutorAgent with MemoryManager
- QAAgent with MemoryManager
"""

import pytest
from uuid import uuid4

from app.memory.manager import MemoryManager
from app.agents.specialized.supervisor import SupervisorAgent
from app.agents.specialized.planner import PlannerAgent
from app.agents.specialized.executor import ExecutorAgent
from app.agents.specialized.qa import QAAgent
from app.models.memory import MemoryType


@pytest.fixture
def memory_manager() -> MemoryManager:
    return MemoryManager()


# ── SupervisorAgent Integration ─────────────────────────────────


class TestSupervisorWithMemory:
    @pytest.mark.asyncio
    async def test_supervisor_with_memory_manager(self, memory_manager: MemoryManager) -> None:
        agent = SupervisorAgent(memory_manager=memory_manager)
        result = await agent.process({"task": "research AI trends"})
        assert result["success"] is True

        summary = await memory_manager.summarize_memory()
        assert summary["episodic"] >= 1

    @pytest.mark.asyncio
    async def test_supervisor_stores_episode(self, memory_manager: MemoryManager) -> None:
        agent = SupervisorAgent(memory_manager=memory_manager)
        await agent.process({"task": "write sorting algorithm", "agents": ["coder"]})

        episodes = await memory_manager.episodic.get_recent_episodes(limit=5)
        assert len(episodes) >= 1
        assert "sorting" in episodes[0].task_description.lower()

    @pytest.mark.asyncio
    async def test_supervisor_retrieves_past_context(self, memory_manager: MemoryManager) -> None:
        agent1 = SupervisorAgent(memory_manager=memory_manager)
        await agent1.process({"task": "security scan of web application"})

        agent2 = SupervisorAgent(memory_manager=memory_manager)
        result = await agent2.process({"task": "security scan of web application"})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_supervisor_without_memory_still_works(self) -> None:
        agent = SupervisorAgent(memory_manager=None)
        result = await agent.process({"task": "research topic"})
        assert result["success"] is True


# ── PlannerAgent Integration ────────────────────────────────────


class TestPlannerWithMemory:
    @pytest.mark.asyncio
    async def test_planner_with_memory_manager(self, memory_manager: MemoryManager) -> None:
        agent = PlannerAgent(memory_manager=memory_manager)
        result = await agent.process({"goal": "Build a REST API"})
        assert result["success"] is True
        assert len(result["plan"]) >= 1

    @pytest.mark.asyncio
    async def test_planner_enriches_context(self, memory_manager: MemoryManager) -> None:
        await memory_manager.store_memory(
            content="Previous task: Build REST API with Python FastAPI",
            memory_type=MemoryType.EPISODIC,
        )

        agent = PlannerAgent(memory_manager=memory_manager)
        result = await agent.process({"goal": "Build a REST API"})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_planner_without_memory_still_works(self) -> None:
        agent = PlannerAgent(memory_manager=None)
        result = await agent.process({"goal": "Deploy to cloud"})
        assert result["success"] is True


# ── ExecutorAgent Integration ───────────────────────────────────


class TestExecutorWithMemory:
    @pytest.mark.asyncio
    async def test_executor_with_memory_manager(self, memory_manager: MemoryManager) -> None:
        from app.models.state import AgentState, PlanStep

        state = AgentState()
        state.goal = "Execute task"
        state.plan = [
            PlanStep(index=0, description="Do something", rationale="needed"),
        ]

        agent = ExecutorAgent(memory_manager=memory_manager)
        result = await agent.process({"state": state})
        assert result["success"] is True

        summary = await memory_manager.summarize_memory()
        assert summary["short_term"] >= 1

    @pytest.mark.asyncio
    async def test_executor_stores_success_knowledge(self, memory_manager: MemoryManager) -> None:
        from app.models.state import AgentState, PlanStep

        state = AgentState()
        state.goal = "Simple reasoning task"
        state.plan = [
            PlanStep(index=0, description="Think", rationale="reasoning"),
        ]

        agent = ExecutorAgent(memory_manager=memory_manager)
        await agent.process({"state": state})

        results = await memory_manager.search_memory(
            query="Simple reasoning task",
            memory_type=MemoryType.LONG_TERM,
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_executor_without_memory_still_works(self) -> None:
        from app.models.state import AgentState, PlanStep

        state = AgentState()
        state.goal = "Task"
        state.plan = [PlanStep(index=0, description="Step", rationale="r")]

        agent = ExecutorAgent(memory_manager=None)
        result = await agent.process({"state": state})
        assert result["success"] is True


# ── QAAgent Integration ─────────────────────────────────────────


class TestQAWithMemory:
    @pytest.mark.asyncio
    async def test_qa_with_memory_manager(self, memory_manager: MemoryManager) -> None:
        agent = QAAgent(memory_manager=memory_manager)
        result = await agent.process({"task": "Verify login works"})
        assert result["success"] is True
        assert result["verdict"] in ("pass", "partial", "fail", "no_tests")

        summary = await memory_manager.summarize_memory()
        assert summary["long_term"] >= 1

    @pytest.mark.asyncio
    async def test_qa_stores_feedback(self, memory_manager: MemoryManager) -> None:
        agent = QAAgent(memory_manager=memory_manager)
        await agent.process({
            "task": "Test API endpoint",
            "expected_output": "200 OK",
            "actual_output": "200 OK",
        })

        results = await memory_manager.search_memory(
            query="QA verdict",
            memory_type=MemoryType.LONG_TERM,
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_qa_without_memory_still_works(self) -> None:
        agent = QAAgent(memory_manager=None)
        result = await agent.process({"task": "Run tests"})
        assert result["success"] is True
