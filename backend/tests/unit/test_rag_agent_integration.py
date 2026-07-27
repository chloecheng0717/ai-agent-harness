"""
Unit Tests for RAG Agent Integration

Tests that ResearchAgent, SecurityAgent, and SupervisorAgent
correctly use the RAG pipeline when provided.
"""

import pytest

from app.agents.specialized.research import ResearchAgent
from app.agents.specialized.security import SecurityAgent
from app.agents.specialized.supervisor import SupervisorAgent
from app.rag.pipeline import RAGPipeline
from app.rag.knowledge_base import KnowledgeBaseManager
from app.memory.long_term import LongTermMemory
from app.rag.embeddings import EmbeddingProvider
from app.rag.text_splitter import TextSplitter


@pytest.fixture
def rag_pipeline() -> RAGPipeline:
    kb = KnowledgeBaseManager(
        long_term=LongTermMemory(collection_name="test_agent_rag"),
        embedding_provider=EmbeddingProvider(dimension=64),
        text_splitter=TextSplitter(chunk_size=200, chunk_overlap=50),
    )
    return RAGPipeline(knowledge_base=kb, context_limit=3, min_score=0.1)


class TestResearchAgentRAG:
    def test_creates_without_rag(self) -> None:
        agent = ResearchAgent()
        assert agent.rag_pipeline is None

    def test_creates_with_rag(self, rag_pipeline: RAGPipeline) -> None:
        agent = ResearchAgent(rag_pipeline=rag_pipeline)
        assert agent.rag_pipeline is rag_pipeline

    @pytest.mark.asyncio
    async def test_process_without_rag(self) -> None:
        agent = ResearchAgent()
        result = await agent.process({"task": "research AI"})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_process_with_rag_empty_kb(self, rag_pipeline: RAGPipeline) -> None:
        agent = ResearchAgent(rag_pipeline=rag_pipeline)
        result = await agent.process({"task": "research AI"})
        assert result["success"] is True
        assert len(result["findings"]) >= 1


class TestSecurityAgentRAG:
    def test_creates_without_rag(self) -> None:
        agent = SecurityAgent()
        assert agent.rag_pipeline is None

    def test_creates_with_rag(self, rag_pipeline: RAGPipeline) -> None:
        agent = SecurityAgent(rag_pipeline=rag_pipeline)
        assert agent.rag_pipeline is rag_pipeline

    @pytest.mark.asyncio
    async def test_process_without_rag(self) -> None:
        agent = SecurityAgent()
        result = await agent.process({
            "task": "scan for vulnerabilities",
            "target": "localhost",
        })
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_process_with_rag_empty_kb(self, rag_pipeline: RAGPipeline) -> None:
        agent = SecurityAgent(rag_pipeline=rag_pipeline)
        result = await agent.process({
            "task": "scan for vulnerabilities",
            "target": "localhost",
        })
        assert result["success"] is True
        assert len(result["vulnerabilities"]) >= 1


class TestSupervisorAgentRAG:
    def test_creates_without_rag(self) -> None:
        agent = SupervisorAgent()
        assert agent.rag_pipeline is None

    def test_creates_with_rag(self, rag_pipeline: RAGPipeline) -> None:
        agent = SupervisorAgent(rag_pipeline=rag_pipeline)
        assert agent.rag_pipeline is rag_pipeline

    @pytest.mark.asyncio
    async def test_process_without_rag(self) -> None:
        agent = SupervisorAgent()
        result = await agent.process({"task": "analyse security"})
        assert "success" in result

    @pytest.mark.asyncio
    async def test_process_with_rag_empty_kb(self, rag_pipeline: RAGPipeline) -> None:
        agent = SupervisorAgent(rag_pipeline=rag_pipeline)
        result = await agent.process({"task": "analyse security"})
        assert "success" in result
