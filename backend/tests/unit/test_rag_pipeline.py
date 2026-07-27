"""
Unit Tests for RAGPipeline

Tests cover:
- Retrieve context
- Augment prompt
- Process query
- Empty knowledge base
- Custom prompt template
"""

import pytest

from app.rag.pipeline import RAGPipeline
from app.rag.knowledge_base import KnowledgeBaseManager
from app.memory.long_term import LongTermMemory
from app.rag.embeddings import EmbeddingProvider
from app.rag.text_splitter import TextSplitter


@pytest.fixture
def rag_pipeline() -> RAGPipeline:
    kb = KnowledgeBaseManager(
        long_term=LongTermMemory(collection_name="test_rag"),
        embedding_provider=EmbeddingProvider(dimension=64),
        text_splitter=TextSplitter(chunk_size=200, chunk_overlap=50),
    )
    return RAGPipeline(
        knowledge_base=kb,
        context_limit=3,
        min_score=0.1,
    )


class TestRAGPipelineInit:
    def test_creates_with_defaults(self) -> None:
        pipeline = RAGPipeline()
        assert pipeline.context_limit == 3
        assert pipeline.min_score == 0.3
        assert "{context}" in pipeline.prompt_template
        assert "{query}" in pipeline.prompt_template

    def test_creates_with_custom_template(self) -> None:
        template = "Ctx: {context}\nQ: {query}\nA:"
        pipeline = RAGPipeline(prompt_template=template)
        assert pipeline.prompt_template == template


class TestRAGPipelineRetrieve:
    @pytest.mark.asyncio
    async def test_retrieve_empty(self, rag_pipeline: RAGPipeline) -> None:
        result = await rag_pipeline.retrieve("test query")
        assert result["query"] == "test query"
        assert result["context_text"] == ""
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_retrieve_with_data(self, rag_pipeline: RAGPipeline, tmp_path) -> None:
        txt_file = tmp_path / "info.txt"
        txt_file.write_text("The Eiffel Tower is located in Paris, France.")
        await rag_pipeline.knowledge_base.add_document(str(txt_file))
        result = await rag_pipeline.retrieve("Where is the Eiffel Tower?")
        assert result["query"] == "Where is the Eiffel Tower?"
        assert isinstance(result["results"], list)


class TestRAGPipelineAugmentPrompt:
    @pytest.mark.asyncio
    async def test_augment_empty(self, rag_pipeline: RAGPipeline) -> None:
        result = await rag_pipeline.augment_prompt("question?")
        assert result["augmented_prompt"] == "question?"
        assert result["context_text"] == ""

    @pytest.mark.asyncio
    async def test_augment_with_data(self, rag_pipeline: RAGPipeline, tmp_path) -> None:
        txt_file = tmp_path / "aug.txt"
        txt_file.write_text("Python was created by Guido van Rossum.")
        await rag_pipeline.knowledge_base.add_document(str(txt_file))
        result = await rag_pipeline.augment_prompt("Who created Python?")
        assert "augmented_prompt" in result
        assert isinstance(result["augmented_prompt"], str)


class TestRAGPipelineProcessQuery:
    @pytest.mark.asyncio
    async def test_process_query_empty(self, rag_pipeline: RAGPipeline) -> None:
        result = await rag_pipeline.process_query("hello")
        assert result["query"] == "hello"
        assert result["retrieved_context"] == ""
        assert result["extra_context"] == ""
        assert result["combined_context"] == ""

    @pytest.mark.asyncio
    async def test_process_query_with_extra_context(self, rag_pipeline: RAGPipeline) -> None:
        result = await rag_pipeline.process_query(
            "hello",
            extra_context="Additional info",
        )
        assert result["extra_context"] == "Additional info"
        assert result["combined_context"] == "Additional info"

    @pytest.mark.asyncio
    async def test_process_query_with_retrieved_and_extra(
        self, rag_pipeline: RAGPipeline, tmp_path,
    ) -> None:
        txt_file = tmp_path / "both.txt"
        txt_file.write_text("Retrieved information about AI.")
        await rag_pipeline.knowledge_base.add_document(str(txt_file))
        result = await rag_pipeline.process_query(
            "Tell me about AI",
            extra_context="User preference: concise",
        )
        assert result["extra_context"] == "User preference: concise"
        assert isinstance(result["combined_context"], str)


class TestRAGPipelineFormatContext:
    def test_format_empty(self, rag_pipeline: RAGPipeline) -> None:
        ctx = rag_pipeline._format_context([])
        assert ctx == ""

    def test_format_with_results(self, rag_pipeline: RAGPipeline) -> None:
        results = [
            {"content": "chunk one", "source_file": "a.txt"},
            {"content": "chunk two", "source_file": "b.txt"},
        ]
        ctx = rag_pipeline._format_context(results)
        assert "chunk one" in ctx
        assert "chunk two" in ctx
        assert "a.txt" in ctx
