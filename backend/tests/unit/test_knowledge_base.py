"""
Unit Tests for KnowledgeBaseManager

Tests cover:
- Adding documents
- Removing documents
- Searching knowledge
- Retrieving context
- Listing documents
- Health check
"""

import pytest

from app.rag.knowledge_base import KnowledgeBaseManager, DocumentRecord
from app.memory.long_term import LongTermMemory
from app.rag.embeddings import EmbeddingProvider
from app.rag.text_splitter import TextSplitter


@pytest.fixture
def knowledge_base() -> KnowledgeBaseManager:
    return KnowledgeBaseManager(
        long_term=LongTermMemory(collection_name="test_kb"),
        embedding_provider=EmbeddingProvider(dimension=64),
        text_splitter=TextSplitter(chunk_size=200, chunk_overlap=50),
    )


class TestKnowledgeBaseManagerInit:
    def test_creates_with_defaults(self) -> None:
        kb = KnowledgeBaseManager()
        assert kb.collection_name == "knowledge_base"
        assert kb.long_term is not None
        assert kb.embedding_provider is not None

    def test_creates_with_custom_params(self) -> None:
        lt = LongTermMemory(collection_name="custom")
        ep = EmbeddingProvider(dimension=128)
        ts = TextSplitter(chunk_size=500, chunk_overlap=50)
        kb = KnowledgeBaseManager(
            long_term=lt,
            collection_name="custom",
            embedding_provider=ep,
            text_splitter=ts,
        )
        assert kb.collection_name == "custom"
        assert kb.embedding_provider.dimension == 128
        assert kb.text_splitter.chunk_size == 500


class TestKnowledgeBaseManagerAddDocument:
    @pytest.mark.asyncio
    async def test_add_document(self, knowledge_base: KnowledgeBaseManager, tmp_path) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is test content for the knowledge base.")
        record = await knowledge_base.add_document(str(txt_file))
        assert isinstance(record, DocumentRecord)
        assert record.file_name == "test.txt"
        assert record.chunk_count >= 1

    @pytest.mark.asyncio
    async def test_add_document_stores_in_listed(self, knowledge_base: KnowledgeBaseManager, tmp_path) -> None:
        txt_file = tmp_path / "listed.txt"
        txt_file.write_text("Some content here.")
        record = await knowledge_base.add_document(str(txt_file))
        docs = knowledge_base.list_documents()
        assert any(d["id"] == record.id for d in docs)


class TestKnowledgeBaseManagerSearchKnowledge:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, knowledge_base: KnowledgeBaseManager, tmp_path) -> None:
        txt_file = tmp_path / "searchable.txt"
        txt_file.write_text("Python is a programming language used for data science.")
        await knowledge_base.add_document(str(txt_file))
        results = await knowledge_base.search_knowledge("Python programming")
        assert len(results) >= 1
        assert "content" in results[0]
        assert "score" in results[0]

    @pytest.mark.asyncio
    async def test_search_empty_kb(self, knowledge_base: KnowledgeBaseManager) -> None:
        results = await knowledge_base.search_knowledge("anything")
        assert results == []


class TestKnowledgeBaseManagerRetrieveContext:
    @pytest.mark.asyncio
    async def test_retrieve_context_with_data(self, knowledge_base: KnowledgeBaseManager, tmp_path) -> None:
        txt_file = tmp_path / "ctx.txt"
        txt_file.write_text("Machine learning is a subset of artificial intelligence.")
        await knowledge_base.add_document(str(txt_file))
        context = await knowledge_base.retrieve_context("machine learning")
        assert isinstance(context, str)

    @pytest.mark.asyncio
    async def test_retrieve_context_empty(self, knowledge_base: KnowledgeBaseManager) -> None:
        context = await knowledge_base.retrieve_context("no data")
        assert context == ""


class TestKnowledgeBaseManagerRemoveDocument:
    @pytest.mark.asyncio
    async def test_remove_document(self, knowledge_base: KnowledgeBaseManager, tmp_path) -> None:
        txt_file = tmp_path / "remove_me.txt"
        txt_file.write_text("This will be removed.")
        record = await knowledge_base.add_document(str(txt_file))
        removed = await knowledge_base.remove_document(record.id)
        assert removed is True
        docs = knowledge_base.list_documents()
        assert not any(d["id"] == record.id for d in docs)

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, knowledge_base: KnowledgeBaseManager) -> None:
        removed = await knowledge_base.remove_document("nonexistent")
        assert removed is False


class TestKnowledgeBaseManagerListDocuments:
    @pytest.mark.asyncio
    async def test_list_empty(self, knowledge_base: KnowledgeBaseManager) -> None:
        docs = knowledge_base.list_documents()
        assert docs == []

    @pytest.mark.asyncio
    async def test_list_multiple(self, knowledge_base: KnowledgeBaseManager, tmp_path) -> None:
        for name in ["a.txt", "b.txt", "c.txt"]:
            (tmp_path / name).write_text(f"Content of {name}")
        await knowledge_base.add_directory(str(tmp_path))
        docs = knowledge_base.list_documents()
        assert len(docs) == 3


class TestKnowledgeBaseManagerHealth:
    @pytest.mark.asyncio
    async def test_health_check(self, knowledge_base: KnowledgeBaseManager) -> None:
        healthy = await knowledge_base.health_check()
        assert healthy is True
