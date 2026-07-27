"""
Knowledge Base Manager

Manages a document-level knowledge base backed by ChromaDB (or an
in-memory fallback).  Uses the existing LongTermMemory infrastructure
for vector storage and the DocumentLoader + TextSplitter for ingestion.
"""

import hashlib
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from app.memory.long_term import LongTermMemory
from app.models.memory import MemoryEntry, MemorySearchQuery, MemoryType
from app.rag.document_loader import DocumentLoader, LoadedDocument
from app.rag.embeddings import EmbeddingProvider
from app.rag.text_splitter import TextChunk, TextSplitter

logger = structlog.get_logger().bind(component="knowledge_base")


class DocumentRecord(BaseModel):
    """Metadata record for an ingested document."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    file_path: str
    file_name: str
    file_type: str
    chunk_count: int = 0
    content_hash: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseManager:
    """
    High-level manager for the RAG knowledge base.

    Coordinates document loading, chunking, embedding, and storage
    in a LongTermMemory-backed vector collection.

    Parameters
    ----------
    long_term : LongTermMemory | None
        Existing long-term memory instance.  A new one is created when
        ``None``.
    collection_name : str
        ChromaDB collection name dedicated to the knowledge base.
    embedding_provider : EmbeddingProvider | None
        Provider for text embeddings.  Created with defaults when ``None``.
    text_splitter : TextSplitter | None
        Text chunker.  Created with defaults when ``None``.
    document_loader : DocumentLoader | None
        File loader.  Created with defaults when ``None``.
    """

    def __init__(
        self,
        long_term: Optional[LongTermMemory] = None,
        collection_name: str = "knowledge_base",
        embedding_provider: Optional[EmbeddingProvider] = None,
        text_splitter: Optional[TextSplitter] = None,
        document_loader: Optional[DocumentLoader] = None,
    ) -> None:
        self.collection_name = collection_name
        self.long_term = long_term or LongTermMemory(
            collection_name=collection_name,
        )
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        self.text_splitter = text_splitter or TextSplitter()
        self.document_loader = document_loader or DocumentLoader()
        self._documents: Dict[str, DocumentRecord] = {}

    # ── Public API ────────────────────────────────────────────

    async def add_document(self, file_path: str) -> DocumentRecord:
        """
        Load, chunk, embed, and store a document.

        Returns the document record with chunk count and content hash.
        """
        loaded = self.document_loader.load_file(file_path)
        return await add_loaded_document(self, loaded)

    async def add_directory(
        self,
        dir_path: str,
        recursive: bool = True,
    ) -> List[DocumentRecord]:
        """Load all supported files from a directory."""
        loaded_docs = self.document_loader.load_directory(
            dir_path, recursive=recursive,
        )
        records: List[DocumentRecord] = []
        for doc in loaded_docs:
            record = await add_loaded_document(self, doc)
            records.append(record)
        return records

    async def remove_document(self, document_id: str) -> bool:
        """
        Remove all chunks belonging to a document from the knowledge base.

        Returns ``True`` if the document was found and removed.
        """
        record = self._documents.pop(document_id, None)
        if record is None:
            return False

        removed = 0
        for i in range(record.chunk_count):
            try:
                chunk_hex = f"{record.id[:8]}{i:04d}{'0' * 20}"
                chunk_id = UUID(hex=chunk_hex)
            except ValueError:
                break
            entry = await self.long_term.get(chunk_id)
            if entry is not None:
                await self.long_term.delete(entry.id)
                removed += 1

        logger.info(
            "Document removed",
            document_id=document_id,
            file_name=record.file_name,
            chunks_removed=removed,
        )
        return True

    async def search_knowledge(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for chunks relevant to the query.

        Returns a list of dicts with ``content``, ``score``, and
        ``metadata`` keys.
        """
        embedding = self.embedding_provider.embed(query)

        search_query = MemorySearchQuery(
            query=query,
            type=MemoryType.LONG_TERM,
            limit=limit,
            min_score=min_score,
        )

        results = await self.long_term.search(search_query)

        output: List[Dict[str, Any]] = []
        for result in results:
            chunk_meta = result.memory.metadata
            output.append({
                "content": result.memory.content,
                "score": result.score,
                "chunk_id": str(result.memory.id),
                "source_file": chunk_meta.get("source_file", ""),
                "chunk_index": chunk_meta.get("chunk_index", 0),
                "metadata": chunk_meta,
            })

        return output

    async def retrieve_context(
        self,
        query: str,
        limit: int = 3,
        min_score: float = 0.3,
    ) -> str:
        """
        Retrieve relevant context as a single concatenated string.

        Suitable for injecting into LLM prompts.
        """
        results = await self.search_knowledge(
            query=query, limit=limit, min_score=min_score,
        )
        if not results:
            return ""

        parts: List[str] = []
        for i, r in enumerate(results, 1):
            source = r.get("source_file", "unknown")
            parts.append(f"[Context {i} from {source}]\n{r['content']}")

        return "\n\n".join(parts)

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return metadata for all ingested documents."""
        return [doc.model_dump() for doc in self._documents.values()]

    async def health_check(self) -> bool:
        """Check if the underlying vector store is healthy."""
        return await self.long_term.health_check()


# ── Module-level helper to avoid circular references ──────────


async def add_loaded_document(
    kb: KnowledgeBaseManager,
    loaded: LoadedDocument,
) -> DocumentRecord:
    """Chunk, embed, and store a loaded document."""
    content_hash = hashlib.sha256(loaded.content.encode()).hexdigest()[:16]

    chunks = kb.text_splitter.split_text(
        loaded.content,
        source_file=loaded.file_path,
    )

    record = DocumentRecord(
        file_path=loaded.file_path,
        file_name=loaded.file_name,
        file_type=loaded.file_type,
        chunk_count=len(chunks),
        content_hash=content_hash,
        metadata=loaded.metadata,
    )

    for chunk in chunks:
        try:
            chunk_id = UUID(hex=f"{record.id[:8]}{chunk.chunk_index:04d}{'0' * 20}")
        except ValueError:
            chunk_id = uuid4()

        embedding = kb.embedding_provider.embed(chunk.content)

        entry = MemoryEntry(
            id=chunk_id,
            type=MemoryType.LONG_TERM,
            content=chunk.content,
            embedding=embedding,
            metadata={
                "document_id": record.id,
                "file_name": loaded.file_name,
                "file_path": loaded.file_path,
                "file_type": loaded.file_type,
                "chunk_index": chunk.chunk_index,
                "source_file": loaded.file_path,
                "content_hash": content_hash,
            },
        )
        await kb.long_term.add(entry)

    kb._documents[record.id] = record

    logger.info(
        "Document added to knowledge base",
        document_id=record.id,
        file_name=loaded.file_name,
        chunks=len(chunks),
    )
    return record
