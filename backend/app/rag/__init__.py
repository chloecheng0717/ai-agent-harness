"""
RAG (Retrieval-Augmented Generation) Package

Provides document ingestion, knowledge base management, and
RAG pipeline for enriching agent responses with retrieved context.
"""

from app.rag.document_loader import DocumentLoader, LoadedDocument
from app.rag.text_splitter import TextSplitter, TextChunk
from app.rag.embeddings import EmbeddingProvider
from app.rag.knowledge_base import KnowledgeBaseManager
from app.rag.pipeline import RAGPipeline

__all__ = [
    "DocumentLoader",
    "LoadedDocument",
    "TextSplitter",
    "TextChunk",
    "EmbeddingProvider",
    "KnowledgeBaseManager",
    "RAGPipeline",
]
