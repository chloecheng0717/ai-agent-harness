"""
RAG Pipeline

Orchestrates the full retrieval-augmented generation flow:

  User Query → Embedding → Vector Search → Context Retrieval → Prompt Augmentation

The pipeline does not call the LLM directly — it prepares an
augmented prompt that the caller (an agent) can feed to its LLM.
"""

from typing import Any, Dict, List, Optional

import structlog

from app.rag.knowledge_base import KnowledgeBaseManager

logger = structlog.get_logger().bind(component="rag_pipeline")


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Parameters
    ----------
    knowledge_base : KnowledgeBaseManager | None
        Knowledge base to retrieve from.  Created with defaults when ``None``.
    context_limit : int
        Default maximum number of context chunks to retrieve.
    min_score : float
        Default minimum similarity score threshold.
    prompt_template : str | None
        Custom template for the augmented prompt.  Must contain
        ``{context}`` and ``{query}`` placeholders.
    """

    _DEFAULT_TEMPLATE = (
        "Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    )

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBaseManager] = None,
        context_limit: int = 3,
        min_score: float = 0.3,
        prompt_template: Optional[str] = None,
    ) -> None:
        self.knowledge_base = knowledge_base or KnowledgeBaseManager()
        self.context_limit = context_limit
        self.min_score = min_score
        self.prompt_template = prompt_template or self._DEFAULT_TEMPLATE

    async def retrieve(
        self,
        query: str,
        context_limit: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query.

        Returns
        -------
        dict
            ``{"query", "context", "results", "context_text"}``
        """
        limit = context_limit or self.context_limit
        score = min_score or self.min_score

        results = await self.knowledge_base.search_knowledge(
            query=query,
            limit=limit,
            min_score=score,
        )

        context_text = self._format_context(results)

        return {
            "query": query,
            "context": context_text,
            "results": results,
            "context_text": context_text,
        }

    async def augment_prompt(
        self,
        query: str,
        context_limit: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve context and build an augmented prompt.

        Returns
        -------
        dict
            ``{"query", "augmented_prompt", "context_text", "results"}``
        """
        retrieval = await self.retrieve(query, context_limit, min_score)
        context_text = retrieval["context_text"]

        if context_text:
            augmented = self.prompt_template.format(
                context=context_text, query=query,
            )
        else:
            augmented = query

        return {
            "query": query,
            "augmented_prompt": augmented,
            "context_text": context_text,
            "results": retrieval["results"],
        }

    async def process_query(
        self,
        query: str,
        extra_context: str = "",
        context_limit: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: retrieve context, augment prompt, return all data
        needed by an agent to generate a response.

        Returns
        -------
        dict
            ``{"query", "augmented_prompt", "context_text", "retrieved_context",
              "extra_context", "results"}``
        """
        retrieval = await self.augment_prompt(query, context_limit, min_score)

        retrieved = retrieval["context_text"]
        combined_context_parts: List[str] = []
        if retrieved:
            combined_context_parts.append(retrieved)
        if extra_context:
            combined_context_parts.append(extra_context)

        combined_context = "\n\n".join(combined_context_parts)

        return {
            "query": query,
            "augmented_prompt": retrieval["augmented_prompt"],
            "context_text": retrieval["context_text"],
            "retrieved_context": retrieved,
            "extra_context": extra_context,
            "combined_context": combined_context,
            "results": retrieval["results"],
        }

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved results into a context string."""
        if not results:
            return ""

        parts: List[str] = []
        for i, r in enumerate(results, 1):
            source = r.get("source_file", "unknown")
            parts.append(f"[Source {i}: {source}]\n{r['content']}")
        return "\n\n".join(parts)
