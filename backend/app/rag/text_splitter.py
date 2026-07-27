"""
Text Splitter

Splits documents into overlapping chunks suitable for embedding
and vector storage.
"""

from typing import List

from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """A chunk of text with position metadata."""

    content: str = Field(..., description="Text content of the chunk")
    chunk_index: int = Field(..., description="Index of this chunk in the document")
    start_char: int = Field(..., description="Start character offset in the original text")
    end_char: int = Field(..., description="End character offset in the original text")
    source_file: str = Field(default="", description="Source file path")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class TextSplitter:
    """
    Splits text into overlapping chunks.

    Parameters
    ----------
    chunk_size : int
        Target size of each chunk in characters.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(
        self,
        text: str,
        source_file: str = "",
        metadata: dict | None = None,
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks.

        Attempts to split at paragraph or sentence boundaries when possible.
        """
        if not text or not text.strip():
            return []

        meta = metadata or {}
        chunks: List[TextChunk] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            if end < len(text):
                end = self._find_split_point(text, start, end)

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        content=chunk_text,
                        chunk_index=chunk_index,
                        start_char=start,
                        end_char=end,
                        source_file=source_file,
                        metadata=meta,
                    )
                )
                chunk_index += 1

            if end >= len(text):
                break
            start = end - self.chunk_overlap
            if start <= 0:
                break

        return chunks

    def _find_split_point(self, text: str, start: int, end: int) -> int:
        """Find a natural split point near the end boundary."""
        search_start = max(start + self.chunk_size // 2, end - 200)

        for sep in ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]:
            last_idx = text.rfind(sep, search_start, end)
            if last_idx > search_start:
                return last_idx + len(sep)

        return end
