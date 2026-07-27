"""
Unit Tests for TextSplitter

Tests cover:
- Basic chunking
- Overlap handling
- Empty text
- Short text (smaller than chunk size)
- Natural split points (paragraphs, sentences)
- Invalid parameters
"""

import pytest

from app.rag.text_splitter import TextSplitter, TextChunk


class TestTextSplitterInit:
    def test_creates_with_defaults(self) -> None:
        splitter = TextSplitter()
        assert splitter.chunk_size == 1000
        assert splitter.chunk_overlap == 200

    def test_creates_with_custom_params(self) -> None:
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 50

    def test_invalid_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextSplitter(chunk_size=0)

    def test_negative_overlap(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            TextSplitter(chunk_overlap=-1)

    def test_overlap_ge_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            TextSplitter(chunk_size=100, chunk_overlap=100)


class TestTextSplitterSplit:
    def test_empty_text(self) -> None:
        splitter = TextSplitter()
        chunks = splitter.split_text("")
        assert chunks == []

    def test_whitespace_only(self) -> None:
        splitter = TextSplitter()
        chunks = splitter.split_text("   \n\n  ")
        assert chunks == []

    def test_short_text_single_chunk(self) -> None:
        splitter = TextSplitter(chunk_size=1000)
        chunks = splitter.split_text("Hello world")
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world"
        assert chunks[0].chunk_index == 0

    def test_long_text_multiple_chunks(self) -> None:
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "word " * 50  # 250 chars
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_indices_are_sequential(self) -> None:
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "sentence one. sentence two. sentence three. " * 10
        chunks = splitter.split_text(text)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_source_file_propagated(self) -> None:
        splitter = TextSplitter(chunk_size=1000)
        chunks = splitter.split_text("content", source_file="/path/to/file.txt")
        assert chunks[0].source_file == "/path/to/file.txt"

    def test_metadata_propagated(self) -> None:
        splitter = TextSplitter(chunk_size=1000)
        meta = {"doc_id": "abc"}
        chunks = splitter.split_text("content", metadata=meta)
        assert chunks[0].metadata == {"doc_id": "abc"}

    def test_prefers_paragraph_split(self) -> None:
        splitter = TextSplitter(chunk_size=40, chunk_overlap=5)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2

    def test_start_and_end_chars(self) -> None:
        splitter = TextSplitter(chunk_size=20, chunk_overlap=5)
        text = "abcdefghij" * 5  # 50 chars
        chunks = splitter.split_text(text)
        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char
