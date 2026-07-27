"""
Unit Tests for DocumentLoader

Tests cover:
- Loading TXT files
- Loading Markdown files
- Loading code files
- Loading directories recursively
- Unsupported file types
- Missing files
- File size limits
"""

import os
import tempfile
from pathlib import Path

import pytest

from app.rag.document_loader import DocumentLoader, LoadedDocument, SUPPORTED_EXTENSIONS


class TestDocumentLoaderInit:
    def test_creates_with_defaults(self) -> None:
        loader = DocumentLoader()
        assert loader.max_file_size_bytes > 0

    def test_creates_with_custom_max_size(self) -> None:
        loader = DocumentLoader(max_file_size_mb=5.0)
        assert loader.max_file_size_bytes == 5 * 1024 * 1024


class TestDocumentLoaderLoadFile:
    def test_load_txt_file(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello world")
        loader = DocumentLoader()
        doc = loader.load_file(str(txt_file))
        assert doc.content == "Hello world"
        assert doc.file_type == "text"
        assert doc.file_name == "test.txt"

    def test_load_markdown_file(self, tmp_path: Path) -> None:
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Title\n\nContent here")
        loader = DocumentLoader()
        doc = loader.load_file(str(md_file))
        assert doc.content == "# Title\n\nContent here"
        assert doc.file_type == "markdown"

    def test_load_python_file(self, tmp_path: Path) -> None:
        py_file = tmp_path / "script.py"
        py_file.write_text("def hello():\n    print('hi')")
        loader = DocumentLoader()
        doc = loader.load_file(str(py_file))
        assert "def hello" in doc.content
        assert doc.file_type == "code"

    def test_load_json_file(self, tmp_path: Path) -> None:
        json_file = tmp_path / "config.json"
        json_file.write_text('{"key": "value"}')
        loader = DocumentLoader()
        doc = loader.load_file(str(json_file))
        assert doc.file_type == "code"

    def test_load_yaml_file(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "config.yml"
        yaml_file.write_text("key: value")
        loader = DocumentLoader()
        doc = loader.load_file(str(yaml_file))
        assert doc.file_type == "code"

    def test_metadata_contains_extension_and_size(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "meta.txt"
        txt_file.write_text("some content here")
        loader = DocumentLoader()
        doc = loader.load_file(str(txt_file))
        assert doc.metadata["extension"] == ".txt"
        assert doc.metadata["size_bytes"] > 0

    def test_file_not_found(self) -> None:
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/file.txt")

    def test_not_a_file(self, tmp_path: Path) -> None:
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Not a file"):
            loader.load_file(str(tmp_path))

    def test_unsupported_file_type(self, tmp_path: Path) -> None:
        xyz_file = tmp_path / "data.xyz"
        xyz_file.write_text("content")
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load_file(str(xyz_file))

    def test_file_too_large(self, tmp_path: Path) -> None:
        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * 2000)
        loader = DocumentLoader(max_file_size_mb=0.001)
        with pytest.raises(ValueError, match="File too large"):
            loader.load_file(str(large_file))


class TestDocumentLoaderLoadDirectory:
    def test_load_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("file A")
        (tmp_path / "b.md").write_text("# File B")
        (tmp_path / "c.py").write_text("x = 1")
        loader = DocumentLoader()
        docs = loader.load_directory(str(tmp_path))
        assert len(docs) == 3

    def test_load_directory_non_recursive(self, tmp_path: Path) -> None:
        (tmp_path / "root.txt").write_text("root")
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "child.txt").write_text("child")
        loader = DocumentLoader()
        docs = loader.load_directory(str(tmp_path), recursive=False)
        assert len(docs) == 1

    def test_load_directory_recursive(self, tmp_path: Path) -> None:
        (tmp_path / "root.txt").write_text("root")
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "child.txt").write_text("child")
        loader = DocumentLoader()
        docs = loader.load_directory(str(tmp_path), recursive=True)
        assert len(docs) == 2

    def test_load_directory_with_extension_filter(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("text")
        (tmp_path / "b.py").write_text("code")
        loader = DocumentLoader()
        docs = loader.load_directory(str(tmp_path), extensions=[".txt"])
        assert len(docs) == 1
        assert docs[0].file_type == "text"

    def test_load_directory_not_found(self) -> None:
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Not a directory"):
            loader.load_directory("/nonexistent/dir")

    def test_load_empty_directory(self, tmp_path: Path) -> None:
        loader = DocumentLoader()
        docs = loader.load_directory(str(tmp_path))
        assert docs == []
