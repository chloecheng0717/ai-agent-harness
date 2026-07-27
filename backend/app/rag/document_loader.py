"""
Document Loader

Loads and extracts text from PDF, TXT, Markdown, and code files.
"""

import os
from pathlib import Path
from typing import List, Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger().bind(component="document_loader")

SUPPORTED_EXTENSIONS = {
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".java": "code",
    ".go": "code",
    ".rs": "code",
    ".c": "code",
    ".cpp": "code",
    ".h": "code",
    ".hpp": "code",
    ".cs": "code",
    ".rb": "code",
    ".php": "code",
    ".swift": "code",
    ".kt": "code",
    ".sh": "code",
    ".bash": "code",
    ".yml": "code",
    ".yaml": "code",
    ".json": "code",
    ".toml": "code",
    ".ini": "code",
    ".cfg": "code",
    ".conf": "code",
    ".html": "code",
    ".css": "code",
    ".sql": "code",
    ".r": "code",
    ".scala": "code",
    ".pdf": "pdf",
}


class LoadedDocument(BaseModel):
    """A loaded document with extracted text and metadata."""

    file_path: str = Field(..., description="Absolute path to the source file")
    file_name: str = Field(..., description="Base name of the file")
    file_type: str = Field(..., description="Detected file type (text, markdown, code, pdf)")
    content: str = Field(..., description="Extracted text content")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

    model_config = {"arbitrary_types_allowed": True}


class DocumentLoader:
    """
    Loads documents from the filesystem and extracts their text content.

    Supports PDF, TXT, Markdown, and common code file extensions.
    """

    def __init__(self, max_file_size_mb: float = 10.0) -> None:
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

    def load_file(self, file_path: str) -> LoadedDocument:
        """Load a single file and extract its text content."""
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        file_size = path.stat().st_size
        if file_size > self.max_file_size_bytes:
            raise ValueError(
                f"File too large: {file_size} bytes "
                f"(max: {self.max_file_size_bytes})"
            )

        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS.keys())}"
            )

        file_type = SUPPORTED_EXTENSIONS[extension]
        content = self._extract_text(path, file_type)

        return LoadedDocument(
            file_path=str(path),
            file_name=path.name,
            file_type=file_type,
            content=content,
            metadata={
                "extension": extension,
                "size_bytes": file_size,
            },
        )

    def load_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> List[LoadedDocument]:
        """Load all supported files from a directory."""
        root = Path(dir_path).resolve()

        if not root.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")

        allowed_exts = set(extensions) if extensions else set(SUPPORTED_EXTENSIONS.keys())
        documents: List[LoadedDocument] = []

        pattern = "**/*" if recursive else "*"

        for file_path in sorted(root.glob(pattern)):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in allowed_exts:
                continue
            try:
                doc = self.load_file(str(file_path))
                documents.append(doc)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning(
                    "Skipping file",
                    path=str(file_path),
                    error=str(exc),
                )

        logger.info(
            "Directory loaded",
            dir_path=dir_path,
            document_count=len(documents),
        )
        return documents

    def _extract_text(self, path: Path, file_type: str) -> str:
        """Extract text content based on file type."""
        if file_type == "pdf":
            return self._extract_pdf(path)
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _extract_pdf(path: Path) -> str:
        """Extract text from a PDF file."""
        try:
            import subprocess
            import sys

            result = subprocess.run(
                [sys.executable, "-m", "pdfminer.high_level", str(path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except Exception:
            pass

        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n".join(text_parts)
        except ImportError:
            pass

        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n".join(text_parts)
        except ImportError:
            pass

        logger.warning("No PDF library available, returning placeholder", path=str(path))
        return f"[PDF content could not be extracted: {path.name}]"
