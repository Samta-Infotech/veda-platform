# =============================================================================
# connectors/document.py
# VEDA — Document Source Connector (Phase 2)
#
# Implements BaseConnector for document sources (PDF, Word, TXT, MD, HTML).
# FilesystemDocumentConnector walks a local directory and yields DocumentChunk
# objects for every supported file.
#
# Optional dependencies (graceful fallback when not installed):
#   pdfplumber     — pip install pdfplumber
#   python-docx    — pip install python-docx
#   beautifulsoup4 — pip install beautifulsoup4 lxml
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import uuid
from pathlib import Path
from typing import Iterator, List

from connectors.base import (
    BaseConnector,
    ConnectorState,
    ConnectorStatus,
    DocumentChunk,
    register_connector,
)
from config import (
    DOC_CHUNK_SIZE,
    DOC_CHUNK_OVERLAP,
    DOC_MAX_FILE_MB,
    DOC_SUPPORTED_FORMATS,
)


# =============================================================================
# Optional dependency imports
# =============================================================================

try:
    import pdfplumber
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

try:
    from docx import Document as _DocxDocument
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False


# =============================================================================
# Text extraction helpers — one per format
# =============================================================================

def _extract_pdf(path: Path) -> List[tuple]:
    """Returns [(page_num, text), ...] for each page. Requires pdfplumber."""
    if not _PDF_AVAILABLE:
        return []
    pages = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((i + 1, text))
    except Exception:
        pass
    return pages


def _extract_docx(path: Path) -> str:
    """Returns full text of a Word document. Requires python-docx."""
    if not _DOCX_AVAILABLE:
        return ""
    try:
        doc = _DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception:
        return ""


def _extract_html(path: Path) -> str:
    """Returns tag-stripped text from HTML. Uses beautifulsoup4 when available."""
    raw = ""
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if not _BS4_AVAILABLE:
        return raw
    try:
        return BeautifulSoup(raw, "lxml").get_text(separator=" ", strip=True)
    except Exception:
        return raw


def _extract_plain(path: Path) -> str:
    """Returns raw text from TXT, MD, or any plain-text file."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""



def _extract_csv(path: "Path") -> str:
    """
    Converts a CSV file into human-readable text for RAG chunking.

    Each row is rendered as a natural language sentence:
      "project_name: Norm Saratoga | num_stories: 12 | status: active"

    This preserves column context so MiniLM can match queries like
    "number of stories in project Norm Saratoga" against the row text.

    No external library needed — stdlib csv only.
    """
    import csv as _csv
    try:
        lines = []
        with open(str(path), encoding="utf-8", errors="ignore", newline="") as f:
            reader = _csv.DictReader(f)
            headers = reader.fieldnames or []
            for i, row in enumerate(reader):
                # Build "col: val | col: val | ..." for each row
                parts = [
                    f"{str(k).strip()}: {str(v).strip()}"
                    for k, v in row.items()
                    if v and str(v).strip()
                ]
                if parts:
                    lines.append(" | ".join(parts))
        # Return all rows as newline-separated sentences
        return chr(10).join(lines)
    except Exception:
        # Fallback: plain text read
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""


# =============================================================================
# Chunking
# =============================================================================

def _chunk_text(
    text:          str,
    chunk_size:    int = DOC_CHUNK_SIZE,
    chunk_overlap: int = DOC_CHUNK_OVERLAP,
) -> List[str]:
    """
    Splits text into overlapping word-based chunks.
    chunk_size / chunk_overlap are in words (proxy for tokens at ~1:1 ratio).
    """
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - chunk_overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


# =============================================================================
# FilesystemDocumentConnector
# =============================================================================

class FilesystemDocumentConnector(BaseConnector):
    """
    Document connector for local filesystem directories.

    Walks a directory, extracts text per file format, chunks it,
    and yields DocumentChunk objects consumed by chunk_embedder.py.
    """

    def __init__(self, source_config: dict) -> None:
        super().__init__(source_config)
        self._path      = Path(source_config.get("path", "."))
        self._formats   = set(source_config.get("formats", DOC_SUPPORTED_FORMATS))
        self._recursive = source_config.get("recursive", True)
        self._max_mb    = source_config.get("max_file_mb", DOC_MAX_FILE_MB)

    @property
    def supports_chunks(self) -> bool:
        return True

    def connect(self) -> ConnectorStatus:
        t0 = time.time()
        try:
            if not self._path.exists():
                raise FileNotFoundError(f"Path does not exist: {self._path}")
            if not self._path.is_dir():
                raise NotADirectoryError(f"Not a directory: {self._path}")
            self._state = ConnectorState.CONNECTED
            return ConnectorStatus(
                ok          = True,
                source_id   = self._source_id,
                source_type = "document",
                engine      = self._engine,
                message     = f"Connected to {self._path}",
                latency_ms  = round((time.time() - t0) * 1000, 2),
                metadata    = {"path": str(self._path), "formats": list(self._formats)},
            )
        except Exception as e:
            self._state = ConnectorState.ERROR
            return ConnectorStatus(
                ok          = False,
                source_id   = self._source_id,
                source_type = "document",
                engine      = self._engine,
                message     = str(e),
                latency_ms  = round((time.time() - t0) * 1000, 2),
            )

    def disconnect(self) -> None:
        self._state = ConnectorState.DISCONNECTED

    def get_document_count(self) -> int:
        return sum(1 for _ in self._iter_files())

    def get_chunks(
        self,
        chunk_size:    int = DOC_CHUNK_SIZE,
        chunk_overlap: int = DOC_CHUNK_OVERLAP,
    ) -> Iterator[DocumentChunk]:
        """
        Yields DocumentChunk for every file in the source directory.
        Files larger than max_file_mb are silently skipped.
        """
        for path in self._iter_files():
            yield from self._process_file(path, chunk_size, chunk_overlap)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # Extensions always accepted regardless of DOC_SUPPORTED_FORMATS config
    _ALWAYS_ACCEPT = {"csv"}

    def _iter_files(self) -> Iterator[Path]:
        pattern = "**/*" if self._recursive else "*"
        for path in self._path.glob(pattern):
            ext = path.suffix.lower().lstrip(".")
            if path.is_file() and (ext in self._formats or ext in self._ALWAYS_ACCEPT):
                yield path

    def _process_file(
        self,
        path:          Path,
        chunk_size:    int,
        chunk_overlap: int,
    ) -> Iterator[DocumentChunk]:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self._max_mb:
            return

        doc_id  = str(uuid.uuid4())
        ext     = path.suffix.lower().lstrip(".")

        if ext == "pdf":
            for page_num, page_text in _extract_pdf(path):
                for idx, chunk_text in enumerate(_chunk_text(page_text, chunk_size, chunk_overlap)):
                    yield DocumentChunk(
                        chunk_id    = str(uuid.uuid4()),
                        source_id   = self._source_id,
                        doc_id      = doc_id,
                        doc_name    = path.name,
                        doc_path    = str(path),
                        doc_format  = "pdf",
                        chunk_index = idx,
                        text        = chunk_text,
                        page_num    = page_num,
                    )
        else:
            if ext == "docx":
                text = _extract_docx(path)
            elif ext in ("html", "htm"):
                text = _extract_html(path)
            elif ext == "csv":
                text = _extract_csv(path)
            else:
                text = _extract_plain(path)

            if not text.strip():
                return

            for idx, chunk_text in enumerate(_chunk_text(text, chunk_size, chunk_overlap)):
                yield DocumentChunk(
                    chunk_id    = str(uuid.uuid4()),
                    source_id   = self._source_id,
                    doc_id      = doc_id,
                    doc_name    = path.name,
                    doc_path    = str(path),
                    doc_format  = ext,
                    chunk_index = idx,
                    text        = chunk_text,
                    page_num    = None,
                )


# =============================================================================
# Connector registration — runs at import time
# =============================================================================

register_connector("document", "filesystem", FilesystemDocumentConnector)


def _ensure_registered() -> None:
    pass   # registration happens at module level above