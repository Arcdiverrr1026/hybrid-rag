"""Domain-neutral data contracts used by the retrieval package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


SearchMode = Literal["hybrid", "vector", "bm25"]


@dataclass(slots=True)
class KnowledgeDocument:
    """A source document supplied by a file loader, database, or API."""

    id: str
    title: str
    content: str
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id).strip()
        self.title = self.title.strip()
        if not self.id:
            raise ValueError("document id cannot be empty")
        if not self.content.strip():
            raise ValueError(f"document {self.id!r} content cannot be empty")


@dataclass(slots=True)
class KnowledgeChunk:
    """A deterministic, searchable section of a source document."""

    id: str
    document_id: str
    title: str
    content: str
    heading_path: tuple[str, ...]
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["heading_path"] = list(self.heading_path)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KnowledgeChunk":
        value = dict(value)
        value["heading_path"] = tuple(value.get("heading_path", ()))
        return cls(**value)


@dataclass(slots=True)
class RetrievalResult:
    """A ranked result with enough evidence for debugging and comparison."""

    document_id: str
    chunk_id: str
    title: str
    content: str
    heading_path: tuple[str, ...]
    metadata: dict[str, Any]
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    vector_score: float | None = None
    bm25_score: float | None = None


@dataclass(slots=True)
class RetrievalConfig:
    """Build-time and query-time retrieval configuration."""

    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    index_path: str = "./vector_index"
    chunk_max_chars: int = 800
    chunk_overlap_chars: int = 100
    vector_candidate_k: int = 20
    bm25_candidate_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60
    max_chunks_per_document: int = 2
    search_mode: SearchMode = "hybrid"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.chunk_max_chars <= 0:
            raise ValueError("chunk_max_chars must be positive")
        if not 0 <= self.chunk_overlap_chars < self.chunk_max_chars:
            raise ValueError("chunk_overlap_chars must be >= 0 and smaller than chunk_max_chars")
        for name in (
            "vector_candidate_k",
            "bm25_candidate_k",
            "final_top_k",
            "rrf_k",
            "max_chunks_per_document",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.search_mode not in {"hybrid", "vector", "bm25"}:
            raise ValueError(f"unsupported search_mode: {self.search_mode}")

    @property
    def index_dir(self) -> Path:
        return Path(self.index_path)

    def build_settings(self) -> dict[str, Any]:
        """Only settings that change vectors or chunks require a rebuild."""
        return {
            "embedding_model": self.embedding_model,
            "chunk_max_chars": self.chunk_max_chars,
            "chunk_overlap_chars": self.chunk_overlap_chars,
        }

