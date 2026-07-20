"""检索模块使用的统一数据结构，与具体业务和数据库模型解耦。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .tokenization import BM25TokenizerMode


# hybrid：融合检索；vector：仅向量检索；bm25：仅关键词检索。
SearchMode = Literal["hybrid", "vector", "bm25"]


@dataclass(slots=True)
class KnowledgeDocument:
    """上游提供的完整知识文档，可以来自文件、数据库或 API。"""

    id: str
    title: str
    content: str
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """统一 ID 和标题格式，并尽早拒绝无法检索的空文档。"""
        self.id = str(self.id).strip()
        self.title = self.title.strip()
        if not self.id:
            raise ValueError("document id cannot be empty")
        if not self.content.strip():
            raise ValueError(f"document {self.id!r} content cannot be empty")


@dataclass(slots=True)
class KnowledgeChunk:
    """从完整文档切出的可检索片段，同一输入会得到稳定的 ID。"""

    id: str
    document_id: str
    title: str
    content: str
    heading_path: tuple[str, ...]
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 JSON 的字典。"""
        value = asdict(self)
        value["heading_path"] = list(self.heading_path)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KnowledgeChunk":
        """从索引目录中的 JSON 数据恢复分块对象。"""
        value = dict(value)
        value["heading_path"] = tuple(value.get("heading_path", ()))
        return cls(**value)


@dataclass(slots=True)
class RetrievalResult:
    """最终排序结果，同时保留两路排名和分数，方便调试与效果对比。"""

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
    """集中保存建索引参数和查询参数。"""

    # 以下三项会改变分块或向量，修改后需要重建索引。
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    index_path: str = "./vector_index"
    chunk_max_chars: int = 800
    chunk_overlap_chars: int = 100

    # 以下参数只影响查询阶段，不需要重建已有索引。
    vector_candidate_k: int = 20
    bm25_candidate_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60
    max_chunks_per_document: int = 2
    search_mode: SearchMode = "hybrid"
    device: str = "cpu"
    bm25_tokenizer: BM25TokenizerMode = "ngram"
    jieba_user_dict: str | None = None
    jieba_domain_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """在服务启动时检查配置，避免错误参数进入后续流程。"""
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
        if self.bm25_tokenizer not in {"ngram", "jieba", "hybrid"}:
            raise ValueError(f"unsupported bm25_tokenizer: {self.bm25_tokenizer}")
        self.jieba_domain_terms = tuple(
            dict.fromkeys(
                str(term).strip() for term in self.jieba_domain_terms if str(term).strip()
            )
        )
        if self.bm25_tokenizer == "ngram" and (
            self.jieba_user_dict is not None or self.jieba_domain_terms
        ):
            raise ValueError(
                "Jieba dictionaries require bm25_tokenizer='jieba' or 'hybrid'"
            )

    @property
    def index_dir(self) -> Path:
        """把字符串形式的索引路径转换为 Path，方便文件操作。"""
        return Path(self.index_path)

    def build_settings(self) -> dict[str, Any]:
        """只返回会改变分块或向量、因而需要重建索引的配置。"""
        return {
            "embedding_model": self.embedding_model,
            "chunk_max_chars": self.chunk_max_chars,
            "chunk_overlap_chars": self.chunk_overlap_chars,
        }
