"""检索服务主入口：统一编排分块、索引构建、加载、查询和融合排序。"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

from langchain_core.embeddings import Embeddings

from .bm25_index import BM25Index
from .chunking import MarkdownChunker
from .manifest import (
    IndexManifest,
    build_fingerprint,
    corpus_fingerprint,
    create_manifest,
)
from .schemas import KnowledgeChunk, KnowledgeDocument, RetrievalConfig, RetrievalResult
from .tokenization import create_bm25_tokenizer
from .vector_index import VectorIndex


logger = logging.getLogger(__name__)


class HybridRetrievalService:
    """与业务无关的混合检索服务，并支持版本化索引。"""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        embeddings: Embeddings | None = None,
        bm25_tokenizer: Callable[[str], list[str]] | None = None,
    ):
        # embeddings 可由调用方注入；测试中使用离线假向量，避免下载真实模型。
        self.config = config or RetrievalConfig()
        self.chunker = MarkdownChunker(self.config)
        self.vector_index = VectorIndex.from_config(self.config, embeddings)
        self.bm25_tokenizer = bm25_tokenizer or create_bm25_tokenizer(
            self.config.bm25_tokenizer,
            self.config.jieba_user_dict,
            self.config.jieba_domain_terms,
        )
        self._uses_custom_bm25_tokenizer = bm25_tokenizer is not None
        self.bm25_index: BM25Index | None = None
        self.chunks: list[KnowledgeChunk] = []
        self.manifest: IndexManifest | None = None

    def build(self, documents: list[KnowledgeDocument]) -> IndexManifest:
        """从完整文档构建新版本索引，并在全部成功后激活该版本。"""
        # 先把文档切成检索的最小单位，BM25 和向量索引共用同一批 chunks。
        chunks = self.chunker.split(documents)
        if not chunks:
            raise ValueError("no searchable chunks were produced")

        # 每次构建写入独立版本目录，失败时不会破坏当前正在使用的版本。
        manifest = create_manifest(documents, self.config, len(chunks))
        version_dir = self.config.index_dir / "versions" / manifest.version
        if version_dir.exists():
            suffix = 1
            while version_dir.with_name(f"{manifest.version}-{suffix}").exists():
                suffix += 1
            version_dir = version_dir.with_name(f"{manifest.version}-{suffix}")
            manifest.version = version_dir.name
        version_dir.mkdir(parents=True, exist_ok=False)

        # 磁盘文件全部写完后才更新 CURRENT，保证版本切换是原子的。
        self.vector_index.build(chunks)
        self.vector_index.save(version_dir / "faiss")
        _save_chunks(version_dir / "chunks.json", chunks)
        manifest.save(version_dir / "manifest.json")
        _activate_version(self.config.index_dir, manifest.version)

        # 同步更新内存状态，使 build 完成后可以立即查询，无需再次 load。
        self.chunks = chunks
        self.bm25_index = BM25Index(chunks, self.bm25_tokenizer)
        self.manifest = manifest
        return manifest

    def load(self) -> IndexManifest:
        """加载 CURRENT 指向的索引版本，并恢复 BM25 与向量检索状态。"""
        version = _read_current_version(self.config.index_dir)
        version_dir = self.config.index_dir / "versions" / version
        manifest = IndexManifest.load(version_dir / "manifest.json")
        # 分块参数或模型变化后，旧向量不能安全复用，要求调用方重新构建。
        if manifest.build_fingerprint != build_fingerprint(self.config):
            raise RuntimeError("index build configuration has changed; rebuild the index")

        chunks = _load_chunks(version_dir / "chunks.json")
        self.vector_index.load(version_dir / "faiss")
        self.bm25_index = BM25Index(chunks, self.bm25_tokenizer)
        self.chunks = chunks
        self.manifest = manifest
        return manifest

    def is_stale(self, documents: list[KnowledgeDocument]) -> bool:
        """判断当前索引是否因文档或建索引配置变化而需要重建。"""
        if self.manifest is None:
            try:
                version = _read_current_version(self.config.index_dir)
                self.manifest = IndexManifest.load(
                    self.config.index_dir / "versions" / version / "manifest.json"
                )
            except (FileNotFoundError, ValueError):
                return True
        return (
            # 两种指纹任意一种不一致，都说明磁盘索引已不是最新状态。
            self.manifest.corpus_fingerprint != corpus_fingerprint(documents)
            or self.manifest.build_fingerprint != build_fingerprint(self.config)
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, object] | None = None,
        mode: str | None = None,
    ) -> list[RetrievalResult]:
        """按指定模式检索，并返回融合排序后的前 top_k 条结果。"""
        if self.bm25_index is None or not self.chunks:
            raise RuntimeError("index is not loaded; call build() or load() first")
        if not query.strip():
            return []

        requested_mode = mode or self.config.search_mode
        if requested_mode not in {"hybrid", "vector", "bm25"}:
            raise ValueError(f"unsupported search mode: {requested_mode}")

        # 两路召回先独立执行，最后统一交给 _rank 排序。
        vector_results: list[tuple[KnowledgeChunk, float]] = []
        bm25_results: list[tuple[KnowledgeChunk, float]] = []
        if requested_mode in {"hybrid", "vector"}:
            try:
                vector_results = self.vector_index.search(
                    query,
                    self.config.vector_candidate_k,
                    filters,
                )
            except Exception:
                # hybrid 模式允许向量服务故障时降级到 BM25；纯 vector 模式则抛出异常。
                if requested_mode == "vector":
                    raise
                logger.exception("vector search failed; continuing with BM25 results")
        if requested_mode in {"hybrid", "bm25"}:
            bm25_results = self.bm25_index.search(
                query,
                self.config.bm25_candidate_k,
                filters,
            )

        return self._rank(
            vector_results,
            bm25_results,
            top_k or self.config.final_top_k,
        )

    def status(self) -> dict[str, object]:
        """返回适合健康检查或调试页面展示的简要状态。"""
        return {
            "ready": self.manifest is not None and bool(self.chunks),
            "version": self.manifest.version if self.manifest else None,
            "document_count": self.manifest.document_count if self.manifest else 0,
            "chunk_count": len(self.chunks),
            "embedding_model": self.config.embedding_model,
            "search_mode": self.config.search_mode,
            "bm25_tokenizer": (
                "custom" if self._uses_custom_bm25_tokenizer else self.config.bm25_tokenizer
            ),
        }

    def _rank(
        self,
        vector_results: list[tuple[KnowledgeChunk, float]],
        bm25_results: list[tuple[KnowledgeChunk, float]],
        top_k: int,
    ) -> list[RetrievalResult]:
        """使用 RRF 融合两路排名，并限制同一文档占据过多结果位置。"""
        evidence: dict[str, dict[str, object]] = {}

        # RRF 只依赖名次，不直接比较量纲不同的向量距离和 BM25 分数。
        # 同一个 chunk 在两路排名都靠前时，会累计得到更高的融合分数。
        for rank, (chunk, raw_score) in enumerate(vector_results, start=1):
            item = evidence.setdefault(chunk.id, {"chunk": chunk, "score": 0.0})
            item["vector_rank"] = rank
            item["vector_score"] = raw_score
            item["score"] = float(item["score"]) + 1.0 / (self.config.rrf_k + rank)
        for rank, (chunk, raw_score) in enumerate(bm25_results, start=1):
            item = evidence.setdefault(chunk.id, {"chunk": chunk, "score": 0.0})
            item["bm25_rank"] = rank
            item["bm25_score"] = raw_score
            item["score"] = float(item["score"]) + 1.0 / (self.config.rrf_k + rank)

        ranked = sorted(evidence.values(), key=lambda item: float(item["score"]), reverse=True)
        per_document: dict[str, int] = {}
        results: list[RetrievalResult] = []
        for item in ranked:
            chunk = item["chunk"]
            assert isinstance(chunk, KnowledgeChunk)
            count = per_document.get(chunk.document_id, 0)
            # 避免同一篇长文档的多个相似分块挤满最终结果。
            if count >= self.config.max_chunks_per_document:
                continue
            per_document[chunk.document_id] = count + 1
            results.append(
                RetrievalResult(
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    title=chunk.title,
                    content=chunk.content,
                    heading_path=chunk.heading_path,
                    metadata=dict(chunk.metadata),
                    score=float(item["score"]),
                    vector_rank=_optional_int(item.get("vector_rank")),
                    bm25_rank=_optional_int(item.get("bm25_rank")),
                    vector_score=_optional_float(item.get("vector_score")),
                    bm25_score=_optional_float(item.get("bm25_score")),
                )
            )
            if len(results) >= top_k:
                break
        return results


def _save_chunks(path: Path, chunks: list[KnowledgeChunk]) -> None:
    """把分块保存为 JSON，供下次启动时重建 BM25 索引和返回原文。"""
    path.write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _load_chunks(path: Path) -> list[KnowledgeChunk]:
    """从 JSON 文件恢复全部分块。"""
    return [
        KnowledgeChunk.from_dict(value)
        for value in json.loads(path.read_text(encoding="utf-8"))
    ]


def _activate_version(index_dir: Path, version: str) -> None:
    """原子更新 CURRENT，确保读取方不会看到写到一半的版本号。"""
    index_dir.mkdir(parents=True, exist_ok=True)
    temporary = index_dir / "CURRENT.tmp"
    temporary.write_text(version, encoding="utf-8")
    os.replace(temporary, index_dir / "CURRENT")


def _read_current_version(index_dir: Path) -> str:
    """读取当前版本号，并拒绝可能逃逸索引目录的非法路径。"""
    path = index_dir / "CURRENT"
    if not path.exists():
        raise FileNotFoundError(f"no active index found in {index_dir}")
    version = path.read_text(encoding="utf-8").strip()
    if not version or "/" in version or "\\" in version:
        raise ValueError("invalid active index version")
    return version


def _optional_int(value: object) -> int | None:
    """把可选的排名值转换为 int。"""
    return int(value) if value is not None else None


def _optional_float(value: object) -> float | None:
    """把可选的原始分数转换为 float。"""
    return float(value) if value is not None else None
