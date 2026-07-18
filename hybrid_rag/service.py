"""Public orchestration API for building, loading, and querying an index."""

from __future__ import annotations

import json
import logging
import os
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
from .vector_index import VectorIndex


logger = logging.getLogger(__name__)


class HybridRetrievalService:
    """Domain-neutral hybrid retrieval service with versioned index builds."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        embeddings: Embeddings | None = None,
    ):
        self.config = config or RetrievalConfig()
        self.chunker = MarkdownChunker(self.config)
        self.vector_index = VectorIndex.from_config(self.config, embeddings)
        self.bm25_index: BM25Index | None = None
        self.chunks: list[KnowledgeChunk] = []
        self.manifest: IndexManifest | None = None

    def build(self, documents: list[KnowledgeDocument]) -> IndexManifest:
        chunks = self.chunker.split(documents)
        if not chunks:
            raise ValueError("no searchable chunks were produced")

        manifest = create_manifest(documents, self.config, len(chunks))
        version_dir = self.config.index_dir / "versions" / manifest.version
        if version_dir.exists():
            suffix = 1
            while version_dir.with_name(f"{manifest.version}-{suffix}").exists():
                suffix += 1
            version_dir = version_dir.with_name(f"{manifest.version}-{suffix}")
            manifest.version = version_dir.name
        version_dir.mkdir(parents=True, exist_ok=False)

        self.vector_index.build(chunks)
        self.vector_index.save(version_dir / "faiss")
        _save_chunks(version_dir / "chunks.json", chunks)
        manifest.save(version_dir / "manifest.json")
        _activate_version(self.config.index_dir, manifest.version)

        self.chunks = chunks
        self.bm25_index = BM25Index(chunks)
        self.manifest = manifest
        return manifest

    def load(self) -> IndexManifest:
        version = _read_current_version(self.config.index_dir)
        version_dir = self.config.index_dir / "versions" / version
        manifest = IndexManifest.load(version_dir / "manifest.json")
        if manifest.build_fingerprint != build_fingerprint(self.config):
            raise RuntimeError("index build configuration has changed; rebuild the index")

        chunks = _load_chunks(version_dir / "chunks.json")
        self.vector_index.load(version_dir / "faiss")
        self.bm25_index = BM25Index(chunks)
        self.chunks = chunks
        self.manifest = manifest
        return manifest

    def is_stale(self, documents: list[KnowledgeDocument]) -> bool:
        if self.manifest is None:
            try:
                version = _read_current_version(self.config.index_dir)
                self.manifest = IndexManifest.load(
                    self.config.index_dir / "versions" / version / "manifest.json"
                )
            except (FileNotFoundError, ValueError):
                return True
        return (
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
        if self.bm25_index is None or not self.chunks:
            raise RuntimeError("index is not loaded; call build() or load() first")
        if not query.strip():
            return []

        requested_mode = mode or self.config.search_mode
        if requested_mode not in {"hybrid", "vector", "bm25"}:
            raise ValueError(f"unsupported search mode: {requested_mode}")

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
        return {
            "ready": self.manifest is not None and bool(self.chunks),
            "version": self.manifest.version if self.manifest else None,
            "document_count": self.manifest.document_count if self.manifest else 0,
            "chunk_count": len(self.chunks),
            "embedding_model": self.config.embedding_model,
            "search_mode": self.config.search_mode,
        }

    def _rank(
        self,
        vector_results: list[tuple[KnowledgeChunk, float]],
        bm25_results: list[tuple[KnowledgeChunk, float]],
        top_k: int,
    ) -> list[RetrievalResult]:
        evidence: dict[str, dict[str, object]] = {}
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
    path.write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _load_chunks(path: Path) -> list[KnowledgeChunk]:
    return [
        KnowledgeChunk.from_dict(value)
        for value in json.loads(path.read_text(encoding="utf-8"))
    ]


def _activate_version(index_dir: Path, version: str) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    temporary = index_dir / "CURRENT.tmp"
    temporary.write_text(version, encoding="utf-8")
    os.replace(temporary, index_dir / "CURRENT")


def _read_current_version(index_dir: Path) -> str:
    path = index_dir / "CURRENT"
    if not path.exists():
        raise FileNotFoundError(f"no active index found in {index_dir}")
    version = path.read_text(encoding="utf-8").strip()
    if not version or "/" in version or "\\" in version:
        raise ValueError("invalid active index version")
    return version


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None

