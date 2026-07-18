"""FAISS adapter isolated from chunking and service orchestration."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .schemas import KnowledgeChunk, RetrievalConfig


class VectorIndex:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.store: FAISS | None = None

    @classmethod
    def from_config(
        cls,
        config: RetrievalConfig,
        embeddings: Embeddings | None = None,
    ) -> "VectorIndex":
        if embeddings is None:
            # Keep model runtimes lazy: callers can inject an API embedding or a
            # test double without importing PyTorch/Sentence Transformers.
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=config.embedding_model,
                model_kwargs={"device": config.device},
                encode_kwargs={"normalize_embeddings": True},
            )
        return cls(embeddings)

    def build(self, chunks: list[KnowledgeChunk]) -> None:
        if not chunks:
            raise ValueError("cannot build a vector index without chunks")
        self.store = FAISS.from_documents(
            [_to_langchain_document(chunk) for chunk in chunks],
            self.embeddings,
        )

    def save(self, path: str | Path) -> None:
        if self.store is None:
            raise RuntimeError("vector index has not been built")
        self.store.save_local(str(path))

    def load(self, path: str | Path) -> None:
        # The pickle sidecar is trusted only because this package creates and owns
        # the index directory. Never point this method at a user-uploaded index.
        self.store = FAISS.load_local(
            str(path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def search(
        self,
        query: str,
        limit: int,
        filters: dict[str, object] | None = None,
    ) -> list[tuple[KnowledgeChunk, float]]:
        if self.store is None:
            raise RuntimeError("vector index is not loaded")
        index_size = int(self.store.index.ntotal)
        if index_size == 0:
            return []
        safe_limit = min(limit, index_size)
        pairs = self.store.similarity_search_with_score(
            query,
            k=safe_limit,
            filter=filters or None,
            fetch_k=min(max(safe_limit * 2, safe_limit), index_size),
        )
        return [(_from_langchain_document(document), float(score)) for document, score in pairs]


def _to_langchain_document(chunk: KnowledgeChunk) -> Document:
    return Document(
        page_content=chunk.content,
        metadata={
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "title": chunk.title,
            "heading_path": list(chunk.heading_path),
            "chunk_index": chunk.chunk_index,
            **chunk.metadata,
        },
    )


def _from_langchain_document(document: Document) -> KnowledgeChunk:
    metadata = dict(document.metadata)
    chunk_id = str(metadata.pop("chunk_id"))
    document_id = str(metadata.pop("document_id"))
    title = str(metadata.pop("title"))
    heading_path = tuple(metadata.pop("heading_path", ()))
    chunk_index = int(metadata.pop("chunk_index", 0))
    return KnowledgeChunk(
        id=chunk_id,
        document_id=document_id,
        title=title,
        content=document.page_content,
        heading_path=heading_path,
        chunk_index=chunk_index,
        metadata=metadata,
    )
