"""FAISS 向量索引适配层，把向量检索与分块、服务编排逻辑隔离。"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .schemas import KnowledgeChunk, RetrievalConfig


# 内部状态与用户元数据分开存储，避免用户恰好使用同名字段时互相覆盖。
_INTERNAL_STATE_KEY = "__hybrid_rag_internal__"
_USER_METADATA_KEY = "__hybrid_rag_user_metadata__"


class VectorIndex:
    """负责构建、持久化、加载和查询 FAISS 索引。"""

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.store: FAISS | None = None

    @classmethod
    def from_config(
        cls,
        config: RetrievalConfig,
        embeddings: Embeddings | None = None,
    ) -> "VectorIndex":
        """根据配置创建向量索引，也允许测试或业务方注入 Embedding 实现。"""
        if embeddings is None:
            # 延迟导入模型运行时：传入 API Embedding 或测试替身时无需加载 PyTorch。
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=config.embedding_model,
                model_kwargs={"device": config.device},
                encode_kwargs={"normalize_embeddings": True},
            )
        return cls(embeddings)

    def build(self, chunks: list[KnowledgeChunk]) -> None:
        """把所有分块转换为 LangChain Document 并建立内存中的 FAISS 索引。"""
        if not chunks:
            raise ValueError("cannot build a vector index without chunks")
        self.store = FAISS.from_documents(
            [_to_langchain_document(chunk) for chunk in chunks],
            self.embeddings,
        )

    def save(self, path: str | Path) -> None:
        """将内存中的 FAISS 索引保存到指定目录。"""
        if self.store is None:
            raise RuntimeError("vector index has not been built")
        self.store.save_local(str(path))

    def load(self, path: str | Path) -> None:
        """加载本组件自己生成的 FAISS 索引。"""
        # FAISS 的 sidecar 使用 pickle。这里仅信任本组件管理的索引目录，
        # 绝不能把用户上传的未知索引路径直接传入此方法。
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
        """返回与查询最相似的分块，以及 FAISS 给出的原始距离分数。"""
        if self.store is None:
            raise RuntimeError("vector index is not loaded")
        index_size = int(self.store.index.ntotal)
        if index_size == 0:
            return []
        # k 和 fetch_k 不能超过实际索引大小，小数据集尤其需要这层保护。
        safe_limit = min(limit, index_size)
        pairs = self.store.similarity_search_with_score(
            query,
            k=safe_limit,
            filter=filters or None,
            fetch_k=min(max(safe_limit * 2, safe_limit), index_size),
        )
        return [(_from_langchain_document(document), float(score)) for document, score in pairs]


def _to_langchain_document(chunk: KnowledgeChunk) -> Document:
    """把项目内部的 KnowledgeChunk 转换为 LangChain Document。"""
    user_metadata = dict(chunk.metadata)
    metadata = dict(user_metadata)
    # 用户元数据保留在顶层供 FAISS 过滤，同时保存一份原样副本，
    # 防止用户字段名与组件内部字段名冲突。
    metadata[_USER_METADATA_KEY] = user_metadata
    metadata[_INTERNAL_STATE_KEY] = {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "heading_path": list(chunk.heading_path),
        "chunk_index": chunk.chunk_index,
        "raw_content": chunk.content,
    }
    return Document(
        page_content=_build_vector_text(chunk),
        metadata=metadata,
    )


def _from_langchain_document(document: Document) -> KnowledgeChunk:
    """把向量检索结果还原为业务侧使用的 KnowledgeChunk。"""
    metadata = dict(document.metadata)
    internal = metadata.pop(_INTERNAL_STATE_KEY, None)
    if internal is None:
        # 兼容引入“增强向量文本”之前生成的旧索引格式。
        internal = {
            "chunk_id": metadata.pop("chunk_id"),
            "document_id": metadata.pop("document_id"),
            "title": metadata.pop("title"),
            "heading_path": metadata.pop("heading_path", ()),
            "chunk_index": metadata.pop("chunk_index", 0),
            "raw_content": document.page_content,
        }
        user_metadata = metadata
    else:
        user_metadata = metadata.pop(_USER_METADATA_KEY, metadata)
    return KnowledgeChunk(
        id=str(internal["chunk_id"]),
        document_id=str(internal["document_id"]),
        title=str(internal["title"]),
        content=str(internal["raw_content"]),
        heading_path=tuple(internal.get("heading_path", ())),
        chunk_index=int(internal.get("chunk_index", 0)),
        metadata=dict(user_metadata),
    )


def _build_vector_text(chunk: KnowledgeChunk) -> str:
    """组合标题、标题路径、标签和正文，作为 Embedding 模型的输入文本。"""
    # 标题和标签通常包含高密度关键词，加入向量文本有助于召回短查询。
    parts = [chunk.title.strip()]
    parts.extend(heading.strip() for heading in chunk.heading_path if heading.strip())

    tags = chunk.metadata.get("tags")
    if isinstance(tags, str):
        if tags.strip():
            parts.append(tags.strip())
    elif isinstance(tags, (list, tuple, set)):
        tag_text = " ".join(str(tag).strip() for tag in tags if str(tag).strip())
        if tag_text:
            parts.append(tag_text)

    parts.append(chunk.content)
    return "\n".join(part for part in parts if part)
