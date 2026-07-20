"""按照 Markdown 标题和字符长度切分文档，并生成稳定的分块 ID。"""

from __future__ import annotations

import hashlib

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .schemas import KnowledgeChunk, KnowledgeDocument, RetrievalConfig


class MarkdownChunker:
    """先按 Markdown 标题分段，再按长度切分过大的段落。"""

    HEADERS = (("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"))

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=list(self.HEADERS),
            strip_headers=False,
        )
        self.length_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_max_chars,
            chunk_overlap=config.chunk_overlap_chars,
            separators=["\n\n", "\n", "。", "！", "？", ";", "；", " ", ""],
            length_function=len,
        )

    def split(self, documents: list[KnowledgeDocument]) -> list[KnowledgeChunk]:
        """依次切分多篇文档，并合并成一个分块列表。"""
        chunks: list[KnowledgeChunk] = []
        for document in documents:
            chunks.extend(self._split_document(document))
        return chunks

    def _split_document(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        """切分单篇文档，同时保留标题路径和原文档元数据。"""
        # 第一层切分保留 Markdown 的章节语义，例如“网络问题 > 掉线”。
        sections = self.header_splitter.split_text(document.content)
        chunk_index = 0
        result: list[KnowledgeChunk] = []

        for section in sections:
            # LangChain 将各级标题保存在 h1/h2/h3/h4 元数据中。
            heading_path = tuple(
                section.metadata[key]
                for key in ("h1", "h2", "h3", "h4")
                if section.metadata.get(key)
            )
            # 第二层切分控制送入 Embedding 模型的文本长度，并保留 overlap 上下文。
            parts = self.length_splitter.split_text(section.page_content)
            for part in parts:
                content = part.strip()
                if not content:
                    continue
                chunk_id = _stable_chunk_id(document.id, heading_path, chunk_index, content)
                metadata = dict(document.metadata)
                if document.path:
                    # setdefault 避免覆盖上游已经显式提供的 path。
                    metadata.setdefault("path", document.path)
                result.append(
                    KnowledgeChunk(
                        id=chunk_id,
                        document_id=document.id,
                        title=document.title,
                        content=content,
                        heading_path=heading_path,
                        chunk_index=chunk_index,
                        metadata=metadata,
                    )
                )
                chunk_index += 1
        return result


def _stable_chunk_id(
    document_id: str,
    heading_path: tuple[str, ...],
    chunk_index: int,
    content: str,
) -> str:
    """根据来源、标题路径、顺序和内容生成可重复计算的分块 ID。"""
    value = "\x1f".join((document_id, *heading_path, str(chunk_index), content))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
