"""Markdown-aware chunking with deterministic identifiers."""

from __future__ import annotations

import hashlib

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .schemas import KnowledgeChunk, KnowledgeDocument, RetrievalConfig


class MarkdownChunker:
    """Split by Markdown headings, then split oversized sections by length."""

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
        chunks: list[KnowledgeChunk] = []
        for document in documents:
            chunks.extend(self._split_document(document))
        return chunks

    def _split_document(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        sections = self.header_splitter.split_text(document.content)
        chunk_index = 0
        result: list[KnowledgeChunk] = []

        for section in sections:
            heading_path = tuple(
                section.metadata[key]
                for key in ("h1", "h2", "h3", "h4")
                if section.metadata.get(key)
            )
            parts = self.length_splitter.split_text(section.page_content)
            for part in parts:
                content = part.strip()
                if not content:
                    continue
                chunk_id = _stable_chunk_id(document.id, heading_path, chunk_index, content)
                metadata = dict(document.metadata)
                if document.path:
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
    value = "\x1f".join((document_id, *heading_path, str(chunk_index), content))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

