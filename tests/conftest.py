from __future__ import annotations

import hashlib

import pytest
from langchain_core.embeddings import Embeddings


class DeterministicEmbeddings(Embeddings):
    """测试专用的离线确定性向量，用于验证 FAISS 构建和持久化流程。"""

    dimensions = 16

    def __init__(self) -> None:
        self.document_texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_texts.extend(texts)
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        # 同一文本始终映射到同一向量，测试无需下载模型，也不会受网络影响。
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[index] / 255.0 for index in range(self.dimensions)]


@pytest.fixture
def embeddings() -> DeterministicEmbeddings:
    """为每个测试提供一个全新的向量模型替身。"""
    return DeterministicEmbeddings()
