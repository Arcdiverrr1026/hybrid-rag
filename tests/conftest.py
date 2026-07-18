from __future__ import annotations

import hashlib

import pytest
from langchain_core.embeddings import Embeddings


class DeterministicEmbeddings(Embeddings):
    """Small offline embedding used to exercise FAISS persistence."""

    dimensions = 16

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[index] / 255.0 for index in range(self.dimensions)]


@pytest.fixture
def embeddings() -> DeterministicEmbeddings:
    return DeterministicEmbeddings()

