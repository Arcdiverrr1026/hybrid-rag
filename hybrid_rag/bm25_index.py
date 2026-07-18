"""BM25 index for deterministic chunks."""

from __future__ import annotations

from collections.abc import Callable

from rank_bm25 import BM25Okapi

from .schemas import KnowledgeChunk
from .tokenization import tokenize_zh


class BM25Index:
    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        tokenizer: Callable[[str], list[str]] = tokenize_zh,
    ):
        self.chunks = chunks
        self.tokenizer = tokenizer
        corpus = [tokenizer(_search_text(chunk)) or [""] for chunk in chunks]
        self.index = BM25Okapi(corpus) if corpus else None

    def search(
        self,
        query: str,
        limit: int,
        filters: dict[str, object] | None = None,
    ) -> list[tuple[KnowledgeChunk, float]]:
        if self.index is None:
            return []
        scores = self.index.get_scores(self.tokenizer(query))
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
        results: list[tuple[KnowledgeChunk, float]] = []
        for index, score in ranked:
            chunk = self.chunks[index]
            if _matches_filters(chunk, filters):
                results.append((chunk, float(score)))
                if len(results) >= limit:
                    break
        return results


def _search_text(chunk: KnowledgeChunk) -> str:
    headings = " ".join(chunk.heading_path)
    tags = chunk.metadata.get("tags", "")
    if isinstance(tags, (list, tuple, set)):
        tags = " ".join(str(tag) for tag in tags)
    return f"{chunk.title}\n{headings}\n{tags}\n{chunk.content}"


def _matches_filters(chunk: KnowledgeChunk, filters: dict[str, object] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = chunk.metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True

