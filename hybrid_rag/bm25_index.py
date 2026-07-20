"""基于确定性文档分块构建 BM25 关键词检索索引。"""

from __future__ import annotations

from collections.abc import Callable

from rank_bm25 import BM25Okapi

from .schemas import KnowledgeChunk
from .tokenization import tokenize_zh


class BM25Index:
    """保存分块及其 BM25 索引，并提供关键词检索和元数据过滤。"""

    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        tokenizer: Callable[[str], list[str]] = tokenize_zh,
    ):
        self.chunks = chunks
        self.tokenizer = tokenizer
        # 每个 chunk 对应 BM25 语料库中的一行，索引位置与 self.chunks 一一对应。
        corpus = [tokenizer(_search_text(chunk)) or [""] for chunk in chunks]
        self.index = BM25Okapi(corpus) if corpus else None

    def search(
        self,
        query: str,
        limit: int,
        filters: dict[str, object] | None = None,
    ) -> list[tuple[KnowledgeChunk, float]]:
        """返回按 BM25 分数从高到低排列的分块及其原始分数。"""
        if self.index is None:
            return []

        # get_scores 会为语料库中的每个分块计算一次相关性分数。
        scores = self.index.get_scores(self.tokenizer(query))
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
        results: list[tuple[KnowledgeChunk, float]] = []
        for index, score in ranked:
            chunk = self.chunks[index]
            # 先过滤再计数，确保最终返回数量尽量达到 limit。
            if _matches_filters(chunk, filters):
                results.append((chunk, float(score)))
                if len(results) >= limit:
                    break
        return results


def _search_text(chunk: KnowledgeChunk) -> str:
    """拼接标题、标题路径、标签和正文，让这些信息都参与关键词匹配。"""
    headings = " ".join(chunk.heading_path)
    tags = chunk.metadata.get("tags", "")
    if isinstance(tags, (list, tuple, set)):
        tags = " ".join(str(tag) for tag in tags)
    return f"{chunk.title}\n{headings}\n{tags}\n{chunk.content}"


def _matches_filters(chunk: KnowledgeChunk, filters: dict[str, object] | None) -> bool:
    """检查分块的用户元数据是否满足全部过滤条件。"""
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
