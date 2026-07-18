from hybrid_rag.bm25_index import BM25Index
from hybrid_rag.schemas import KnowledgeChunk
from hybrid_rag.tokenization import tokenize_zh


def _chunk(chunk_id: str, content: str, category: str = "support") -> KnowledgeChunk:
    return KnowledgeChunk(
        id=chunk_id,
        document_id=chunk_id,
        title=content,
        content=content,
        heading_path=(),
        chunk_index=0,
        metadata={"category": category},
    )


def test_chinese_tokenizer_keeps_error_codes_and_chinese_bigrams():
    tokens = tokenize_zh("客户端出现 111/500002 错误并频繁掉线")

    assert "111/500002" in tokens
    assert "掉线" in tokens
    assert "错误" in tokens


def test_bm25_retrieves_chinese_terms_and_applies_filters():
    index = BM25Index(
        [
            _chunk("disconnect", "加速器连接频繁掉线，请更换节点"),
            _chunk("download", "客户端下载和安装教程"),
            _chunk("other", "掉线测试文档", category="other"),
        ]
    )

    results = index.search("游戏总是掉线", limit=2, filters={"category": "support"})

    assert results[0][0].id == "disconnect"
    assert all(chunk.metadata["category"] == "support" for chunk, _ in results)

