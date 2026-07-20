from hybrid_rag import (
    HybridRetrievalService,
    KnowledgeDocument,
    RetrievalConfig,
    tokenize_zh,
)


def _documents() -> list[KnowledgeDocument]:
    """构造两篇固定的测试文档，供服务层测试重复使用。"""
    return [
        KnowledgeDocument(
            id="disconnect",
            title="连接中断",
            content="# 网络问题\n\n## 掉线\n\n频繁掉线时请更换加速节点。",
            metadata={"category": "support", "tags": ["掉线", "节点"]},
        ),
        KnowledgeDocument(
            id="download",
            title="客户端下载",
            content="# 客户端\n\n## 下载\n\n请从官方网站下载客户端。",
            metadata={"category": "support", "tags": ["下载"]},
        ),
    ]


def test_build_load_search_and_staleness(tmp_path, embeddings):
    """覆盖建索引、重新加载、查询、过滤和过期检测这条完整主流程。"""
    config = RetrievalConfig(
        index_path=str(tmp_path / "index"),
        final_top_k=2,
        vector_candidate_k=2,
        bm25_candidate_k=2,
    )
    service = HybridRetrievalService(config, embeddings)
    manifest = service.build(_documents())

    assert manifest.document_count == 2
    assert service.status()["ready"] is True
    assert service.is_stale(_documents()) is False
    assert service.search("频繁掉线", mode="bm25")[0].document_id == "disconnect"

    reloaded = HybridRetrievalService(config, embeddings)
    reloaded.load()
    results = reloaded.search("频繁掉线", filters={"category": "support"})

    assert results
    assert any(result.bm25_rank is not None for result in results)
    changed = _documents()
    changed[0].content += "\n新增处理步骤。"
    assert reloaded.is_stale(changed) is True


def test_hybrid_falls_back_to_bm25_when_vector_search_fails(tmp_path, embeddings, monkeypatch):
    """验证混合模式在向量检索故障时仍可降级使用 BM25。"""
    config = RetrievalConfig(index_path=str(tmp_path / "index"))
    service = HybridRetrievalService(config, embeddings)
    service.build(_documents())

    def fail(*args, **kwargs):
        raise RuntimeError("vector unavailable")

    monkeypatch.setattr(service.vector_index, "search", fail)
    results = service.search("客户端下载", mode="hybrid")

    assert results
    assert results[0].bm25_rank is not None
    assert all(result.vector_rank is None for result in results)


def test_vector_text_uses_title_and_tags_but_returns_raw_content(tmp_path, embeddings):
    """验证向量文本包含增强信息，但最终结果仍返回原始正文和用户元数据。"""
    raw_content = "请切换节点后重新连接。"
    # 特意加入与内部字段同名的数据，用来验证用户元数据不会被覆盖。
    user_metadata = {
        "category": "support",
        "tags": ["DNF手游韩服", "111/500002"],
        "__hybrid_rag_internal__": "user-owned-value",
        "chunk_id": "also-user-owned",
    }
    document = KnowledgeDocument(
        id="headingless",
        title="DNF 111/500002 报错解决办法",
        content=raw_content,
        metadata=user_metadata,
    )
    string_tags_document = KnowledgeDocument(
        id="string-tags",
        title="客户端登录失败",
        content="请检查客户端版本。",
        metadata={"category": "support", "tags": "登录失败 客户端版本"},
    )
    config = RetrievalConfig(
        index_path=str(tmp_path / "index"),
        vector_candidate_k=2,
        bm25_candidate_k=2,
        final_top_k=2,
    )
    service = HybridRetrievalService(config, embeddings)

    service.build([document, string_tags_document])

    assert len(embeddings.document_texts) == 2
    vector_text = next(
        text for text in embeddings.document_texts if "DNF 111/500002 报错解决办法" in text
    )
    assert "DNF 111/500002 报错解决办法" in vector_text
    assert "DNF手游韩服" in vector_text
    assert "111/500002" in vector_text
    assert raw_content in vector_text
    assert any("登录失败 客户端版本" in text for text in embeddings.document_texts)

    results = service.search("111/500002", mode="vector")
    result = next(item for item in results if item.document_id == "headingless")
    assert result.content == raw_content
    assert result.metadata == user_metadata


def test_service_uses_configured_bm25_tokenizer_for_build_query_and_load(
    tmp_path, embeddings
):
    """BM25 建库、查询和重载必须共享完全相同的 tokenizer 配置。"""
    config = RetrievalConfig(
        index_path=str(tmp_path / "index"),
        bm25_tokenizer="hybrid",
        jieba_domain_terms=("灵缇星链协议",),
    )
    service = HybridRetrievalService(config, embeddings)
    service.build(_documents())

    assert service.status()["bm25_tokenizer"] == "hybrid"
    assert "灵缇星链协议" in service.bm25_index.tokenizer("灵缇星链协议")
    assert service.search("频繁掉线", mode="bm25")

    reloaded = HybridRetrievalService(config, embeddings)
    reloaded.load()

    assert reloaded.bm25_index is not None
    assert reloaded.bm25_index.tokenizer is reloaded.bm25_tokenizer
    assert "灵缇星链协议" in reloaded.bm25_index.tokenizer("灵缇星链协议")
    assert reloaded.search("频繁掉线", mode="bm25")


def test_service_keeps_custom_bm25_tokenizer_injection_compatible(tmp_path, embeddings):
    calls: list[str] = []

    def recording_tokenizer(text: str) -> list[str]:
        calls.append(text)
        return tokenize_zh(text)

    config = RetrievalConfig(index_path=str(tmp_path / "index"))
    service = HybridRetrievalService(config, embeddings, recording_tokenizer)
    service.build(_documents())
    calls.clear()

    service.search("频繁掉线", mode="bm25")

    assert calls == ["频繁掉线"]
    assert service.status()["bm25_tokenizer"] == "custom"
