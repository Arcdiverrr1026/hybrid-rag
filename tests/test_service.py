from hybrid_rag import HybridRetrievalService, KnowledgeDocument, RetrievalConfig


def _documents() -> list[KnowledgeDocument]:
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

