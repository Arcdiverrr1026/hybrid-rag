from hybrid_rag import KnowledgeDocument, MarkdownChunker, RetrievalConfig


def test_chunk_ids_are_stable_and_oversized_sections_are_split():
    document = KnowledgeDocument(
        id="doc-1",
        title="测试文档",
        content="# 主标题\n\n## 排查\n\n" + "连接异常。" * 80,
        metadata={"category": "support"},
    )
    chunker = MarkdownChunker(
        RetrievalConfig(chunk_max_chars=120, chunk_overlap_chars=20)
    )

    first = chunker.split([document])
    second = chunker.split([document])

    assert len(first) > 1
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert all(len(chunk.content) <= 120 for chunk in first)
    assert all(chunk.document_id == "doc-1" for chunk in first)
    assert first[0].heading_path == ("主标题", "排查")

