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


def test_headingless_short_markdown_stays_in_one_chunk():
    content = "客户端无法连接时，请先切换加速节点，然后重新启动客户端。"
    document = KnowledgeDocument(id="short", title="连接问题", content=content)
    chunker = MarkdownChunker(
        RetrievalConfig(chunk_max_chars=120, chunk_overlap_chars=20)
    )

    chunks = chunker.split([document])

    assert len(chunks) == 1
    assert chunks[0].heading_path == ()
    assert chunks[0].content == content


def test_headingless_long_markdown_uses_length_split_and_overlap():
    content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10
    document = KnowledgeDocument(id="long", title="连续文本", content=content)
    chunker = MarkdownChunker(
        RetrievalConfig(chunk_max_chars=80, chunk_overlap_chars=20)
    )

    chunks = chunker.split([document])

    assert len(chunks) > 1
    assert all(chunk.heading_path == () for chunk in chunks)
    assert all(len(chunk.content) <= 80 for chunk in chunks)
    assert chunks[0].content[-20:] == chunks[1].content[:20]
