import gc
import sys
import warnings

import pytest

from hybrid_rag.bm25_index import BM25Index
from hybrid_rag.schemas import KnowledgeChunk, RetrievalConfig
from hybrid_rag.tokenization import create_bm25_tokenizer, tokenize_zh
from main import parse_args


def _chunk(chunk_id: str, content: str, category: str = "support") -> KnowledgeChunk:
    """快速构造一个最小可检索分块。"""
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
    """验证错误码保持完整，同时中文能够生成常用双字组合。"""
    tokens = tokenize_zh("客户端出现 111/500002 错误并频繁掉线")

    assert "111/500002" in tokens
    assert "掉线" in tokens
    assert "错误" in tokens


def test_bm25_retrieves_chinese_terms_and_applies_filters():
    """验证 BM25 能召回中文关键词，并正确应用元数据过滤条件。"""
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "客户端出现 111/500002 错误并频繁掉线",
            [
                "客", "户", "端", "出", "现", "客户", "户端", "端出", "出现",
                "111/500002",
                "错", "误", "并", "频", "繁", "掉", "线",
                "错误", "误并", "并频", "频繁", "繁掉", "掉线",
            ],
        ),
        ("DNF v2.4.1 111/500002", ["dnf", "v2.4.1", "111/500002"]),
        ("节点-切换", ["节", "点", "节点", "切", "换", "切换"]),
        ("", []),
        ("掉线掉线", ["掉", "线", "掉", "线", "掉线", "线掉", "掉线"]),
        ("WiFi6网络", ["wifi6", "网", "络", "网络"]),
    ],
)
def test_ngram_remains_the_default_with_frozen_legacy_output(text, expected):
    """冻结旧版完整输出，避免工厂和实现同时变化时测试仍然通过。"""
    assert RetrievalConfig().bm25_tokenizer == "ngram"
    assert tokenize_zh(text) == expected
    assert create_bm25_tokenizer()(text) == expected


def test_jieba_uses_isolated_domain_terms_and_preserves_protected_tokens():
    tokenizer = create_bm25_tokenizer(
        "jieba",
        domain_terms=("灵缇星链协议",),
    )

    tokens = tokenizer("灵缇星链协议 v2.4.1 报错 111/500002")

    assert "灵缇星链协议" in tokens
    assert "v2.4.1" in tokens
    assert "111/500002" in tokens


def test_jieba_loads_user_dictionary(tmp_path):
    user_dict = tmp_path / "customer-service-terms.txt"
    user_dict.write_text("霜火加速协议 100000 n\n", encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        tokenizer = create_bm25_tokenizer("jieba", user_dict=user_dict)
        gc.collect()

    assert "霜火加速协议" in tokenizer("请检查霜火加速协议是否开启")


def test_hybrid_adds_chinese_bigram_fallback_without_splitting_error_codes():
    tokenizer = create_bm25_tokenizer("hybrid")

    tokens = tokenizer("甲乙丙丁出现 111/500002")

    assert "乙丙" in tokens
    assert "111/500002" in tokens
    assert tokens.count("111/500002") == 1


def test_invalid_tokenizer_mode_and_dictionary_configuration_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported bm25_tokenizer"):
        RetrievalConfig(bm25_tokenizer="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="require bm25_tokenizer"):
        RetrievalConfig(jieba_domain_terms=("灵缇加速器",))
    with pytest.raises(FileNotFoundError, match="user dictionary not found"):
        create_bm25_tokenizer("jieba", user_dict=tmp_path / "missing.txt")


def test_bm25_search_text_supports_string_and_list_tags():
    tokenizer = create_bm25_tokenizer("hybrid")
    chunks = [
        KnowledgeChunk(
            id="list-tags",
            document_id="list-tags",
            title="文档一",
            content="通用说明",
            heading_path=(),
            chunk_index=0,
            metadata={"tags": ["灵缇星链协议", "连接"]},
        ),
        KnowledgeChunk(
            id="string-tags",
            document_id="string-tags",
            title="文档二",
            content="通用说明",
            heading_path=(),
            chunk_index=0,
            metadata={"tags": "霜火加速协议 下载"},
        ),
        _chunk("unrelated", "完全无关的支付说明"),
    ]
    index = BM25Index(chunks, tokenizer)

    assert index.search("灵缇星链协议", 1)[0][0].id == "list-tags"
    assert index.search("霜火加速协议", 1)[0][0].id == "string-tags"


def test_cli_accepts_bm25_tokenizer_dictionary_and_domain_terms(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--bm25-tokenizer",
            "hybrid",
            "--jieba-user-dict",
            "terms.txt",
            "--jieba-domain-term",
            "灵缇加速器",
            "--jieba-domain-term",
            "节点切换",
        ],
    )

    args = parse_args()

    assert args.bm25_tokenizer == "hybrid"
    assert args.jieba_user_dict == "terms.txt"
    assert args.jieba_domain_term == ["灵缇加速器", "节点切换"]
