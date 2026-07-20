"""对外公开的混合检索接口，调用方通常只需要从这里导入。"""

from .chunking import MarkdownChunker
from .loaders import MarkdownDirectoryLoader
from .schemas import (
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievalConfig,
    RetrievalResult,
)
from .service import HybridRetrievalService
from .tokenization import (
    BM25TokenizerMode,
    JiebaBM25Tokenizer,
    create_bm25_tokenizer,
    tokenize_zh,
)

# __all__ 明确包的公共 API，内部辅助函数不会被误当成稳定接口使用。
__all__ = [
    "HybridRetrievalService",
    "BM25TokenizerMode",
    "JiebaBM25Tokenizer",
    "KnowledgeChunk",
    "KnowledgeDocument",
    "MarkdownChunker",
    "MarkdownDirectoryLoader",
    "RetrievalConfig",
    "RetrievalResult",
    "create_bm25_tokenizer",
    "tokenize_zh",
]
