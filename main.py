"""命令行入口：演示如何构建、加载和查询本地混合检索索引。"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict

from hybrid_rag import (
    HybridRetrievalService,
    MarkdownDirectoryLoader,
    RetrievalConfig,
)


def parse_args() -> argparse.Namespace:
    """读取命令行参数，并限制检索模式只能是项目支持的三种模式。"""
    parser = argparse.ArgumentParser(description="Build and query a local hybrid RAG index")
    parser.add_argument("--data", default="./data", help="Markdown directory")
    parser.add_argument("--index", default="./vector_index", help="Index directory")
    parser.add_argument("--query", help="Query to search for")
    parser.add_argument("--rebuild", action="store_true", help="Force a new index version")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=("hybrid", "vector", "bm25"), default="hybrid")
    parser.add_argument(
        "--bm25-tokenizer",
        choices=("ngram", "jieba", "hybrid"),
        default="ngram",
        help="Chinese tokenizer used by BM25 (default: ngram)",
    )
    parser.add_argument("--jieba-user-dict", help="Optional Jieba user dictionary path")
    parser.add_argument(
        "--jieba-domain-term",
        action="append",
        default=[],
        help="Add a domain term to the isolated Jieba dictionary; may be repeated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # 第一步：把 Markdown 目录转换成统一的 KnowledgeDocument 列表。
    documents = MarkdownDirectoryLoader(args.data).load()

    # 第二步：将命令行参数转换成检索服务使用的配置对象。
    config = RetrievalConfig(
        index_path=args.index,
        final_top_k=args.top_k,
        search_mode=args.mode,
        bm25_tokenizer=args.bm25_tokenizer,
        jieba_user_dict=args.jieba_user_dict,
        jieba_domain_terms=tuple(args.jieba_domain_term),
    )
    service = HybridRetrievalService(config)

    # 文档或建索引参数发生变化时必须重建，否则直接加载上次的索引即可。
    if args.rebuild or service.is_stale(documents):
        manifest = service.build(documents)
        print(f"Built index {manifest.version}: {manifest.document_count} documents, {manifest.chunk_count} chunks")
    else:
        manifest = service.load()
        print(f"Loaded index {manifest.version}")

    # 传入 query 时执行检索；没有 query 时只输出当前索引状态。
    if args.query:
        results = service.search(args.query)
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(service.status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
