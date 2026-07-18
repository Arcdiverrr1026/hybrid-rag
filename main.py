"""Command-line example for the domain-neutral hybrid retriever."""

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
    parser = argparse.ArgumentParser(description="Build and query a local hybrid RAG index")
    parser.add_argument("--data", default="./data", help="Markdown directory")
    parser.add_argument("--index", default="./vector_index", help="Index directory")
    parser.add_argument("--query", help="Query to search for")
    parser.add_argument("--rebuild", action="store_true", help="Force a new index version")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=("hybrid", "vector", "bm25"), default="hybrid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    documents = MarkdownDirectoryLoader(args.data).load()
    config = RetrievalConfig(
        index_path=args.index,
        final_top_k=args.top_k,
        search_mode=args.mode,
    )
    service = HybridRetrievalService(config)

    if args.rebuild or service.is_stale(documents):
        manifest = service.build(documents)
        print(f"Built index {manifest.version}: {manifest.document_count} documents, {manifest.chunk_count} chunks")
    else:
        manifest = service.load()
        print(f"Loaded index {manifest.version}")

    if args.query:
        results = service.search(args.query)
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(service.status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

