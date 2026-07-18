"""Backward-compatible configuration import."""

from hybrid_rag.schemas import RetrievalConfig


DEFAULT_CONFIG = RetrievalConfig()
RAGConfig = RetrievalConfig

__all__ = ["DEFAULT_CONFIG", "RAGConfig", "RetrievalConfig"]

