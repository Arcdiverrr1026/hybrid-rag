"""兼容旧代码的配置入口，新代码可直接从 hybrid_rag 导入。"""

from hybrid_rag.schemas import RetrievalConfig


# 旧项目如果仍然导入 DEFAULT_CONFIG 或 RAGConfig，不需要立刻修改调用代码。
DEFAULT_CONFIG = RetrievalConfig()
RAGConfig = RetrievalConfig

__all__ = ["DEFAULT_CONFIG", "RAGConfig", "RetrievalConfig"]
