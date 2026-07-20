"""从文件系统加载知识文档，避免检索逻辑依赖具体数据库或业务模型。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .schemas import KnowledgeDocument


class MarkdownDirectoryLoader:
    """递归读取目录下的 Markdown 文件并转换为 KnowledgeDocument。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load(self) -> list[KnowledgeDocument]:
        """加载所有非空 Markdown 文件，返回顺序稳定的文档列表。"""
        if not self.root.exists():
            raise FileNotFoundError(f"document directory does not exist: {self.root}")

        documents: list[KnowledgeDocument] = []
        # 排序后再读取，保证同一目录每次构建的输入顺序一致。
        for path in sorted(self.root.rglob("*.md")):
            content = path.read_text(encoding="utf-8")
            if not content.strip():
                continue
            relative_path = path.relative_to(self.root).as_posix()
            documents.append(
                KnowledgeDocument(
                    # 使用相对路径生成稳定 ID，目录没有变化时文档 ID 不会变化。
                    id=hashlib.sha256(relative_path.encode("utf-8")).hexdigest(),
                    title=_extract_title(content, path.stem),
                    content=content,
                    path=relative_path,
                    metadata={"source": relative_path},
                )
            )
        return documents


def _extract_title(content: str, fallback: str) -> str:
    """优先使用第一个一级标题；没有一级标题时使用文件名。"""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or fallback
    return fallback
