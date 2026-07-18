"""Load source documents without coupling retrieval to a database or domain."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .schemas import KnowledgeDocument


class MarkdownDirectoryLoader:
    """Recursively load Markdown files from a directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load(self) -> list[KnowledgeDocument]:
        if not self.root.exists():
            raise FileNotFoundError(f"document directory does not exist: {self.root}")

        documents: list[KnowledgeDocument] = []
        for path in sorted(self.root.rglob("*.md")):
            content = path.read_text(encoding="utf-8")
            if not content.strip():
                continue
            relative_path = path.relative_to(self.root).as_posix()
            documents.append(
                KnowledgeDocument(
                    id=hashlib.sha256(relative_path.encode("utf-8")).hexdigest(),
                    title=_extract_title(content, path.stem),
                    content=content,
                    path=relative_path,
                    metadata={"source": relative_path},
                )
            )
        return documents


def _extract_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or fallback
    return fallback

