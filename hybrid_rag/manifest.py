"""记录索引版本信息，并判断文档或建索引配置是否已经变化。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import KnowledgeDocument, RetrievalConfig


# manifest 自身的结构版本，未来字段变化时可据此做兼容处理。
SCHEMA_VERSION = 1


@dataclass(slots=True)
class IndexManifest:
    """一次完整索引构建的摘要信息。"""

    schema_version: int
    version: str
    built_at: str
    corpus_fingerprint: str
    build_fingerprint: str
    document_count: int
    chunk_count: int

    def save(self, path: str | Path) -> None:
        """把 manifest 以便于人工阅读的 JSON 格式写入磁盘。"""
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "IndexManifest":
        """从磁盘读取并恢复 manifest。"""
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def create_manifest(
    documents: list[KnowledgeDocument],
    config: RetrievalConfig,
    chunk_count: int,
) -> IndexManifest:
    """根据当前文档和配置创建一个新的索引版本描述。"""
    corpus = corpus_fingerprint(documents)
    build = build_fingerprint(config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return IndexManifest(
        schema_version=SCHEMA_VERSION,
        version=f"{timestamp}-{corpus[:8]}",
        built_at=datetime.now(timezone.utc).isoformat(),
        corpus_fingerprint=corpus,
        build_fingerprint=build,
        document_count=len(documents),
        chunk_count=chunk_count,
    )


def corpus_fingerprint(documents: list[KnowledgeDocument]) -> str:
    """计算语料指纹；文档内容或元数据变化都会得到不同结果。"""
    # 先按 ID 排序，使指纹不受上游返回文档顺序的影响。
    values = [
        {
            "id": document.id,
            "title": document.title,
            "content": document.content,
            "path": document.path,
            "metadata": document.metadata,
        }
        for document in sorted(documents, key=lambda item: item.id)
    ]
    return _json_hash(values)


def build_fingerprint(config: RetrievalConfig) -> str:
    """计算建索引配置指纹，查询阶段参数不会包含在其中。"""
    return _json_hash(config.build_settings())


def _json_hash(value: Any) -> str:
    """先稳定序列化为 JSON，再计算 SHA-256。"""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
