"""Versioned index metadata and staleness detection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import KnowledgeDocument, RetrievalConfig


SCHEMA_VERSION = 1


@dataclass(slots=True)
class IndexManifest:
    schema_version: int
    version: str
    built_at: str
    corpus_fingerprint: str
    build_fingerprint: str
    document_count: int
    chunk_count: int

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "IndexManifest":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def create_manifest(
    documents: list[KnowledgeDocument],
    config: RetrievalConfig,
    chunk_count: int,
) -> IndexManifest:
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
    return _json_hash(config.build_settings())


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

