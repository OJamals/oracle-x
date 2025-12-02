"""Compatibility wrapper for legacy Qdrant interfaces.

The project migrated from a Qdrant-backed vector store to a local ChromaDB
implementation. Several modules (tests, CLI utilities, historical scripts)
still import ``vector_db.qdrant_store``.  Reintroducing a compatibility layer
allows those callers to operate without modification while delegating all real
work to ``vector_db.local_store``.

All public functions mirror the original signatures but forward to the Chroma
implementation.  Results are normalised into ``QdrantLikeHit`` instances so
existing code that expects ``hit.payload`` style access continues to work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from . import local_store as _local

__all__ = [
    "QdrantLikeHit",
    "ensure_collection",
    "embed_text",
    "batch_embed_text",
    "store_trade_vector",
    "query_similar",
    "batch_query_similar",
    "clear_cache",
    "get_collection_stats",
    "reset_collection",
]


@dataclass(frozen=True)
class QdrantLikeHit:
    """Lightweight container matching the original Qdrant response contract."""

    id: str
    score: float
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation for callers expecting mappings."""

        return {"id": self.id, "score": self.score, "payload": self.payload}


def ensure_collection():
    """Ensure the underlying Chroma collection exists."""

    return _local.ensure_collection()


def embed_text(text: str) -> List[float]:
    """Delegate embedding generation to the local store."""

    return _local.embed_text(text)


def batch_embed_text(texts: Sequence[str]) -> List[List[float]]:
    """Batch embed multiple texts via the local store implementation."""

    return _local.batch_embed_text(list(texts))


def store_trade_vector(trade: Dict[str, Any]) -> bool:
    """Persist a trade vector using the local store."""

    return _local.store_trade_vector(trade)


def _convert(results: Iterable[Dict[str, Any]]) -> List[QdrantLikeHit]:
    converted: List[QdrantLikeHit] = []
    for item in results:
        if isinstance(item, QdrantLikeHit):
            converted.append(item)
            continue

        if not isinstance(item, dict):
            # Unexpected shape – skip but preserve compatibility by coercing.
            converted.append(QdrantLikeHit(id=str(item), score=0.0, payload={}))
            continue

        payload = item.get("payload") or item.get("metadata") or {}
        score = float(item.get("score", 0.0))
        identifier = str(item.get("id", payload.get("id", "")))
        converted.append(QdrantLikeHit(id=identifier, score=score, payload=payload))
    return converted


def query_similar(text: str, top_k: int = 3) -> List[QdrantLikeHit]:
    """Return similar trades in the legacy Qdrant response format."""

    results = _local.query_similar(text, top_k)
    return _convert(results)


def batch_query_similar(texts: Sequence[str], top_k: int = 3) -> List[List[QdrantLikeHit]]:
    """Batch variant mirroring the Qdrant API."""

    batched_results = _local.batch_query_similar(list(texts), top_k)
    return [_convert(batch) for batch in batched_results]


def clear_cache() -> None:
    """Expose cache clearing for compatibility."""

    _local.clear_cache()


def get_collection_stats() -> Dict[str, Any]:
    """Fetch collection statistics from the local store."""

    return _local.get_collection_stats()


def reset_collection() -> None:
    """Reset the underlying collection."""

    _local.reset_collection()
