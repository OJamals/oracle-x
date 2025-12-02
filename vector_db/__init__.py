"""
🔍 ORACLE-X Vector Database Package

Local vector storage for trading intelligence and scenario retrieval.
Uses ChromaDB for persistent storage and OpenAI-compatible embeddings.

Main Components:
- local_store: Primary vector storage implementation
- qdrant_store: Legacy Qdrant implementation (deprecated)
"""

from .local_store import (
    embed_text,
    batch_embed_text,
    store_trade_vector,
    query_similar,
    batch_query_similar,
    ensure_collection,
    clear_cache,
    get_collection_stats,
    reset_collection,
    COLLECTION_NAME,
)

__all__ = [
    "embed_text",
    "batch_embed_text",
    "store_trade_vector",
    "query_similar",
    "batch_query_similar",
    "ensure_collection",
    "clear_cache",
    "get_collection_stats",
    "reset_collection",
    "COLLECTION_NAME",
]
