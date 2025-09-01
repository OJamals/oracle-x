import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from openai import OpenAI

from typing import List, Dict

# Load environment variables from .env if present
load_dotenv()

# Set up your local Qdrant client
qdrant = QdrantClient(
    url="http://localhost:6333",
    api_key=os.environ.get("QDRANT_API_KEY", "your-super-secret-qdrant-api-key")
)


import config_manager  # project-level configuration module

# Embedding configuration (falls back gracefully)
EMBEDDING_MODEL = config_manager.get_embedding_model() if hasattr(config_manager, 'get_embedding_model') else os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_API_BASE = config_manager.get_embedding_api_base() if hasattr(config_manager, 'get_embedding_api_base') else os.environ.get("EMBEDDING_API_BASE")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("EMBEDDING_API_KEY")

client = OpenAI(api_key=API_KEY, base_url=EMBEDDING_API_BASE or config_manager.get_openai_api_base())

COLLECTION_NAME = "oraclex_trades"

# Simple in-memory cache for embeddings and queries
EMBED_CACHE = {}
QUERY_CACHE = {}

def ensure_collection():
    """
    Ensure the Qdrant collection exists, create if missing.
    """
    try:
        if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(size=1024, distance=qmodels.Distance.COSINE)
            )
    except Exception as e:
        print(f"[ERROR] Qdrant collection setup failed: {e}")

def embed_text(text: str) -> list:
    """
    Use Qwen3 embedding server (OpenAI compatible).
    Args:
        text (str): Text to embed.
    Returns:
        list: Embedding vector.
    """
    try:
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return resp.data[0].embedding  # type: ignore[attr-defined]
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return []

# Batch embedding

def batch_embed_text(texts: List[str]) -> List[list]:
    """
    Batch embed a list of texts using Qwen3 embedding server.
    Returns a list of embedding vectors.
    """
    results = []
    uncached = []
    uncached_indices = []
    for i, text in enumerate(texts):
        if text in EMBED_CACHE:
            results.append(EMBED_CACHE[text])
        else:
            results.append(None)
            uncached.append(text)
            uncached_indices.append(i)
    if uncached:
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=uncached
            )
            for idx, emb in zip(uncached_indices, resp.data):  # type: ignore[attr-defined]
                results[idx] = emb.embedding  # type: ignore[attr-defined]
                EMBED_CACHE[uncached[idx]] = emb.embedding  # type: ignore[attr-defined]
        except Exception as e:
            print(f"[ERROR] Batch embedding failed: {e}")
            for idx in uncached_indices:
                results[idx] = []
    return results

def store_trade_vector(trade: dict) -> bool:
    """Store a single trade vector in Qdrant.

    Returns True only if an embedding was generated AND the upsert succeeded.
    Never raises; logs errors and returns False on any failure.
    """
    required_keys = ("ticker", "thesis", "scenario_tree", "counter_signal")
    for k in required_keys:
        if k not in trade:
            print(f"[ERROR] Trade missing key '{k}' – cannot embed.")
            return False
    text = f"{trade['ticker']} {trade['thesis']} {trade['scenario_tree']} {trade['counter_signal']}"
    vector = embed_text(text)
    if not vector:
        print(f"[ERROR] No vector generated for {trade.get('ticker','UNKNOWN')}")
        return False
    payload = {
        "ticker": trade.get("ticker"),
        "direction": trade.get("direction"),
        "thesis": trade.get("thesis"),
        "date": trade.get("date", "unknown")
    }
    import uuid
    try:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{trade.get('ticker','UNK')}-{trade.get('date', 'unknown')}-{trade.get('direction', '')}"))
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[qmodels.PointStruct(id=point_id, vector=vector, payload=payload)]
        )
        print(f"✅ Stored {trade.get('ticker','UNKNOWN')} vector to Qdrant.")
        return True
    except Exception as e:
        print(f"[ERROR] Qdrant upsert failed: {e}")
        return False

# Batch query for similar vectors

def batch_query_similar(texts: List[str], top_k: int = 3) -> List[List[Dict]]:
    """
    Batch query Qdrant for similar vectors for a list of texts.
    Returns a list of lists of Qdrant search results.
    """
    vectors = batch_embed_text(texts)
    results = []
    for text, vector in zip(texts, vectors):
        cache_key = (text, top_k)
        if cache_key in QUERY_CACHE:
            results.append(QUERY_CACHE[cache_key])
            continue
        if not vector:
            results.append([])
            continue
        try:
            res = qdrant.search(
                collection_name=COLLECTION_NAME, query_vector=vector, limit=top_k
            )
            QUERY_CACHE[cache_key] = res
            results.append(res)
        except Exception as e:
            print(f"[ERROR] Qdrant batch search failed: {e}")
            results.append([])
    return results

# Update query_similar to use cache

def query_similar(text: str, top_k: int = 3):
    cache_key = (text, top_k)
    if cache_key in QUERY_CACHE:
        return QUERY_CACHE[cache_key]
    vector = embed_text(text)
    if not vector:
        return []
    try:
        res = qdrant.search(
            collection_name=COLLECTION_NAME, query_vector=vector, limit=top_k
        )
        QUERY_CACHE[cache_key] = res
        return res
    except Exception as e:
        print(f"[ERROR] Qdrant search failed: {e}")
        return []

# Initialize collection on module import
ensure_collection()
