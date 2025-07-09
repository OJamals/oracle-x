import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI
from functools import lru_cache
from typing import List, Dict

# Load environment variables from .env if present
load_dotenv()

# Set up your local Qdrant client
qdrant = QdrantClient(
    url="http://localhost:6333",
    api_key=os.environ.get("QDRANT_API_KEY", "your-super-secret-qdrant-api-key")
)


# Always use Qwen3 embedding endpoint for embeddings
QWEN3_API_BASE = "http://localhost:8000/v1"
QWEN3_API_KEY = os.environ.get("QWEN3_API_KEY", "qwen3-local-key")
client = OpenAI(api_key=QWEN3_API_KEY, base_url=QWEN3_API_BASE)

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
                vectors_config={
                    "size": 1024,
                    "distance": "Cosine"
                }
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
            model="Qwen3-embedding",
            input=text
        )
        return resp.data[0].embedding
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
                model="Qwen3-embedding",
                input=uncached
            )
            for idx, emb in zip(uncached_indices, resp.data):
                results[idx] = emb.embedding
                EMBED_CACHE[uncached[idx]] = emb.embedding
        except Exception as e:
            print(f"[ERROR] Batch embedding failed: {e}")
            for idx in uncached_indices:
                results[idx] = []
    return results

def store_trade_vector(trade: dict) -> None:
    """
    Store a single trade with scenario & anomaly context.
    Args:
        trade (dict): Trade dictionary.
    """
    text = f"{trade['ticker']} {trade['thesis']} {trade['scenario_tree']} {trade['counter_signal']}"
    vector = embed_text(text)
    if not vector:
        print(f"[ERROR] No vector generated for {trade['ticker']}")
        return
    payload = {
        "ticker": trade["ticker"],
        "direction": trade["direction"],
        "thesis": trade["thesis"],
        "date": trade.get("date", "unknown")
    }
    import uuid
    try:
        # Use a UUID for the point ID to comply with Qdrant's requirements
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{trade['ticker']}-{trade.get('date', 'unknown')}-{trade.get('direction', '')}"))
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": payload
                }
            ]
        )
        print(f"✅ Stored {trade['ticker']} vector to Qdrant.")
    except Exception as e:
        print(f"[ERROR] Qdrant upsert failed: {e}")

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
