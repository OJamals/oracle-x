import os
import json
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Load environment variables from .env if present
load_dotenv()

import config_manager  # project-level configuration module

# Embedding configuration (falls back gracefully)
EMBEDDING_MODEL = config_manager.get_embedding_model() if hasattr(config_manager, 'get_embedding_model') else os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_API_BASE = config_manager.get_embedding_api_base() if hasattr(config_manager, 'get_embedding_api_base') else os.environ.get("EMBEDDING_API_BASE")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("EMBEDDING_API_KEY")

# Lazy-initialized client
_client = None

def _get_client():
    """Get or create OpenAI client (lazy initialization)"""
    global _client
    if _client is None:
        api_key = API_KEY or "dummy-key-for-testing"
        base_url = EMBEDDING_API_BASE or (config_manager.get_openai_api_base() if hasattr(config_manager, 'get_openai_api_base') else None)
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client

COLLECTION_NAME = "oraclex_trades"

# Local storage paths
VECTOR_STORE_DIR = Path("data/vector_store")
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
VECTORS_FILE = VECTOR_STORE_DIR / "vectors.pkl"
METADATA_FILE = VECTOR_STORE_DIR / "metadata.json"

# Simple in-memory cache for embeddings and queries
EMBED_CACHE = {}
QUERY_CACHE = {}

@dataclass
class VectorRecord:
    """Record structure for stored vectors"""
    id: str
    vector: List[float]
    payload: Dict[str, Any]

def _load_vectors() -> Dict[str, VectorRecord]:
    """Load vectors from local storage"""
    if not VECTORS_FILE.exists() or not METADATA_FILE.exists():
        return {}
    
    try:
        with open(VECTORS_FILE, 'rb') as f:
            vectors_data = pickle.load(f)
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        records = {}
        for record_id, data in metadata.items():
            if record_id in vectors_data:
                records[record_id] = VectorRecord(
                    id=record_id,
                    vector=vectors_data[record_id],
                    payload=data['payload']
                )
        return records
    except Exception as e:
        print(f"[ERROR] Failed to load vectors: {e}")
        return {}

def _save_vectors(records: Dict[str, VectorRecord]):
    """Save vectors to local storage"""
    try:
        vectors_data = {rid: rec.vector for rid, rec in records.items()}
        metadata = {rid: {'payload': rec.payload} for rid, rec in records.items()}
        
        with open(VECTORS_FILE, 'wb') as f:
            pickle.dump(vectors_data, f)
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save vectors: {e}")

def ensure_collection():
    """
    Ensure the local vector storage directory exists.
    """
    try:
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        # Initialize empty storage if doesn't exist
        if not VECTORS_FILE.exists():
            _save_vectors({})
    except Exception as e:
        print(f"[ERROR] Vector storage setup failed: {e}")

def embed_text(text: str) -> list:
    """
    Use OpenAI-compatible embedding server (from OPENAI_API_BASE).
    Args:
        text (str): Text to embed.
    Returns:
        list: Embedding vector.
    """
    try:
        client = _get_client()
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
    Batch embed a list of texts using OpenAI-compatible embedding server.
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
            client = _get_client()
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
    """Store a single trade vector in local storage.

    Returns True only if an embedding was generated AND the storage succeeded.
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
        
        # Load existing records
        records = _load_vectors()
        
        # Add or update record
        records[point_id] = VectorRecord(
            id=point_id,
            vector=vector,
            payload=payload
        )
        
        # Save updated records
        _save_vectors(records)
        
        print(f"✅ Stored {trade.get('ticker','UNKNOWN')} vector to local storage.")
        return True
    except Exception as e:
        print(f"[ERROR] Vector storage failed: {e}")
        return False

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    try:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    except Exception:
        return 0.0

@dataclass
class SearchResult:
    """Search result matching Qdrant's structure"""
    id: str
    score: float
    payload: Dict[str, Any]

def _search_similar(query_vector: List[float], top_k: int = 3) -> List[SearchResult]:
    """Search for similar vectors in local storage"""
    records = _load_vectors()
    
    if not records:
        return []
    
    # Calculate similarities
    similarities = []
    for record_id, record in records.items():
        similarity = _cosine_similarity(query_vector, record.vector)
        similarities.append((record_id, similarity, record.payload))
    
    # Sort by similarity (descending) and take top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_results = similarities[:top_k]
    
    # Return in Qdrant-compatible format
    return [
        SearchResult(id=rid, score=score, payload=payload)
        for rid, score, payload in top_results
    ]

# Batch query for similar vectors

def batch_query_similar(texts: List[str], top_k: int = 3) -> List[List[SearchResult]]:
    """
    Batch query local storage for similar vectors for a list of texts.
    Returns a list of lists of search results.
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
            res = _search_similar(vector, top_k)
            QUERY_CACHE[cache_key] = res
            results.append(res)
        except Exception as e:
            print(f"[ERROR] Vector search failed: {e}")
            results.append([])
    return results

# Update query_similar to use cache

def query_similar(text: str, top_k: int = 3) -> List[SearchResult]:
    cache_key = (text, top_k)
    if cache_key in QUERY_CACHE:
        return QUERY_CACHE[cache_key]
    vector = embed_text(text)
    if not vector:
        return []
    try:
        res = _search_similar(vector, top_k)
        QUERY_CACHE[cache_key] = res
        return res
    except Exception as e:
        print(f"[ERROR] Vector search failed: {e}")
        return []

# Initialize collection on module import
ensure_collection()
