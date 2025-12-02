"""
🔍 ORACLE-X Local Vector Storage

Local vector database implementation using ChromaDB for storing and retrieving
trading scenarios and market intelligence. Uses SentenceTransformers for local embeddings.

Features:
- Local persistent storage (no external database required)
- Local embedding generation using SentenceTransformers (no API calls)
- Efficient similarity search with caching
- Batch operations for performance
- Automatic collection management

Usage:
    from vector_db.local_store import VectorStore

    store = VectorStore()
    store.store_trade_vector(trade)
    similar = store.query_similar("AAPL bullish momentum trade")
"""

import os
import uuid
import time
from typing import List, Dict, Optional, Any
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError(
        "ChromaDB is required for local vector storage. "
        "Install with: pip install chromadb>=0.4.0"
    )

from core.config import config

# Force SentenceTransformers to use PyTorch backend and silence noisy TF/absl logs
os.environ.setdefault('SENTENCE_TRANSFORMERS_BACKEND', 'torch')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('ABSL_MIN_LOG_LEVEL', '1')

# ========================== CONFIGURATION ==========================

# Vector DB configuration
STORAGE_PATH = str(config.vector_db.get_full_path())
COLLECTION_NAME = config.vector_db.collection_name
DISABLE_VECTOR_DB = os.environ.get("ORACLEX_DISABLE_VECTOR_DB", "").strip().lower() in {"1", "true", "yes"}

# Lazy-loaded embedding dependencies
SentenceTransformer = None  # populated on first use
EMBEDDING_MODEL: Optional[Any] = None


class VectorStore:
    """Encapsulated vector store with ChromaDB backend."""

    def __init__(self):
        self.chroma_client: Optional[Any] = None
        self.collection: Optional[Any] = None
        self.embed_cache: Dict[str, List[float]] = {}
        self.query_cache: Dict[tuple, List[Dict]] = {}
        self._chroma_legacy_mode = False
        self._disabled_notice_shown = False

        if not DISABLE_VECTOR_DB:
            self._load_embedding_model()
            self._initialize_client()
            self._ensure_collection()

    def _mark_disabled(self, message: str) -> None:
        """Print a single warning when vector DB features are unavailable."""
        if not self._disabled_notice_shown:
            print(f"[WARN] {message}")
            self._disabled_notice_shown = True

    def _load_embedding_model(self) -> None:
        """Load the SentenceTransformer model lazily."""
        global SentenceTransformer, EMBEDDING_MODEL

        if EMBEDDING_MODEL is not None:
            return

        if not self._ensure_transformers_tf_stub():
            return

        try:
            from sentence_transformers import SentenceTransformer as ST
            SentenceTransformer = ST
        except Exception as e:
            self._disable_vector_db(f"SentenceTransformers unavailable; vector DB disabled ({e})")
            return

        try:
            print("🔧 Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
            EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            EMBEDDING_MODEL.max_seq_length = 256
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            self._disable_vector_db(f"Embedding model load failed; vector DB disabled ({e})")

    def _ensure_transformers_tf_stub(self) -> bool:
        """Provide a minimal stub for transformers compatibility."""
        try:
            import transformers
        except Exception as e:
            self._disable_vector_db(f"Transformers unavailable for embeddings: {e}")
            return False

        if not hasattr(transformers, "TFPreTrainedModel"):
            class _DummyTFPreTrainedModel:
                pass
            transformers.TFPreTrainedModel = _DummyTFPreTrainedModel
        return True

    def _disable_vector_db(self, reason: str) -> None:
        """Globally disable vector DB operations."""
        global DISABLE_VECTOR_DB
        DISABLE_VECTOR_DB = True
        self.chroma_client = None
        self.collection = None
        EMBEDDING_MODEL = None
        self._mark_disabled(reason)

    def _initialize_client(self) -> None:
        """Initialize ChromaDB client with fallbacks."""
        if DISABLE_VECTOR_DB:
            self._mark_disabled("ChromaDB disabled via ORACLEX_DISABLE_VECTOR_DB environment flag.")
            return

        storage_dir = Path(STORAGE_PATH)
        storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory mode for tests/pipelines
        if self._should_use_inmemory():
            try:
                print("[INFO] Using in-memory ChromaDB")
                self.chroma_client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
                return
            except Exception as e:
                print(f"[WARN] In-memory ChromaDB failed, falling back to persistent: {e}")

        # Persistent client with retry
        last_error = None
        for attempt in range(3):
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=str(storage_dir),
                    settings=Settings(anonymized_telemetry=False, allow_reset=True)
                )
                return
            except Exception as e:
                last_error = e
                print(f"[WARN] Persistent ChromaDB attempt {attempt + 1} failed: {e}")
                if attempt == 0:
                    self._cleanup_locks(storage_dir)
                if attempt == 2:
                    self._try_legacy_client(storage_dir, last_error=last_error)

        if self.chroma_client is None:
            self._disable_vector_db("ChromaDB unavailable; vector similarity features disabled.")

    def _should_use_inmemory(self) -> bool:
        """Check whether to force in-memory mode."""
        flag = os.environ.get("ORACLEX_INMEMORY_VECTOR_DB", "").strip().lower()
        return flag in ("1", "true", "yes", "pipeline")

    def _cleanup_locks(self, storage_dir: Path) -> None:
        """Clean up stale ChromaDB locks."""
        lock_patterns = ["*.lock", "*.lck", "chroma.sqlite3-wal", "chroma.sqlite3-shm"]
        for pattern in lock_patterns:
            for lock_file in storage_dir.glob(pattern):
                try:
                    if lock_file.stat().st_mtime < (time.time() - 300):
                        lock_file.unlink()
                        print(f"[INFO] Removed stale lock file: {lock_file.name}")
                except Exception as e:
                    print(f"[WARN] Could not remove lock file {lock_file.name}: {e}")

    def _try_legacy_client(self, storage_dir: Path, last_error: Optional[Exception] = None) -> None:
        """Fallback to legacy Chroma client."""
        try:
            settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(storage_dir),
                anonymized_telemetry=False,
                allow_reset=True
            )
            self.chroma_client = chromadb.Client(settings=settings)
            self._chroma_legacy_mode = True
            print("[WARN] Falling back to legacy Chroma client")
        except Exception as e:
            if last_error:
                print(f"[ERROR] Persistent client failed: {last_error}")
            self._disable_vector_db("ChromaDB unavailable; vector similarity features disabled.")

    def _ensure_collection(self) -> None:
        """Ensure collection exists with correct dimensions."""
        if self.chroma_client is None:
            return

        try:
            self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
            # Check dimensions if collection has data
            result = self.collection.peek(limit=1)
            if result and result.get('embeddings'):
                existing_dim = len(result['embeddings'][0])
                if existing_dim != 384:
                    self._reset_collection()
        except Exception:
            self._create_collection()

    def _create_collection(self) -> None:
        """Create new collection."""
        if self.chroma_client is None:
            return
        try:
            self.collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "description": "ORACLE-X trading scenarios and intelligence",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "embedding_dimensions": "384"
                }
            )
            print(f"✅ Created collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"[ERROR] Collection creation failed: {e}")
            self.collection = None

    def _reset_collection(self) -> None:
        """Reset collection for new embeddings."""
        if self.chroma_client is None:
            return
        try:
            self.chroma_client.delete_collection(name=COLLECTION_NAME)
            print(f"🗑️ Deleted old collection: {COLLECTION_NAME}")
            self._create_collection()
        except Exception as e:
            print(f"[ERROR] Collection reset failed: {e}")

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of texts with shared caching logic."""
        if EMBEDDING_MODEL is None:
            return [[] for _ in texts]

        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached: List[str] = []
        uncached_indices: List[int] = []

        for idx, text in enumerate(texts):
            cached = self.embed_cache.get(text)
            if cached is not None:
                results[idx] = cached
            else:
                uncached.append(text)
                uncached_indices.append(idx)

        if uncached:
            try:
                embeddings = EMBEDDING_MODEL.encode(uncached, normalize_embeddings=True)
                for idx, embedding in zip(uncached_indices, embeddings):
                    embedding_list = embedding.tolist()
                    results[idx] = embedding_list
                    self.embed_cache[texts[idx]] = embedding_list
            except Exception as e:
                print(f"[ERROR] Embedding failed: {e}")
                for idx in uncached_indices:
                    results[idx] = []

        return [vec if vec is not None else [] for vec in results]

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using local model."""
        vectors = self._encode_texts([text])
        return vectors[0] if vectors else []

    def batch_embed_text(self, texts: List[str]) -> List[List[float]]:
        """Batch embed texts with caching."""
        return self._encode_texts(texts)

    def store_trade_vector(self, trade: dict) -> bool:
        """Store a trade vector."""
        if self.collection is None:
            return False

        required_keys = ("ticker", "thesis", "scenario_tree", "counter_signal")
        if not all(k in trade for k in required_keys):
            print(f"[ERROR] Trade missing required keys")
            return False

        text = f"{trade['ticker']} {trade['thesis']} {trade['scenario_tree']} {trade['counter_signal']}"
        vector = self.embed_text(text)
        if not vector:
            return False

        metadata = {
            "ticker": str(trade.get("ticker", "")),
            "direction": str(trade.get("direction", "")),
            "thesis": str(trade.get("thesis", ""))[:1000],
            "date": str(trade.get("date", "unknown"))
        }

        point_id = str(uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"{trade.get('ticker', 'UNK')}-{trade.get('date', 'unknown')}-{trade.get('direction', '')}"
        ))

        try:
            self.collection.upsert(ids=[point_id], embeddings=[vector], metadatas=[metadata])
            print(f"✅ Stored {trade.get('ticker', 'UNKNOWN')} vector")
            return True
        except Exception as e:
            print(f"[ERROR] Upsert failed: {e}")
            return False

    def query_similar(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query for similar vectors."""
        if self.collection is None:
            return []

        cache_key = (text, top_k)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        vector = self.embed_text(text)
        if not vector:
            return []

        try:
            results = self.collection.query(query_embeddings=[vector], n_results=top_k)
            formatted = self._format_query_results(results)
            self.query_cache[cache_key] = formatted
            return formatted
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return []

    def _format_query_results(self, results) -> List[Dict[str, Any]]:
        """Format ChromaDB results."""
        if not results or not results.get('ids') or not results['ids'][0]:
            return []

        ids_list = results['ids'][0]
        distances = results.get('distances', [[]])
        distances_list = distances[0] if distances else []
        metadatas = results.get('metadatas', [[]])
        metadatas_list = metadatas[0] if metadatas else []

        formatted = []
        for i in range(len(ids_list)):
            score = 1.0 - distances_list[i] if i < len(distances_list) else 1.0
            payload = metadatas_list[i] if i < len(metadatas_list) else {}
            formatted.append({'id': ids_list[i], 'score': score, 'payload': payload})
        return formatted

    def batch_query_similar(self, texts: List[str], top_k: int = 3) -> List[List[Dict[str, Any]]]:
        """Batch query for similar vectors."""
        return [self.query_similar(text, top_k) for text in texts]

    def clear_cache(self) -> None:
        """Clear caches."""
        self.embed_cache.clear()
        self.query_cache.clear()
        print("✅ Cleared caches")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if self.collection is None:
            return {
                'name': COLLECTION_NAME,
                'storage_path': STORAGE_PATH,
                'total_vectors': 0,
                'cache_size': len(self.embed_cache),
                'query_cache_size': len(self.query_cache),
                'status': 'disabled',
                'mode': 'legacy' if self._chroma_legacy_mode else 'persistent'
            }

        try:
            count = self.collection.count()
            return {
                'name': COLLECTION_NAME,
                'storage_path': STORAGE_PATH,
                'total_vectors': count,
                'cache_size': len(self.embed_cache),
                'query_cache_size': len(self.query_cache),
                'mode': 'legacy' if self._chroma_legacy_mode else 'persistent'
            }
        except Exception as e:
            print(f"[ERROR] Stats failed: {e}")
            return {}

    def reset_collection(self) -> None:
        """Reset the collection."""
        if self.chroma_client is None:
            print("[WARN] Client unavailable")
            return

        try:
            self.chroma_client.delete_collection(name=COLLECTION_NAME)
            print(f"✅ Deleted collection: {COLLECTION_NAME}")
            self._create_collection()
            self.clear_cache()
        except Exception as e:
            print(f"[ERROR] Reset failed: {e}")


# ========================== LEGACY COMPATIBILITY ==========================

# Global instance for backward compatibility
_vector_store = VectorStore()

def embed_text(text: str) -> List[float]:
    return _vector_store.embed_text(text)

def batch_embed_text(texts: List[str]) -> List[List[float]]:
    return _vector_store.batch_embed_text(texts)

def store_trade_vector(trade: dict) -> bool:
    return _vector_store.store_trade_vector(trade)

def query_similar(text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    return _vector_store.query_similar(text, top_k)

def batch_query_similar(texts: List[str], top_k: int = 3) -> List[List[Dict[str, Any]]]:
    return _vector_store.batch_query_similar(texts, top_k)

def ensure_collection():
    _vector_store._ensure_collection()

def clear_cache():
    _vector_store.clear_cache()

def get_collection_stats() -> Dict[str, Any]:
    return _vector_store.get_collection_stats()

def reset_collection():
    _vector_store.reset_collection()

collection = _vector_store.collection

__all__ = [
    'embed_text',
    'batch_embed_text',
    'store_trade_vector',
    'query_similar',
    'batch_query_similar',
    'ensure_collection',
    'clear_cache',
    'get_collection_stats',
    'reset_collection',
    'collection',
    'COLLECTION_NAME',
]
