# Vector Storage Migration Guide

## Overview

Oracle-X has migrated from Qdrant (external vector database) to a simple local vector storage system. This change eliminates external dependencies while maintaining full functionality.

## What Changed

### Before (Qdrant-based)
- **Storage**: External Qdrant vector database (required separate service)
- **Dependencies**: `qdrant-client` library
- **Configuration**: Required `QDRANT_API_KEY` and `QDRANT_URL`
- **Setup**: Required running Qdrant service on `localhost:6333`

### After (Local Storage)
- **Storage**: Local JSON/pickle files in `data/vector_store/`
- **Dependencies**: `numpy` (already included with pandas)
- **Configuration**: No additional configuration required
- **Setup**: Automatic - storage directory created on first use

## Technical Details

### Storage Format
- **Vectors**: Stored in `data/vector_store/vectors.pkl` (pickle format)
- **Metadata**: Stored in `data/vector_store/metadata.json` (JSON format)
- **Structure**: Each vector has ID, embedding vector, and payload (ticker, direction, thesis, date)

### Embedding Provider
- **Provider**: OpenAI-compatible API
- **Configuration**: Uses `OPENAI_API_BASE` from environment
- **Model**: Configured via `EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- **Fallback**: Uses `config_manager.get_openai_api_base()` if `EMBEDDING_API_BASE` not set

### Similarity Search
- **Algorithm**: Cosine similarity using NumPy
- **Performance**: In-memory similarity calculation
- **Caching**: Built-in embedding and query caching for performance

## Migration Steps

### For Existing Installations

1. **Update Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   This will install NumPy and remove Qdrant client.

2. **Remove Qdrant Configuration** (optional)
   Remove these lines from your `.env` file:
   ```bash
   QDRANT_API_KEY=...
   QDRANT_URL=...
   ```

3. **No Data Migration Required**
   - Old Qdrant data remains in your Qdrant instance (not automatically migrated)
   - New vectors will be stored locally going forward
   - System will rebuild vector database naturally as new trades are processed

4. **Verify Storage Directory**
   The directory `data/vector_store/` will be created automatically on first use.

### For New Installations

No special steps required! Just:
1. Install dependencies: `pip install -r requirements.txt`
2. Configure OpenAI API: Set `OPENAI_API_KEY` and optionally `OPENAI_API_BASE`
3. Run the system - vector storage is created automatically

## API Compatibility

### Unchanged Functions
All existing vector storage functions work exactly the same:

```python
from vector_db import qdrant_store

# Store a trade vector
qdrant_store.store_trade_vector(trade_dict)

# Query similar vectors
results = qdrant_store.query_similar(text, top_k=3)

# Batch operations
embeddings = qdrant_store.batch_embed_text(texts)
results = qdrant_store.batch_query_similar(texts, top_k=3)
```

### Result Format
Search results maintain Qdrant-compatible format:
```python
@dataclass
class SearchResult:
    id: str
    score: float  # Cosine similarity (0.0 to 1.0)
    payload: Dict[str, Any]
```

## Performance Characteristics

### Advantages
- ✅ No external service dependency
- ✅ Simpler deployment (one less service to manage)
- ✅ Portable (works anywhere Python works)
- ✅ Built-in caching for embeddings and queries
- ✅ Human-readable metadata (JSON format)

### Considerations
- ⚠️ Scales to thousands of vectors (sufficient for typical trading scenarios)
- ⚠️ For millions of vectors, consider implementing FAISS or Annoy backend
- ⚠️ In-memory similarity calculation (fast for typical use cases)

## Configuration Reference

### Required Environment Variables
```bash
# OpenAI API for embeddings
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1  # or your custom endpoint

# Embedding model (optional, defaults shown)
EMBEDDING_MODEL=text-embedding-3-small
```

### Optional Configuration
```bash
# Override embedding API base separately (optional)
EMBEDDING_API_BASE=https://your-custom-embedding-service.com/v1
```

## Testing

Run the migration test to verify everything works:

```bash
python test_vector_migration.py
```

Expected output:
```
✅ Module import successful
✅ Storage directory created successfully
✅ Embedding generated: 1536 dimensions
✅ Trade vector stored successfully
✅ Found 1 similar vectors
```

## Troubleshooting

### Issue: "No module named 'numpy'"
**Solution**: Install NumPy: `pip install numpy`

### Issue: Embedding failures
**Solution**: Check that `OPENAI_API_KEY` is set correctly and `OPENAI_API_BASE` points to a valid endpoint.

### Issue: Permission errors creating storage directory
**Solution**: Ensure write permissions for `data/vector_store/` directory.

### Issue: Search returns no results
**Solution**: Check that vectors have been stored first. Run a few trades through the system to populate the database.

## Future Enhancements

Potential improvements for scaling:
- Add FAISS backend for faster similarity search at scale
- Implement vector compression for storage efficiency
- Add vector database compaction/cleanup utilities
- Support for multiple collections/namespaces

## Support

For issues or questions:
1. Check this migration guide
2. Review the test script: `test_vector_migration.py`
3. Examine the implementation: `vector_db/qdrant_store.py`
4. Open an issue on GitHub

## Summary

The migration to local vector storage simplifies Oracle-X deployment while maintaining full functionality. No code changes are required for existing users - just update dependencies and the system handles the rest automatically.
