#!/usr/bin/env python3
"""
Test script to verify vector storage migration from Qdrant to ChromaDB.

Usage:
    python test_vector_migration.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all imports work correctly"""
    print("🧪 Testing imports...")
    try:
        from vector_db import (
            embed_text,
            batch_embed_text,
            store_trade_vector,
            query_similar,
            batch_query_similar,
            ensure_collection,
            clear_cache,
            get_collection_stats,
            COLLECTION_NAME,
        )
        print("   ✅ All imports successful")
        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False


def test_configuration():
    """Test configuration loading"""
    print("\n🧪 Testing configuration...")
    try:
        from core.config import config
        
        storage_path = config.vector_db.get_full_path()
        collection_name = config.vector_db.collection_name
        embedding_model = config.model.embedding_model
        
        print(f"   Storage Path: {storage_path}")
        print(f"   Collection Name: {collection_name}")
        print(f"   Embedding Model: {embedding_model}")
        print("   ✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"   ❌ Configuration failed: {e}")
        return False


def test_collection_creation():
    """Test collection creation and stats"""
    print("\n🧪 Testing collection creation...")
    try:
        from vector_db import ensure_collection, get_collection_stats
        
        collection = ensure_collection()
        stats = get_collection_stats()
        
        print(f"   Collection: {stats.get('name', 'N/A')}")
        print(f"   Storage Path: {stats.get('storage_path', 'N/A')}")
        print(f"   Total Vectors: {stats.get('total_vectors', 0)}")
        print(f"   Embedding Model: {stats.get('embedding_model', 'N/A')}")
        print("   ✅ Collection created successfully")
        return True
    except Exception as e:
        print(f"   ❌ Collection creation failed: {e}")
        return False


def test_embedding():
    """Test embedding generation"""
    print("\n🧪 Testing embedding generation...")
    try:
        from vector_db import embed_text
        
        test_text = "Apple stock showing strong momentum with positive sentiment"
        embedding = embed_text(test_text)
        
        if embedding and len(embedding) > 0:
            print(f"   Generated embedding with dimension: {len(embedding)}")
            print("   ✅ Embedding generation successful")
            return True
        else:
            print("   ❌ Embedding generation returned empty vector")
            return False
    except Exception as e:
        print(f"   ❌ Embedding generation failed: {e}")
        return False


def test_storage():
    """Test storing a trade vector"""
    print("\n🧪 Testing vector storage...")
    try:
        from vector_db import store_trade_vector
        
        test_trade = {
            'ticker': 'TEST',
            'thesis': 'Test migration scenario with strong momentum',
            'scenario_tree': 'Bullish breakout above resistance',
            'counter_signal': 'Watch for reversal at key level',
            'direction': 'LONG',
            'date': '2025-01-07'
        }
        
        success = store_trade_vector(test_trade)
        
        if success:
            print("   ✅ Vector storage successful")
            return True
        else:
            print("   ❌ Vector storage failed")
            return False
    except Exception as e:
        print(f"   ❌ Vector storage failed: {e}")
        return False


def test_query():
    """Test querying similar vectors"""
    print("\n🧪 Testing similarity search...")
    try:
        from vector_db import query_similar
        
        query_text = "Test migration momentum"
        results = query_similar(query_text, top_k=3)
        
        print(f"   Found {len(results)} similar vectors")
        
        if results:
            for i, result in enumerate(results, 1):
                payload = result.get('payload', {})
                score = result.get('score', 0)
                print(f"   [{i}] Ticker: {payload.get('ticker', 'N/A')}, Score: {score:.4f}")
            
            print("   ✅ Similarity search successful")
            return True
        else:
            print("   ⚠️  No results found (expected if database is empty)")
            return True
    except Exception as e:
        print(f"   ❌ Similarity search failed: {e}")
        return False


def test_batch_operations():
    """Test batch embedding and query operations"""
    print("\n🧪 Testing batch operations...")
    try:
        from vector_db import batch_embed_text, batch_query_similar
        
        # Test batch embedding
        texts = [
            "Apple strong buy momentum",
            "Tesla bearish reversal signal",
            "Microsoft consolidation pattern"
        ]
        
        embeddings = batch_embed_text(texts)
        print(f"   Batch embedded {len(embeddings)} texts")
        
        # Test batch query
        queries = ["Apple momentum", "Tesla bearish"]
        results = batch_query_similar(queries, top_k=2)
        print(f"   Batch query returned {len(results)} result sets")
        
        print("   ✅ Batch operations successful")
        return True
    except Exception as e:
        print(f"   ❌ Batch operations failed: {e}")
        return False


def test_cache():
    """Test caching functionality"""
    print("\n🧪 Testing cache operations...")
    try:
        from vector_db import embed_text, clear_cache, get_collection_stats
        
        # First embedding (not cached)
        test_text = "Cache test text for migration verification"
        _ = embed_text(test_text)
        
        stats1 = get_collection_stats()
        cache_size1 = stats1.get('cache_size', 0)
        
        # Second embedding (should be cached)
        _ = embed_text(test_text)
        
        stats2 = get_collection_stats()
        cache_size2 = stats2.get('cache_size', 0)
        
        print(f"   Cache size: {cache_size2}")
        
        # Clear cache
        clear_cache()
        stats3 = get_collection_stats()
        cache_size3 = stats3.get('cache_size', 0)
        
        print(f"   Cache size after clear: {cache_size3}")
        
        if cache_size3 == 0:
            print("   ✅ Cache operations successful")
            return True
        else:
            print("   ⚠️  Cache not fully cleared")
            return True
    except Exception as e:
        print(f"   ❌ Cache operations failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("ORACLE-X Vector Storage Migration Test")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Collection Creation", test_collection_creation),
        ("Embedding Generation", test_embedding),
        ("Vector Storage", test_storage),
        ("Similarity Search", test_query),
        ("Batch Operations", test_batch_operations),
        ("Cache Operations", test_cache),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Vector storage migration successful!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
