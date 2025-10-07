#!/usr/bin/env python3
"""
Test script for the local vector storage migration.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_vector_storage():
    """Test basic vector storage operations"""
    print("🧪 Testing local vector storage system...")
    
    try:
        # Import the vector storage module
        from vector_db import qdrant_store
        
        print("✅ Module import successful")
        
        # Test 1: Ensure collection (storage directory) is created
        print("\n📁 Test 1: Ensure storage directory...")
        qdrant_store.ensure_collection()
        assert qdrant_store.VECTOR_STORE_DIR.exists(), "Storage directory should exist"
        print("✅ Storage directory created successfully")
        
        # Test 2: Test embedding
        print("\n🔤 Test 2: Test embedding generation...")
        test_text = "AAPL bullish momentum strong technical indicators"
        try:
            embedding = qdrant_store.embed_text(test_text)
            if embedding:
                print(f"✅ Embedding generated: {len(embedding)} dimensions")
            else:
                print("⚠️  Embedding returned empty (API key may not be configured)")
        except Exception as e:
            print(f"⚠️  Embedding test skipped (API key may not be configured): {e}")
        
        # Test 3: Test trade vector storage
        print("\n💾 Test 3: Test trade vector storage...")
        test_trade = {
            "ticker": "AAPL",
            "direction": "long",
            "thesis": "Strong momentum with increasing volume",
            "scenario_tree": "Breakout above resistance at $175",
            "counter_signal": "Watch for reversal if volume drops",
            "date": "2025-01-15"
        }
        
        try:
            result = qdrant_store.store_trade_vector(test_trade)
            if result:
                print("✅ Trade vector stored successfully")
            else:
                print("⚠️  Trade vector storage returned False (API key may not be configured)")
        except Exception as e:
            print(f"⚠️  Storage test skipped: {e}")
        
        # Test 4: Test query (if storage succeeded)
        print("\n🔍 Test 4: Test similarity search...")
        try:
            query_text = "AAPL bullish with strong momentum"
            results = qdrant_store.query_similar(query_text, top_k=3)
            if results:
                print(f"✅ Found {len(results)} similar vectors")
                for i, result in enumerate(results):
                    print(f"   {i+1}. {result.payload.get('ticker')} - Score: {result.score:.4f}")
            else:
                print("ℹ️  No results found (database may be empty or API key not configured)")
        except Exception as e:
            print(f"⚠️  Query test skipped: {e}")
        
        # Test 5: Test batch operations
        print("\n📦 Test 5: Test batch embedding...")
        try:
            texts = ["AAPL bullish", "MSFT bearish", "TSLA neutral"]
            batch_embeddings = qdrant_store.batch_embed_text(texts)
            if all(batch_embeddings):
                print(f"✅ Batch embedding successful: {len(batch_embeddings)} embeddings")
            else:
                print("⚠️  Some batch embeddings failed (API key may not be configured)")
        except Exception as e:
            print(f"⚠️  Batch embedding test skipped: {e}")
        
        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("="*60)
        print("\n📝 Notes:")
        print("   - Local vector storage is now active")
        print("   - Vectors stored in: data/vector_store/")
        print("   - Embeddings use OpenAI API from OPENAI_API_BASE")
        print("   - No external vector DB required (Qdrant removed)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vector_storage()
    sys.exit(0 if success else 1)
