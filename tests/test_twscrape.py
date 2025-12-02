#!/usr/bin/env python3
"""
Test twscrape functionality with available accounts
"""

import asyncio
from twscrape import API

async def test_twscrape():
    """Test twscrape with available accounts"""
    print("="*70)
    print("🐦 TESTING TWSCRAPE")
    print("="*70)
    
    api = API()
    
    # Check accounts
    print("\n📋 Checking available accounts...")
    accounts = await api.pool.accounts_info()
    
    print(f"   Found {len(accounts)} accounts")
    for i, acc in enumerate(accounts[:5], 1):  # Show first 5
        print(f"   {i}. @{acc.username} - Status: {acc.active}")
    
    if len(accounts) == 0:
        print("\n❌ No accounts found")
        print("   Run: twscrape add_accounts accounts.txt")
        return False
    
    # Test search
    print("\n🔍 Testing search for 'AAPL'...")
    tweets = []
    try:
        async for tweet in api.search("AAPL", limit=10):
            tweets.append(tweet)
            if len(tweets) >= 10:
                break
        
        print(f"✅ Found {len(tweets)} tweets")
        
        if len(tweets) > 0:
            print("\n📊 Sample Tweet:")
            sample = tweets[0]
            text = getattr(sample, 'rawContent', '') or getattr(sample, 'content', '')
            print(f"   Text: {text[:100]}...")
            print(f"   Likes: {getattr(sample, 'likeCount', 0)}")
            print(f"   Retweets: {getattr(sample, 'retweetCount', 0)}")
            return True
        else:
            print("⚠️  No tweets returned")
            return False
            
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_twscrape())
    print("\n" + "="*70)
    if result:
        print("✅ TWSCRAPE WORKING")
    else:
        print("❌ TWSCRAPE NEEDS SETUP")
    print("="*70)
