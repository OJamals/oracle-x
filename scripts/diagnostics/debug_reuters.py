#!/usr/bin/env python3
"""
Debug Reuters RSS feed content
"""

import feedparser
import requests

def test_reuters_rss():
    """Test Reuters RSS feed directly"""
    url = "https://www.reuters.com/arc/outboundfeeds/rss/?outputType=xml"
    
    print(f"Testing Reuters RSS: {url}")
    
    # Try with requests first
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Content Length: {len(response.content)}")
        print(f"Content Type: {response.headers.get('content-type', 'Unknown')}")
        
        if response.status_code == 200:
            # Parse with feedparser
            feed = feedparser.parse(response.content)
            print(f"Feed Title: {getattr(feed.feed, 'title', 'No title')}")
            print(f"Feed Description: {getattr(feed.feed, 'description', 'No description')}")
            print(f"Number of entries: {len(feed.entries)}")
            
            if feed.entries:
                print(f"\nFirst 10 articles:")
                financial_count = 0
                for i, entry in enumerate(feed.entries[:10]):
                    print(f"\n  Article {i+1}:")
                    print(f"    Title: {entry.get('title', 'No title')}")
                    print(f"    Published: {entry.get('published', 'No date')}")
                    summary = entry.get('summary', 'No summary')
                    print(f"    Summary: {summary[:100] if summary else 'No summary'}...")
                    
                    # Check for financial keywords
                    text = f"{entry.get('title', '')} {entry.get('summary', '')}".lower()
                    financial_keywords = [
                        'stock', 'stocks', 'shares', 'earnings', 'revenue', 'profit', 'financial',
                        'market', 'trading', 'investor', 'investment', 'analyst',
                        'business', 'company', 'corporation', 'corporate', 'ceo', 'cfo'
                    ]
                    matching_keywords = [kw for kw in financial_keywords if kw in text]
                    print(f"    Financial keywords found: {matching_keywords}")
                    if matching_keywords:
                        financial_count += 1
                
                print(f"\nFinancial articles found in first 10: {financial_count}")
                
                # Search through all articles for financial content
                total_financial = 0
                for entry in feed.entries:
                    text = f"{entry.get('title', '')} {entry.get('summary', '')}".lower()
                    financial_keywords = [
                        'stock', 'stocks', 'shares', 'earnings', 'revenue', 'profit', 'financial',
                        'market', 'trading', 'investor', 'investment', 'analyst',
                        'business', 'company', 'corporation', 'corporate', 'ceo', 'cfo'
                    ]
                    if any(kw in text for kw in financial_keywords):
                        total_financial += 1
                
                print(f"Total financial articles found in all {len(feed.entries)}: {total_financial}")
            else:
                print("No entries found in feed")
        else:
            print(f"Failed to fetch RSS feed: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_reuters_rss()
