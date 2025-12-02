#!/usr/bin/env python3
"""
Direct RSS Feed Test - Check what articles are actually available
"""

import requests
import feedparser
from datetime import datetime

# RSS feeds to test
RSS_FEEDS = {
    "reuters": "https://reutersbest.com/feed/",
    "benzinga": "http://feeds.benzinga.com/benzinga",
    "fortune": "https://fortune.com/feed/fortune-feeds/?id=3230629",
    "financial_times": "https://www.ft.com/rss/home",
    "seeking_alpha": "https://seekingalpha.com/feed.xml",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "cnn_business": "http://rss.cnn.com/rss/money_latest.rss",
}


def test_rss_feed(name, url):
    """Test a single RSS feed and show what articles are available"""
    print(f"\n{'='*60}")
    print(f"Testing {name.upper()}: {url}")
    print(f"{'='*60}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            feed = feedparser.parse(response.content)
            print(f"Feed Title: {getattr(feed.feed, 'title', 'N/A')}")
            print(f"Feed Description: {getattr(feed.feed, 'description', 'N/A')}")
            print(f"Total Entries: {len(feed.entries)}")

            if feed.entries:
                print("\nFirst 3 Articles:")
                for i, entry in enumerate(feed.entries[:3]):
                    title = entry.get("title", "N/A") or "N/A"
                    print(f"\n  Article {i+1}:")
                    print(f"    Title: {title[:100] if len(title) > 100 else title}...")
                    print(f"    Published: {entry.get('published', 'N/A')}")
                    print(f"    Link: {entry.get('link', 'N/A')}")

                    # Check for AAPL mentions
                    text = f"{entry.get('title', '')} {entry.get('description', '') or entry.get('summary', '')}".lower()
                    apple_terms = ["apple", "aapl", "iphone", "ipad", "tim cook"]
                    if any(term in text for term in apple_terms):
                        print("    🍎 APPLE MENTION FOUND!")
            else:
                print("  No articles found in feed")
        else:
            print(f"❌ Failed with status {response.status_code}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


def main():
    print("🔍 RSS Feed Direct Test")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing {len(RSS_FEEDS)} RSS feeds...")

    for name, url in RSS_FEEDS.items():
        test_rss_feed(name, url)

    print(f"\n{'='*60}")
    print("Direct RSS Test Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
